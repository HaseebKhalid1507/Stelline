"""Core Stelline intelligence extraction pipeline."""
import fcntl
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import StellineConfig
from .tracker import SessionTracker
from .context import ContextLoader
from .llm import LLMClient
from .parsers.pi import PiSessionParser

log = logging.getLogger("stelline")


class StellinePipeline:
    """Core Stelline intelligence extraction pipeline."""
    
    LOCK_FILE = Path("~/.config/stelline/stelline.lock").expanduser()
    
    def __init__(self, config: StellineConfig, tracker: SessionTracker,
                 context_loader: ContextLoader):
        self.config = config
        self.tracker = tracker
        self.context_loader = context_loader
        self._lock_fd = None
        self.llm_client = LLMClient(model=config.model, backend=config.backend)
    
    def acquire_lock(self) -> bool:
        """Acquire file lock. Returns False if another instance is running."""
        self.LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = open(self.LOCK_FILE, 'w')
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fd.write(str(os.getpid()))
            self._lock_fd.flush()
            return True
        except BlockingIOError:
            self._lock_fd.close()
            self._lock_fd = None
            return False
    
    def release_lock(self):
        """Release file lock."""
        if self._lock_fd:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None
            if self.LOCK_FILE.exists():
                self.LOCK_FILE.unlink()
        
    def process_session(self, session_file, dry_run: bool = False) -> Dict[str, Any]:
        """Process a single session through the intelligence pipeline."""
        sid = str(getattr(session_file, 'session_id', 'unknown'))[:8]
        size_kb = getattr(session_file, 'size_bytes', 0) // 1024 if isinstance(getattr(session_file, 'size_bytes', 0), int) else '?'
        fname = str(getattr(session_file, 'path', ''))[-19:]
        source = getattr(session_file, 'source', 'unknown')
        log.info(f"[{sid}] Processing session from {source} ({size_kb}KB, {fname})")

        start_time = time.time()

        try:
            # STEP 1: Parse session transcript
            transcript = self._parse_session(session_file.path)
            log.info(f"[{sid}] Parsed transcript: {len(transcript)} chars")

            if len(transcript) < 500:
                log.info(f"[{sid}] Skipped — transcript too short ({len(transcript)} chars)")
                return {"status": "skipped", "reason": "transcript too short"}

            # STEP 2: Search existing memories via VelociRAG
            memkoshi = self._get_memkoshi_instance(session_file.source)
            existing_memories = self.context_loader.search_existing_memories(
                transcript, memkoshi
            )
            log.info(f"[{sid}] Found {len(existing_memories)} existing memories for context")

            # STEP 3: Load project context
            project_context = self.context_loader.load_project_context()

            # STEP 4: Build prompt
            prompt = self._build_prompt(transcript, existing_memories, project_context)
            log.info(f"[{sid}] Prompt built: {len(prompt)} chars")

            if dry_run:
                log.info(f"[{sid}] Dry run — skipping LLM call")
                return {
                    "status": "dry_run",
                    "transcript_chars": len(transcript),
                    "existing_memories_count": len(existing_memories),
                    "prompt_chars": len(prompt)
                }

            # STEP 5: Call LLM → extract memories (chunked for large transcripts)
            if len(transcript) > self.CHUNK_THRESHOLD:
                memories = self._process_chunked(transcript, existing_memories, project_context)
            else:
                memories = self._call_llm_and_parse(prompt)
            backend_used = getattr(self.llm_client, 'last_backend_used', None) or "unknown"
            
            # STEP 6: Quality gate — filter before staging
            accepted, rejected = self._quality_gate(memories)
            if rejected:
                log.info(f"[{sid}] Quality gate: {len(accepted)} accepted, {len(rejected)} rejected")
            
            staged_count = 0
            for mem in accepted:
                memkoshi.storage.stage_memory(mem)
                staged_count += 1
            
            # Incrementally index new memories (not full reindex)
            if staged_count > 0:
                try:
                    for mem in accepted:
                        memkoshi.search.index_memory(mem)
                except Exception as e:
                    log.debug(f"[{sid}] Incremental index skipped: {e}")

            duration = time.time() - start_time
            log.info(f"[{sid}] ✅ Success — {staged_count} memories staged via {backend_used} in {duration:.0f}s")

            self.tracker.record_session(
                session_id=session_file.session_id,
                session_file=str(session_file.path),
                source=session_file.source,
                model=self.config.model,
                session_date=session_file.timestamp.isoformat(),
                memory_count=staged_count,
                transcript_chars=len(transcript),
                duration_seconds=duration
            )

            return {
                "status": "success",
                "session_id": session_file.session_id,
                "memories_extracted": staged_count,
                "transcript_chars": len(transcript),
                "duration_seconds": duration,
                "backend": backend_used,
            }

        except Exception as e:
            duration = time.time() - start_time
            log.error(f"[{sid}] ❌ Failed after {duration:.0f}s — {e}")

            self.tracker.record_session(
                session_id=session_file.session_id,
                session_file=str(session_file.path),
                source=session_file.source,
                session_date=session_file.timestamp.isoformat(),
                status="failed",
                error=str(e),
                duration_seconds=duration
            )

            return {
                "status": "failed",
                "session_id": session_file.session_id,
                "error": str(e),
                "duration_seconds": duration
            }
    
    def _parse_session(self, session_path: Path) -> str:
        """Parse session JSONL file to clean transcript."""
        parser = PiSessionParser()
        return parser.to_transcript(str(session_path))
    
    def _get_memkoshi_instance(self, source: str):
        """Get Memkoshi instance for the given source."""
        from memkoshi import Memkoshi
        
        # Check for source-specific storage override
        source_config = next((s for s in self.config.sources if s.name == source), None)
        storage_path = (source_config.memkoshi_storage if source_config and 
                       source_config.memkoshi_storage else self.config.memkoshi_storage)
        
        memkoshi = Memkoshi(
            storage_path=Path(storage_path).expanduser(),
            extractor="hybrid",  # Local only — Stelline IS the LLM, Memkoshi is just storage + search
        )
        memkoshi.init()
        return memkoshi
    
    # Max context to include — keep prompt focused on transcript
    MAX_PROJECT_CONTEXT = 3000
    MAX_EXISTING_MEMORIES = 5000
    
    def _build_prompt(self, transcript: str, existing_memories: List[str],
                      project_context: str) -> str:
        """Build the memory creation prompt. Transcript-heavy, context-light."""
        
        # Truncate transcript if needed
        if len(transcript) > self.config.max_transcript_chars:
            keep = self.config.max_transcript_chars // 2 - 100
            transcript = (transcript[:keep] + 
                         "\n\n[... TRUNCATED ...]\n\n" + 
                         transcript[-keep:])
        
        prompt_parts = []
        
        # Existing memories — compact, just titles to avoid duplication
        if existing_memories:
            # Keep it tight — just enough to prevent duplicates
            memories_text = "\n".join(f"• {memory}" for memory in existing_memories)
            if len(memories_text) > self.MAX_EXISTING_MEMORIES:
                memories_text = memories_text[:self.MAX_EXISTING_MEMORIES] + "\n[... more existing memories ...]" 
            prompt_parts.append(f"=== EXISTING MEMORIES (do not recreate these) ===\n{memories_text}")
        
        # Project context — heavily compressed, just active project names
        if project_context:
            compressed = self._compress_project_context(project_context)
            prompt_parts.append(f"=== ACTIVE PROJECTS (for reference only) ===\n{compressed}")
        
        # Session transcript — this is the main event
        prompt_parts.append(f"=== SESSION TRANSCRIPT ===\n{transcript}")
        
        prompt_parts.append("Extract memories from this session. Be granular — separate distinct facts, decisions, emotions, and patterns into individual memories. Include specific quotes and details. Return JSON with a \"memories\" array.")
        
        return "\n\n".join(prompt_parts)
    
    def _compress_project_context(self, context: str) -> str:
        """Compress project context to just project names and one-line status."""
        lines = context.split('\n')
        compressed = []
        for line in lines:
            # Grab ## headings (project names with status)
            if line.startswith('## '):
                # Take just the first line of each project
                compressed.append(line.replace('## ', '- '))
        
        result = '\n'.join(compressed)
        if len(result) > self.MAX_PROJECT_CONTEXT:
            result = result[:self.MAX_PROJECT_CONTEXT] + '\n[...]'
        
        return result if result else context[:self.MAX_PROJECT_CONTEXT]
    
    # Chunking
    CHUNK_THRESHOLD = 30000  # chars — split transcripts above this
    CHUNK_SIZE = 25000        # target chunk size
    
    def _process_chunked(self, transcript: str, existing_memories: List[str],
                         project_context: str) -> List:
        """Split large transcript into chunks, process each, dedup results."""
        chunks = self._split_transcript(transcript)
        log.info(f"Chunked transcript ({len(transcript)} chars) into {len(chunks)} chunks")
        
        all_memories = []
        seen_titles = set()
        
        for i, chunk in enumerate(chunks, 1):
            prompt = self._build_prompt(chunk, existing_memories, project_context)
            log.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            
            try:
                chunk_memories = self._call_llm_and_parse(prompt)
                # Dedup against previous chunks by title similarity
                for mem in chunk_memories:
                    title = str(getattr(mem, 'title', '')).lower().strip()
                    # Simple dedup — skip if first 50 chars of title match
                    title_key = title[:50]
                    if title_key not in seen_titles:
                        seen_titles.add(title_key)
                        all_memories.append(mem)
                    else:
                        log.debug(f"Dedup: skipped '{title[:60]}' (duplicate across chunks)")
            except Exception as e:
                log.warning(f"Chunk {i}/{len(chunks)} failed: {e}")
                continue
        
        return all_memories
    
    def _split_transcript(self, transcript: str) -> List[str]:
        """Split transcript at message boundaries, not mid-message."""
        if len(transcript) <= self.CHUNK_SIZE:
            return [transcript]
        
        chunks = []
        lines = transcript.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            # Message boundaries: lines starting with USER: or ASSISTANT: or JAWZ: etc
            is_boundary = any(line.startswith(prefix) for prefix in 
                            ['USER:', 'ASSISTANT:', 'JAWZ:', 'Human:', 'Assistant:'])
            
            if is_boundary and current_size >= self.CHUNK_SIZE:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += len(line) + 1
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    # Quality gate thresholds
    MIN_CONTENT_LENGTH = 50
    MIN_IMPORTANCE = 0.3
    MIN_TITLE_LENGTH = 10
    MIN_TOPIC_LENGTH = 5
    VALID_CATEGORIES = {'events', 'preferences', 'entities', 'cases', 'patterns'}
    
    def _quality_gate(self, memories: List) -> tuple:
        """Filter memories by quality. Returns (accepted, rejected) lists."""
        accepted = []
        rejected = []
        
        for mem in memories:
            reasons = []
            
            try:
                content = str(getattr(mem, 'content', '') or '')
                title = str(getattr(mem, 'title', '') or '')
                topic = str(getattr(mem, 'topic', '') or '')
                importance = float(getattr(mem, 'importance', 0.5) or 0.5)
                category = getattr(mem, 'category', None)
            except (TypeError, ValueError):
                accepted.append(mem)  # can't inspect, let it through
                continue
            
            if len(content) < self.MIN_CONTENT_LENGTH:
                reasons.append(f'content too short ({len(content)}<{self.MIN_CONTENT_LENGTH})')
            if importance < self.MIN_IMPORTANCE:
                reasons.append(f'importance too low ({importance}<{self.MIN_IMPORTANCE})')
            if len(title) < self.MIN_TITLE_LENGTH:
                reasons.append(f'title too short ({len(title)}<{self.MIN_TITLE_LENGTH})')
            if len(topic) < self.MIN_TOPIC_LENGTH:
                reasons.append(f'topic too short ({len(topic)}<{self.MIN_TOPIC_LENGTH})')
            
            if reasons:
                log.debug(f"Quality gate rejected: {title[:60]} — {', '.join(reasons)}")
                rejected.append(mem)
            else:
                accepted.append(mem)
        
        return accepted, rejected

    def _call_llm_and_parse(self, prompt: str) -> List:
        """Call LLM with prompt and parse response into Memory objects."""
        return self.llm_client.extract_memories(prompt)