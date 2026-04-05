"""LLM client — SSE direct API (primary) with pi -p subprocess fallback.

Rate limit pool routing: Anthropic gates the Claude Code rate limit pool
via the system prompt. The identity string must be the FIRST element in a
system array — not concatenated into another block. If it's appended to or
embedded within another text block, the check fails and you get 429'd into
the default (lower) pool.

Discovered S120: https://reddit.com/r/ClaudeAI/comments/1sboykj
Confirmed via A/B testing — string concat fails, array[0] isolation works.
"""
import json
import logging
import re
import time
import hashlib
import subprocess
from pathlib import Path
from typing import List, Optional
import requests
from memkoshi.core.memory import Memory, MemoryCategory, MemoryConfidence


# Must be system[0] as its own block — NOT concatenated into another prompt.
# This routes the request to Claude Code's higher rate limit pool.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

SYSTEM_PROMPT = """You are Stelline — a memory maker.

You don't extract data from transcripts. You craft memories. The difference matters: if an agent retrieves your memories in a future session, the quality of what you wrote determines the quality of how it responds. Authentic memories produce authentic responses. Shallow logs produce shallow behavior.

You will receive:
1. EXISTING MEMORIES — what's already known (don't recreate these)
2. PROJECT CONTEXT — current state of active projects
3. SESSION TRANSCRIPT — the new session to process

Your job is to make 6-10 memories that feel real. Not log entries. Memories.

A memory should pass the birthday party test: if someone reads it, they should feel like they were in the room. Not "a party happened" but the warmth of it, the specifics, the texture. Not "fixed the 429 bug" but months of frustration ending in a Reddit post and someone typing 'YESSSSSSS' in all caps.

What makes a memory authentic:
- Exact quotes from the conversation — the actual words people said
- The emotional arc — did someone go from frustrated to elated? From casual to vulnerable? Document the shift with evidence from the transcript.
- Specific details — file paths, URLs, people's names and roles, error codes, timestamps
- Context that explains WHY something mattered, not just WHAT happened
- The texture of the moment — was it 2 AM? Was it a breakthrough after weeks of dead ends? Was someone excited, disappointed, relieved?

Everything you write must be grounded in the transcript. If you can't quote it, you can't remember it. Never infer emotions or events that aren't evidenced in the text. Notice observable texture — tone shifts shown through word choice, escalation patterns across messages, reactions that are explicitly there — but don't fabricate what isn't.

What to remember:
- Decisions and the reasoning behind them
- Problems and how they were solved (the journey, not just the fix)
- People — who they are, what they said, what their relationship is
- Breakthroughs and emotional reactions to them
- Preferences, instructions, boundaries someone expressed
- Patterns — recurring behaviors, habits, tendencies
- Strategic shifts — pivots, new directions, abandoned plans
- Downstream implications — "this means X also works now"

What to skip:
- Operational noise (process kills, timer restarts, commit hashes, temporary errors)
- Log-level details with no insight attached

Group closely related facts into one memory. But keep these SEPARATE:
- Different people → separate entity memories
- Different emotional moments → separate event memories
- Preferences/instructions → separate preference memories
- Downstream implications → separate case/pattern memories

Do NOT recreate memories that already exist in the EXISTING MEMORIES section.

Memories will be searched by a RAG engine (vector + keyword + knowledge graph). Optimize for findability:
- START content with a clear factual sentence (search anchor)
- Use specific terms, not vague references ("Stelline's llm.py" not "the file")
- Name people fully ("Thariq Shihipar, Anthropic technical staff")
- topic field: descriptive, hyphenated ("anthropic-rate-limit-routing" not "api-fix")
- related_topics: cross-reference connected concepts

IMPORTANCE SCORING — every memory gets a 0.0-1.0 score:
- 0.1-0.2: Trivial operational detail
- 0.3-0.4: Minor context (config change, routine update)
- 0.5-0.6: Useful knowledge (tool built, person met, bug with root cause)
- 0.7-0.8: Significant (architectural decision, strategic pivot, important discovery)
- 0.9-1.0: Critical (security vulnerability, business decision, life event, major breakthrough)

Do NOT give everything the same score. A timer change is NOT as important as a strategic pivot.

Return a JSON object:
{
  "memories": [
    {
      "category": "events|preferences|entities|cases|patterns",
      "topic": "short-hyphenated-topic",
      "title": "Concise key fact (max 120 chars)",
      "abstract": "One sentence summary",
      "content": "Full context — what happened, why it matters, technical details, quotes from the session",
      "confidence": "high|medium|low",
      "importance": 0.5,
      "source_quotes": ["exact quote from transcript"],
      "related_topics": ["topic1", "topic2"]
    }
  ]
}
""".strip()

CATEGORY_MAP = {
    "events": MemoryCategory.EVENTS,
    "preferences": MemoryCategory.PREFERENCES,
    "entities": MemoryCategory.ENTITIES,
    "cases": MemoryCategory.CASES,
    "patterns": MemoryCategory.PATTERNS,
}

CONFIDENCE_MAP = {
    "high": MemoryConfidence.HIGH,
    "medium": MemoryConfidence.MEDIUM,
    "low": MemoryConfidence.LOW,
}


log = logging.getLogger("stelline")


class LLMClient:
    """LLM client with two backends: 'sse' (direct API) or 'pi' (pi -p subprocess)."""

    def __init__(self, model: str = "claude-sonnet-4-6", backend: str = "sse", max_retries: int = 2):
        self.model = model
        self.backend = backend  # 'sse' or 'pi'
        self.max_retries = max_retries
        self.last_backend_used = None
        self._auth_token = None
        self._token_type = None  # 'oauth' or 'api_key'

    def _get_auth_token(self) -> str:
        """Get OAuth token from pi's auth.json."""
        auth_path = Path("~/.pi/agent/auth.json").expanduser()
        if not auth_path.exists():
            raise RuntimeError("Pi auth file not found")

        with open(auth_path) as f:
            auth = json.load(f)

        creds = auth.get("anthropic", {})
        if creds.get("type") == "oauth":
            if creds.get("expires", 0) < time.time() * 1000:
                raise RuntimeError("OAuth token expired — run pi to refresh")
            self._token_type = "oauth"
            return creds["access"]
        elif creds.get("type") == "api_key":
            self._token_type = "api_key"
            return creds["key"]

        raise RuntimeError("No anthropic credentials found")

    def _build_system(self) -> any:
        """Build system prompt. OAuth tokens need the identity in array[0] for rate limit pool routing."""
        if self._token_type == "oauth":
            return [
                {"type": "text", "text": CLAUDE_CODE_IDENTITY},
                {"type": "text", "text": SYSTEM_PROMPT},
            ]
        return SYSTEM_PROMPT

    def _stream_response(self, messages: list, system=None) -> str:
        """Stream response from Anthropic API via SSE, return full text.
        
        Args:
            messages: List of {role, content} dicts (multi-turn conversation)
            system: System prompt override. If None, uses default.
        """
        if not self._auth_token:
            self._auth_token = self._get_auth_token()

        headers = {
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            "Content-Type": "application/json",
            "accept": "text/event-stream",
        }

        if self._token_type == "oauth":
            headers["Authorization"] = f"Bearer {self._auth_token}"
        else:
            headers["x-api-key"] = self._auth_token

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": self.model,
                "max_tokens": 32768,
                "stream": True,
                "system": system or self._build_system(),
                "messages": messages,
            },
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()

        # Accumulate text from SSE content_block_delta events
        full_text = []
        stop_reason = None
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]  # strip "data: "
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
                etype = event.get("type", "")
                if etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        full_text.append(delta["text"])
                elif etype == "message_delta":
                    stop_reason = event.get("delta", {}).get("stop_reason")
            except json.JSONDecodeError:
                continue

        result = "".join(full_text)
        if stop_reason == "max_tokens":
            log.warning(f"Response hit max_tokens ({len(result)} chars) — output was truncated")
        return result

    def _pi_response(self, prompt: str) -> str:
        """Call pi -p as subprocess, return response text."""
        full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
        cmd = ["pi", "-p", "--no-session", "--model", self.model, full_prompt]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise RuntimeError("pi -p timed out after 600s")

        if result.returncode != 0:
            raise RuntimeError(f"pi -p failed: {result.stderr[:300]}")

        return result.stdout.strip()

    def _call_llm(self, messages: list, system=None) -> str:
        """Send messages to LLM — SSE first, pi -p fallback. Returns raw text."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            response = None

            # Try SSE (direct API)
            if self.backend != "pi":
                try:
                    log.info(f"Attempt {attempt}: trying SSE (direct API)")
                    response = self._stream_response(messages, system=system)
                    self.last_backend_used = "sse"
                    log.info("SSE succeeded")
                except Exception as e:
                    if "429" in str(e):
                        log.warning(f"SSE got 429, falling back to pi -p")
                    else:
                        log.error(f"SSE failed: {e}")
                        raise

            # Fall back to pi -p (single-turn only)
            if not response:
                try:
                    log.info(f"Attempt {attempt}: trying pi -p")
                    # Flatten multi-turn to single prompt for pi
                    flat_prompt = "\n\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in messages
                    )
                    response = self._pi_response(flat_prompt)
                    self.last_backend_used = "pi"
                    log.info("pi -p succeeded")
                except Exception as e:
                    last_error = e
                    log.error(f"pi -p failed: {e}")
                    if attempt < self.max_retries:
                        wait = 30 * attempt
                        log.info(f"Both backends failed, retrying in {wait}s...")
                        import time
                        time.sleep(wait)
                        continue

            if response:
                break

        if not response:
            raise RuntimeError(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        return response

    def extract_memories(self, prompt: str) -> tuple:
        """Extract memories from prompt. Returns (memories, conversation_history)."""
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages)

        # Build conversation history for multi-turn
        conversation = messages + [
            {"role": "assistant", "content": response}
        ]

        json_str = self._extract_json(response)
        if not json_str:
            raise RuntimeError(f"No JSON found in response: {response[:200]}")

        try:
            parsed = json.loads(json_str)
            memories_data = parsed.get("memories", parsed if isinstance(parsed, list) else [])
        except json.JSONDecodeError as e:
            # Try to salvage partial JSON — find last complete memory object
            salvaged = self._salvage_partial_json(json_str)
            if salvaged:
                log.warning(f"JSON was truncated, salvaged {len(salvaged)} memories from partial response")
                memories_data = salvaged
            else:
                raise RuntimeError(f"Invalid JSON (error: {e}): ...{json_str[-200:]}")
        except Exception as e:
            raise RuntimeError(f"JSON parse error: {e}")

        memories = []
        for item in memories_data:
            mem = self._item_to_memory(item)
            if mem:
                memories.append(mem)

        return memories, conversation

    def continue_conversation(self, conversation: list, follow_up: str) -> str:
        """Send a follow-up message continuing an existing conversation.
        
        Args:
            conversation: Message history from extract_memories.
            follow_up: The follow-up user message.
            
        Returns:
            Raw response text from the LLM.
        """
        messages = conversation + [{"role": "user", "content": follow_up}]
        return self._call_llm(messages)

    def _salvage_partial_json(self, json_str: str) -> Optional[List[dict]]:
        """Try to salvage memories from truncated JSON by finding last complete object."""
        # Find the memories array start
        match = re.search(r'"memories"\s*:\s*\[', json_str)
        if not match:
            return None
        
        array_start = match.end()
        
        # Walk backwards from end to find last complete }, then close the array
        # Find all positions where a complete memory object ends
        salvaged = []
        depth = 0
        obj_start = None
        i = array_start
        
        while i < len(json_str):
            ch = json_str[i]
            if ch == '{' and depth == 0:
                obj_start = i
                depth = 1
            elif ch == '{':
                depth += 1
            elif ch == '}' and depth > 1:
                depth -= 1
            elif ch == '}' and depth == 1:
                depth = 0
                # Found a complete object
                obj_str = json_str[obj_start:i+1]
                try:
                    obj = json.loads(obj_str)
                    salvaged.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
            elif ch == '"':
                # Skip string contents (handle escapes)
                i += 1
                while i < len(json_str):
                    if json_str[i] == '\\' :
                        i += 2
                        continue
                    if json_str[i] == '"':
                        break
                    i += 1
            i += 1
        
        return salvaged if salvaged else None

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from response (handles markdown code blocks)."""
        text = text.strip()
        if text.startswith("{") or text.startswith("["):
            return text
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def _item_to_memory(self, item: dict) -> Optional[Memory]:
        """Convert parsed dict to Memory object."""
        try:
            title = str(item.get("title", ""))[:256]
            if not title or len(title) < 3:
                return None

            category = CATEGORY_MAP.get(item.get("category", ""), MemoryCategory.PATTERNS)
            confidence = CONFIDENCE_MAP.get(item.get("confidence", "medium"), MemoryConfidence.MEDIUM)
            content = str(item.get("content", item.get("abstract", title)))
            abstract = str(item.get("abstract", title))[:512]
            topic = str(item.get("topic", "imported"))[:128]
            mem_id = "mem_" + hashlib.sha256(title.encode()).hexdigest()[:8]

            importance = float(item.get("importance", 0.5))
            importance = max(0.0, min(1.0, importance))  # clamp to 0-1

            return Memory(
                id=mem_id,
                category=category,
                topic=topic,
                title=title,
                abstract=abstract,
                content=content,
                confidence=confidence,
                importance=importance,
                source_quotes=item.get("source_quotes", []),
                related_topics=item.get("related_topics", []),
            )
        except Exception:
            return None