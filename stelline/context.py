"""Context loader for project state and existing memories."""
from pathlib import Path
from typing import List, Dict, Optional


class ContextLoader:
    """Loads project context and existing memories for delta extraction."""
    
    def __init__(self, config):
        self.config = config
        
    def load_project_context(self) -> str:
        """Load current project context from context files."""
        context_parts = []
        
        # Load active projects
        projects_file = Path(self.config.context.projects_active).expanduser()
        if projects_file.exists():
            context_parts.append(f"=== ACTIVE PROJECTS ===\n{projects_file.read_text()}")
            
        # Load recent sessions summary  
        sessions_file = Path(self.config.context.sessions_recent).expanduser()
        if sessions_file.exists():
            context_parts.append(f"=== RECENT SESSIONS ===\n{sessions_file.read_text()}")
            
        # Load people context
        people_file = Path(self.config.context.people).expanduser()
        if people_file.exists():
            context_parts.append(f"=== PEOPLE ===\n{people_file.read_text()}")
        
        return "\n\n".join(context_parts)
    
    def search_existing_memories(self, transcript: str, memkoshi_instance) -> List[str]:
        """Search for related existing memories using VelociRAG."""
        memories = []
        seen = set()
        
        # Build a search query from transcript:
        # Take first 500 chars (usually contains session topic/intent)
        # plus last 300 chars (usually contains conclusions/outcomes)
        query = transcript[:500]
        if len(transcript) > 800:
            query += " " + transcript[-300:]
        
        # Single VelociRAG search — 4 layers handle the rest
        try:
            import logging
            log = logging.getLogger('stelline')
            logging.getLogger('memkoshi').setLevel(logging.ERROR)
            logging.getLogger('velocirag').setLevel(logging.ERROR)
            results = memkoshi_instance.search.search(query, self.config.max_recall_memories)
            for result in results:
                text = self._extract_memory_text(result)
                if text and text not in seen:
                    seen.add(text)
                    memories.append(text)
        except Exception as e:
            import logging
            logging.getLogger('stelline').debug(f"Memory search failed (may be first run): {e}")
        
        return memories[:self.config.max_recall_memories]
    
    def _extract_memory_text(self, memory_result) -> Optional[str]:
        """Extract text from Memkoshi memory/search result."""
        if isinstance(memory_result, dict):
            # Memkoshi search returns {title, abstract, category, ...}
            title = memory_result.get('title', '')
            abstract = memory_result.get('abstract', '')
            if title and abstract:
                return f"[{memory_result.get('category', '')}] {title}: {abstract}"
            # Fallback to other common fields
            return title or abstract or memory_result.get('text') or memory_result.get('content') or None
        if isinstance(memory_result, str):
            return memory_result
        return None