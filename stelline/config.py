"""Configuration management for Stelline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
import yaml


@dataclass
class SourceConfig:
    """Configuration for a session source."""
    name: str
    pattern: str
    enabled: bool = True
    memkoshi_storage: Optional[str] = None


@dataclass
class ContextConfig:
    """Configuration for optional context files that enrich memory extraction."""
    projects_active: str = os.environ.get("STELLINE_CTX_PROJECTS", "")
    sessions_recent: str = os.environ.get("STELLINE_CTX_SESSIONS", "")
    people: str = os.environ.get("STELLINE_CTX_PEOPLE", "")


@dataclass
class StellineConfig:
    """Stelline configuration."""
    session_dir: Path = field(default_factory=lambda: Path("~/.pi/agent/sessions").expanduser())
    sources: List[SourceConfig] = field(default_factory=lambda: [
        SourceConfig("default", os.environ.get("STELLINE_SOURCE", "")),
    ])
    batch_size: int = 5
    model: str = "claude-sonnet-4-6"
    backend: str = "sse"  # 'sse' (direct API) or 'pi' (pi -p subprocess)
    context: ContextConfig = field(default_factory=ContextConfig)
    max_recall_memories: int = 20
    max_transcript_chars: int = 50000
    min_session_messages: int = 5
    memkoshi_storage: str = "~/.stelline/memkoshi"
    db_path: str = "~/.config/stelline/stelline.db"
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        session_dir = Path(
            os.environ.get("STELLINE_SESSION_DIR", "~/.pi/agent/sessions")
        ).expanduser()
        
        batch_size = int(os.environ.get("STELLINE_BATCH_SIZE", "5"))
        
        return cls(
            session_dir=session_dir,
            batch_size=batch_size,
        )
    
    @classmethod
    def load(cls, config_path: str):
        """Load config from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate and convert types
        if 'session_dir' in data:
            data['session_dir'] = Path(data['session_dir']).expanduser()
        
        if 'batch_size' in data:
            if not isinstance(data['batch_size'], int):
                try:
                    data['batch_size'] = int(data['batch_size'])
                except (ValueError, TypeError):
                    raise ValueError(f"batch_size must be an integer, got {data['batch_size']}")
        
        return cls(**data)