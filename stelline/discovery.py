"""Session file discovery and filtering."""
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SessionFile:
    """Represents a discovered session file."""
    path: Path
    session_id: str
    timestamp: datetime
    source: str
    size_bytes: int
    
    @classmethod
    def from_path(cls, path: Path, source: str) -> 'SessionFile':
        """Parse session file metadata from filename."""
        # Filename: 2024-12-23T12-00-00_session1.jsonl
        stem = path.stem
        if '_' in stem:
            timestamp_str, uuid_str = stem.split('_', 1)
        else:
            timestamp_str = stem
            uuid_str = stem
            
        try:
            # Convert filename timestamp format: 2024-12-23T15-00-00 -> 2024-12-23T15:00:00
            if 'T' in timestamp_str:
                date_part, time_part = timestamp_str.split('T')
                # Replace dashes in time part only
                time_part = time_part.replace('-', ':')
                timestamp_str = f"{date_part}T{time_part}"
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            timestamp = datetime.now()
        
        return cls(
            path=path,
            session_id=uuid_str,
            timestamp=timestamp,
            source=source,
            size_bytes=path.stat().st_size
        )


class SessionDiscovery:
    """Discovers and filters session files for processing."""
    
    def __init__(self, config, tracker):
        self.config = config
        self.tracker = tracker
        
    def discover_unprocessed(self, source=None) -> List[SessionFile]:
        """Find all unprocessed session files."""
        unprocessed = []
        
        sources = [s for s in self.config.sources if s.enabled]
        if source:
            sources = [s for s in sources if s.name == source]
            
        for source_config in sources:
            source_dir = self.config.session_dir / source_config.pattern
            if not source_dir.exists():
                continue
                
            for session_file in source_dir.glob("*.jsonl"):
                session = SessionFile.from_path(session_file, source_config.name)
                
                # Skip if already processed
                if self.tracker.is_processed(session.session_id):
                    continue
                
                # Skip tiny sessions (likely empty or error cases)
                if session.size_bytes < 1000:
                    continue

                # Skip sessions with too few user messages
                if self.config.min_session_messages > 0:
                    user_msgs = self._count_user_messages(session.path)
                    if user_msgs < self.config.min_session_messages:
                        continue

                unprocessed.append(session)
        
        # Sort by timestamp (oldest first)
        return sorted(unprocessed, key=lambda s: s.timestamp)
        
    def get_source_stats(self) -> Dict[str, Dict[str, int]]:
        """Get session count stats by source."""
        stats = {}
        
        for source_config in self.config.sources:
            if not source_config.enabled:
                continue
                
            source_dir = self.config.session_dir / source_config.pattern
            if not source_dir.exists():
                stats[source_config.name] = {"total": 0, "processed": 0, "unprocessed": 0}
                continue
                
            # Count all .jsonl files
            total = len(list(source_dir.glob("*.jsonl")))
            
            # Count processed sessions
            processed = self.tracker.count_processed(source_config.name)
            
            # Count harvestable (passes message filter)
            harvestable = total
            if self.config.min_session_messages > 0:
                harvestable = 0
                for f in source_dir.glob("*.jsonl"):
                    if f.stat().st_size < 1000:
                        continue
                    if self._count_user_messages(f) >= self.config.min_session_messages:
                        harvestable += 1

            stats[source_config.name] = {
                "total": total,
                "processed": processed,
                "unprocessed": harvestable - processed
            }
            
        return stats

    @staticmethod
    def _count_user_messages(path: Path) -> int:
        """Count user messages in a JSONL session file."""
        count = 0
        try:
            with open(path) as f:
                for line in f:
                    if '"role":"user"' in line or '"role": "user"' in line:
                        count += 1
        except (OSError, UnicodeDecodeError):
            pass
        return count