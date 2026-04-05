"""Session tracking database for Stelline."""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json


class SessionTracker:
    """SQLite-based tracking of session processing state."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize tracking database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS processed_sessions (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT UNIQUE NOT NULL,
                    session_file TEXT NOT NULL,
                    source TEXT NOT NULL,
                    model TEXT,
                    session_date TEXT,
                    processed_at TEXT NOT NULL,
                    memory_count INTEGER DEFAULT 0,
                    transcript_chars INTEGER,
                    status TEXT DEFAULT 'success',
                    error TEXT,
                    duration_seconds REAL
                );
                
                CREATE TABLE IF NOT EXISTS harvest_runs (
                    id INTEGER PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    sessions_processed INTEGER DEFAULT 0,
                    sessions_failed INTEGER DEFAULT 0,
                    total_memories INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    metadata TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_session_id ON processed_sessions(session_id);
                CREATE INDEX IF NOT EXISTS idx_source ON processed_sessions(source);
                CREATE INDEX IF NOT EXISTS idx_processed_at ON processed_sessions(processed_at);
                CREATE INDEX IF NOT EXISTS idx_harvest_started ON harvest_runs(started_at);
            """)
    
    def record_session(self, session_id: str, session_file: str, source: str,
                      model: Optional[str] = None, session_date: Optional[str] = None,
                      memory_count: int = 0, transcript_chars: Optional[int] = None,
                      status: str = "success", error: Optional[str] = None,
                      duration_seconds: Optional[float] = None) -> None:
        """Record session processing result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processed_sessions 
                (session_id, session_file, source, model, session_date, processed_at,
                 memory_count, transcript_chars, status, error, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, session_file, source, model, session_date,
                datetime.now().isoformat(), memory_count, transcript_chars,
                status, error, duration_seconds
            ))
    
    def is_processed(self, session_id: str) -> bool:
        """Check if session has been processed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM processed_sessions WHERE session_id = ?", 
                (session_id,)
            )
            return cursor.fetchone() is not None
    
    def count_processed(self, source: Optional[str] = None) -> int:
        """Count processed sessions by source."""
        with sqlite3.connect(self.db_path) as conn:
            if source:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM processed_sessions WHERE source = ?", (source,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM processed_sessions")
            return cursor.fetchone()[0]
    
    def start_harvest_run(self, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Start a new harvest run, return run ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO harvest_runs (started_at, metadata)
                VALUES (?, ?)
            """, (datetime.now().isoformat(), json.dumps(metadata) if metadata else None))
            return cursor.lastrowid
    
    def complete_harvest_run(self, run_id: int, sessions_processed: int,
                           sessions_failed: int, total_memories: int) -> None:
        """Mark harvest run as completed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE harvest_runs 
                SET completed_at = ?, sessions_processed = ?, sessions_failed = ?,
                    total_memories = ?, status = 'completed'
                WHERE id = ?
            """, (datetime.now().isoformat(), sessions_processed, sessions_failed,
                  total_memories, run_id))
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent harvest run history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM harvest_runs 
                ORDER BY started_at DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Overall stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_processed,
                    SUM(memory_count) as total_memories,
                    AVG(duration_seconds) as avg_duration,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count
                FROM processed_sessions
            """)
            overall = dict(cursor.fetchone())
            
            # By source stats
            cursor = conn.execute("""
                SELECT 
                    source,
                    COUNT(*) as processed,
                    SUM(memory_count) as memories,
                    AVG(duration_seconds) as avg_duration
                FROM processed_sessions
                GROUP BY source
                ORDER BY processed DESC
            """)
            by_source = [dict(row) for row in cursor.fetchall()]
            
            return {
                "overall": overall,
                "by_source": by_source
            }