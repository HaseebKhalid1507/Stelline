"""Tests for tracker module."""
import sqlite3
import pytest
from pathlib import Path


def test_db_initializes_with_correct_tables(tmp_path):
    """SessionTracker should initialize database with required tables."""
    from stelline.tracker import SessionTracker
    
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Check database file exists
    assert db_path.exists()
    
    # Check tables exist with correct schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check processed_sessions table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='processed_sessions'")
    assert cursor.fetchone() is not None
    
    # Check harvest_runs table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='harvest_runs'")
    assert cursor.fetchone() is not None
    
    # Check indexes exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
    indexes = [row[0] for row in cursor.fetchall()]
    expected_indexes = ['idx_session_id', 'idx_source', 'idx_processed_at', 'idx_harvest_started']
    
    for idx in expected_indexes:
        assert idx in indexes
    
    conn.close()


def test_record_session_stores_and_retrieves(tmp_path):
    """record_session should store session data that can be retrieved."""
    from stelline.tracker import SessionTracker
    from datetime import datetime
    
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Record a session
    tracker.record_session(
        session_id="test_session_123",
        session_file="/tmp/test.jsonl",
        source="test_source",
        model="claude-3-haiku",
        session_date="2024-12-23T12:00:00Z",
        memory_count=5,
        transcript_chars=1000,
        status="success",
        duration_seconds=30.5
    )
    
    # Verify data is stored
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM processed_sessions WHERE session_id = ?", ("test_session_123",))
    result = cursor.fetchone()
    
    assert result is not None
    assert result[1] == "test_session_123"  # session_id
    assert result[2] == "/tmp/test.jsonl"   # session_file
    assert result[3] == "test_source"       # source
    assert result[4] == "claude-3-haiku"   # model
    assert result[5] == "2024-12-23T12:00:00Z"  # session_date
    assert result[7] == 5                   # memory_count
    assert result[8] == 1000               # transcript_chars
    assert result[9] == "success"          # status
    assert result[11] == 30.5              # duration_seconds
    
    conn.close()


def test_is_processed_returns_true_for_recorded_sessions(tmp_path):
    """is_processed should return True for sessions that have been recorded."""
    from stelline.tracker import SessionTracker
    
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Record a session
    tracker.record_session(
        session_id="recorded_session",
        session_file="/tmp/test.jsonl",
        source="test_source"
    )
    
    # Check it's marked as processed
    assert tracker.is_processed("recorded_session") is True


def test_is_processed_returns_false_for_unknown_sessions(tmp_path):
    """is_processed should return False for sessions that haven't been recorded."""
    from stelline.tracker import SessionTracker
    
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Check unknown session is not processed
    assert tracker.is_processed("unknown_session") is False


def test_count_processed_by_source(tmp_path):
    """count_processed should return correct counts by source."""
    from stelline.tracker import SessionTracker
    
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Record sessions from different sources
    tracker.record_session("session1", "/tmp/s1.jsonl", "jawz")
    tracker.record_session("session2", "/tmp/s2.jsonl", "jawz") 
    tracker.record_session("session3", "/tmp/s3.jsonl", "dexter")
    
    # Test counts by source
    assert tracker.count_processed("jawz") == 2
    assert tracker.count_processed("dexter") == 1
    assert tracker.count_processed("unknown") == 0
    
    # Test total count (no source filter)
    assert tracker.count_processed() == 3


def test_start_and_complete_harvest_run(tmp_path):
    """start_harvest_run and complete_harvest_run should track harvest runs."""
    from stelline.tracker import SessionTracker
    import sqlite3
    
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Start a harvest run
    run_id = tracker.start_harvest_run({"batch_size": 5})
    
    # Verify run was created
    assert isinstance(run_id, int)
    assert run_id > 0
    
    # Check database state
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM harvest_runs WHERE id = ?", (run_id,))
    result = cursor.fetchone()
    
    assert result is not None
    assert result[0] == run_id  # id
    assert result[1] is not None  # started_at
    assert result[2] is None  # completed_at (not completed yet)
    assert result[6] == "running"  # status
    
    # Complete the harvest run
    tracker.complete_harvest_run(run_id, sessions_processed=10, sessions_failed=1, total_memories=25)
    
    # Check updated state
    cursor.execute("SELECT * FROM harvest_runs WHERE id = ?", (run_id,))
    result = cursor.fetchone()
    
    assert result[2] is not None  # completed_at
    assert result[3] == 10  # sessions_processed
    assert result[4] == 1   # sessions_failed
    assert result[5] == 25  # total_memories
    assert result[6] == "completed"  # status
    
    conn.close()


def test_get_recent_runs(tmp_path):
    """get_recent_runs should return recent harvest run history."""
    from stelline.tracker import SessionTracker
    
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Create some harvest runs
    run1 = tracker.start_harvest_run({"source": "jawz"})
    run2 = tracker.start_harvest_run({"source": "dexter"})
    tracker.complete_harvest_run(run1, 5, 1, 10)
    
    # Get recent runs
    runs = tracker.get_recent_runs(limit=5)
    
    assert isinstance(runs, list)
    assert len(runs) == 2
    
    # Should be sorted by started_at desc (newest first)
    assert runs[0]['id'] == run2  # more recent
    assert runs[1]['id'] == run1  # older
    
    # Check data structure
    run = runs[0]
    assert 'id' in run
    assert 'started_at' in run
    assert 'status' in run
    assert 'sessions_processed' in run


def test_get_stats_aggregation(tmp_path):
    """get_stats should return aggregated statistics."""
    from stelline.tracker import SessionTracker
    
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Record sessions from different sources
    tracker.record_session("s1", "/tmp/s1.jsonl", "jawz", memory_count=5, duration_seconds=10.0)
    tracker.record_session("s2", "/tmp/s2.jsonl", "jawz", memory_count=3, duration_seconds=15.0)
    tracker.record_session("s3", "/tmp/s3.jsonl", "dexter", memory_count=7, duration_seconds=20.0)
    tracker.record_session("s4", "/tmp/s4.jsonl", "dexter", status="failed")
    
    stats = tracker.get_stats()
    
    # Check structure
    assert isinstance(stats, dict)
    assert 'overall' in stats
    assert 'by_source' in stats
    
    # Check overall stats
    overall = stats['overall']
    assert overall['total_processed'] == 4
    assert overall['total_memories'] == 15  # 5+3+7+0
    assert overall['failed_count'] == 1
    assert overall['avg_duration'] == 15.0  # (10+15+20+0)/4
    
    # Check by_source stats
    by_source = stats['by_source']
    assert len(by_source) == 2
    
    # Find jawz stats
    jawz_stats = next(s for s in by_source if s['source'] == 'jawz')
    assert jawz_stats['processed'] == 2
    assert jawz_stats['memories'] == 8  # 5+3
    
    # Find dexter stats
    dexter_stats = next(s for s in by_source if s['source'] == 'dexter')
    assert dexter_stats['processed'] == 2
    assert dexter_stats['memories'] == 7  # 7+0