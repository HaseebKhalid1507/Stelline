"""Tests for discovery module."""
import pytest
from pathlib import Path


def test_discover_finds_jsonl_files_in_session_directory(tmp_path):
    """Discovery should find .jsonl files in session directory."""
    from stelline.discovery import SessionDiscovery
    from stelline.config import StellineConfig, SourceConfig
    from stelline.tracker import SessionTracker
    
    # Create test session directory structure
    session_dir = tmp_path / "sessions"
    source_dir = session_dir / "test_source"
    source_dir.mkdir(parents=True)
    
    # Create test session files (1500 bytes each to pass size filter)
    content = "x" * 1500
    (source_dir / "2024-12-23T12-00-00_session1.jsonl").write_text(content)
    (source_dir / "2024-12-23T13-00-00_session2.jsonl").write_text(content)
    (source_dir / "not_session.txt").write_text("not a session")
    
    # Create config with test session directory (min_session_messages=0 for plain text fixtures)
    config = StellineConfig(
        session_dir=session_dir,
        sources=[SourceConfig("test_source", "test_source", True)],
        min_session_messages=0,
    )
    
    # Create tracker (empty for now)
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Test discovery
    discovery = SessionDiscovery(config, tracker)
    unprocessed = discovery.discover_unprocessed()
    
    # Should find 2 .jsonl files
    assert len(unprocessed) == 2
    
    # Should be SessionFile objects
    session_files = unprocessed
    for session_file in session_files:
        assert hasattr(session_file, 'path')
        assert hasattr(session_file, 'session_id')
        assert hasattr(session_file, 'source')
        assert session_file.source == "test_source"
        assert session_file.path.suffix == ".jsonl"


def test_discover_skips_already_processed_sessions(tmp_path):
    """Discovery should skip sessions already marked as processed."""
    from stelline.discovery import SessionDiscovery
    from stelline.config import StellineConfig, SourceConfig
    from stelline.tracker import SessionTracker
    
    # Create test session directory
    session_dir = tmp_path / "sessions"
    source_dir = session_dir / "test_source"
    source_dir.mkdir(parents=True)
    
    # Create test session files (1500 bytes each to pass size filter)
    content = "x" * 1500
    (source_dir / "2024-12-23T12-00-00_processed.jsonl").write_text(content)
    (source_dir / "2024-12-23T13-00-00_unprocessed.jsonl").write_text(content)
    
    # Create config (min_session_messages=0 for plain text fixtures)
    config = StellineConfig(
        session_dir=session_dir,
        sources=[SourceConfig("test_source", "test_source", True)],
        min_session_messages=0,
    )
    
    # Create tracker and mark one session as processed
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    tracker.record_session("processed", "/tmp/test.jsonl", "test_source")
    
    # Test discovery
    discovery = SessionDiscovery(config, tracker)
    unprocessed = discovery.discover_unprocessed()
    
    # Should only find the unprocessed one
    assert len(unprocessed) == 1
    assert unprocessed[0].session_id == "unprocessed"


def test_discover_filters_by_source(tmp_path):
    """Discovery should filter by source when specified."""
    from stelline.discovery import SessionDiscovery
    from stelline.config import StellineConfig, SourceConfig
    from stelline.tracker import SessionTracker
    
    # Create test session directories for two sources
    session_dir = tmp_path / "sessions"
    jawz_dir = session_dir / "jawz_source"
    dexter_dir = session_dir / "dexter_source"
    jawz_dir.mkdir(parents=True)
    dexter_dir.mkdir(parents=True)
    
    # Create session files in both sources (1500 bytes each to pass size filter)
    content = "x" * 1500
    (jawz_dir / "2024-12-23T12-00-00_jawz1.jsonl").write_text(content)
    (jawz_dir / "2024-12-23T13-00-00_jawz2.jsonl").write_text(content)
    (dexter_dir / "2024-12-23T14-00-00_dexter1.jsonl").write_text(content)
    
    # Create config with both sources (min_session_messages=0 for plain text fixtures)
    config = StellineConfig(
        session_dir=session_dir,
        sources=[
            SourceConfig("jawz", "jawz_source", True),
            SourceConfig("dexter", "dexter_source", True)
        ],
        min_session_messages=0,
    )
    
    # Create tracker
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Test discovery with source filter
    discovery = SessionDiscovery(config, tracker)
    
    # All sources
    all_sessions = discovery.discover_unprocessed()
    assert len(all_sessions) == 3
    
    # Filter by jawz only
    jawz_sessions = discovery.discover_unprocessed(source="jawz")
    assert len(jawz_sessions) == 2
    assert all(s.source == "jawz" for s in jawz_sessions)
    
    # Filter by dexter only
    dexter_sessions = discovery.discover_unprocessed(source="dexter")
    assert len(dexter_sessions) == 1
    assert dexter_sessions[0].source == "dexter"


def test_discover_skips_tiny_files(tmp_path):
    """Discovery should skip files smaller than 1000 bytes."""
    from stelline.discovery import SessionDiscovery
    from stelline.config import StellineConfig, SourceConfig
    from stelline.tracker import SessionTracker
    
    # Create test session directory
    session_dir = tmp_path / "sessions"
    source_dir = session_dir / "test_source"
    source_dir.mkdir(parents=True)
    
    # Create tiny file (< 1000 bytes)
    tiny_file = source_dir / "2024-12-23T12-00-00_tiny.jsonl"
    tiny_file.write_text("tiny")  # 4 bytes
    
    # Create normal file (> 1000 bytes)
    normal_file = source_dir / "2024-12-23T13-00-00_normal.jsonl"
    normal_file.write_text("x" * 1500)  # 1500 bytes
    
    # Create config (min_session_messages=0 for plain text fixtures)
    config = StellineConfig(
        session_dir=session_dir,
        sources=[SourceConfig("test_source", "test_source", True)],
        min_session_messages=0,
    )
    
    # Create tracker
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Test discovery
    discovery = SessionDiscovery(config, tracker)
    unprocessed = discovery.discover_unprocessed()
    
    # Should only find the normal file, not the tiny one
    assert len(unprocessed) == 1
    assert unprocessed[0].session_id == "normal"


def test_discover_sorts_oldest_first(tmp_path):
    """Discovery should sort sessions by timestamp, oldest first."""
    from stelline.discovery import SessionDiscovery
    from stelline.config import StellineConfig, SourceConfig
    from stelline.tracker import SessionTracker
    
    # Create test session directory
    session_dir = tmp_path / "sessions"
    source_dir = session_dir / "test_source"
    source_dir.mkdir(parents=True)
    
    # Create files with different timestamps (1500 bytes each to pass size filter)
    content = "x" * 1500
    (source_dir / "2024-12-23T15-00-00_newest.jsonl").write_text(content)   # 15:00 
    (source_dir / "2024-12-23T12-00-00_oldest.jsonl").write_text(content)   # 12:00
    (source_dir / "2024-12-23T13-00-00_middle.jsonl").write_text(content)   # 13:00
    
    # Create config (min_session_messages=0 for plain text fixtures)
    config = StellineConfig(
        session_dir=session_dir,
        sources=[SourceConfig("test_source", "test_source", True)],
        min_session_messages=0,
    )
    
    # Create tracker
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    
    # Test discovery
    discovery = SessionDiscovery(config, tracker)
    unprocessed = discovery.discover_unprocessed()
    
    # Should be sorted by timestamp, oldest first
    assert len(unprocessed) == 3
    assert unprocessed[0].session_id == "oldest"   # 12:00
    assert unprocessed[1].session_id == "middle"   # 13:00
    assert unprocessed[2].session_id == "newest"   # 15:00


def test_get_source_stats_returns_correct_counts(tmp_path):
    """get_source_stats should return correct session counts by source."""
    from stelline.discovery import SessionDiscovery
    from stelline.config import StellineConfig, SourceConfig
    from stelline.tracker import SessionTracker
    
    # Create test session directories for multiple sources
    session_dir = tmp_path / "sessions"
    jawz_dir = session_dir / "jawz_source"
    dexter_dir = session_dir / "dexter_source"
    jawz_dir.mkdir(parents=True)
    dexter_dir.mkdir(parents=True)
    
    # Create session files (1500 bytes each to pass size filter)
    content = "x" * 1500
    (jawz_dir / "2024-12-23T12-00-00_jawz1.jsonl").write_text(content)
    (jawz_dir / "2024-12-23T13-00-00_jawz2.jsonl").write_text(content)
    (jawz_dir / "2024-12-23T14-00-00_jawz3.jsonl").write_text(content)
    (dexter_dir / "2024-12-23T15-00-00_dexter1.jsonl").write_text(content)
    (dexter_dir / "2024-12-23T16-00-00_dexter2.jsonl").write_text(content)
    
    # Create config with both sources (min_session_messages=0 for plain text fixtures)
    config = StellineConfig(
        session_dir=session_dir,
        sources=[
            SourceConfig("jawz", "jawz_source", True),
            SourceConfig("dexter", "dexter_source", True)
        ],
        min_session_messages=0,
    )
    
    # Create tracker and mark some sessions as processed
    db_path = tmp_path / "test.db"
    tracker = SessionTracker(str(db_path))
    tracker.record_session("jawz1", "/tmp/jawz1.jsonl", "jawz")
    tracker.record_session("dexter1", "/tmp/dexter1.jsonl", "dexter")
    
    # Test get_source_stats
    discovery = SessionDiscovery(config, tracker)
    stats = discovery.get_source_stats()
    
    # Check structure and values
    assert isinstance(stats, dict)
    assert "jawz" in stats
    assert "dexter" in stats
    
    # Check jawz stats: 3 total, 1 processed, 2 unprocessed
    jawz_stats = stats["jawz"]
    assert jawz_stats["total"] == 3
    assert jawz_stats["processed"] == 1
    assert jawz_stats["unprocessed"] == 2
    
    # Check dexter stats: 2 total, 1 processed, 1 unprocessed  
    dexter_stats = stats["dexter"]
    assert dexter_stats["total"] == 2
    assert dexter_stats["processed"] == 1
    assert dexter_stats["unprocessed"] == 1


def _make_session_jsonl(user_message_count: int) -> str:
    """Create a minimal valid JSONL session with N user messages."""
    import json
    lines = []
    # Session metadata
    lines.append(json.dumps({"type": "session", "version": 3, "id": "test", "timestamp": "2024-01-01T00:00:00Z", "cwd": "/tmp"}))
    # Model change
    lines.append(json.dumps({"type": "model_change", "id": "m1", "parentId": None, "timestamp": "2024-01-01T00:00:00Z", "provider": "anthropic", "modelId": "claude-3"}))
    for i in range(user_message_count):
        # User message
        lines.append(json.dumps({"type": "message", "id": f"u{i}", "parentId": None, "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "user", "content": [{"type": "text", "text": f"User message {i}"}]}}))
        # Assistant response
        lines.append(json.dumps({"type": "message", "id": f"a{i}", "parentId": f"u{i}", "timestamp": "2024-01-01T00:00:00Z", "message": {"role": "assistant", "content": [{"type": "text", "text": f"Response {i} " + "x" * 500}]}}))
    return "\n".join(lines)


def test_discover_skips_sessions_below_min_user_messages(tmp_path):
    """Discovery should skip sessions with fewer user messages than min_session_messages."""
    from stelline.discovery import SessionDiscovery
    from stelline.config import StellineConfig, SourceConfig
    from stelline.tracker import SessionTracker

    session_dir = tmp_path / "sessions"
    source_dir = session_dir / "test_source"
    source_dir.mkdir(parents=True)

    # Agent run: 1 user message
    (source_dir / "2024-12-23T12-00-00_agent-run.jsonl").write_text(_make_session_jsonl(1))
    # Short interaction: 3 user messages
    (source_dir / "2024-12-23T13-00-00_short.jsonl").write_text(_make_session_jsonl(3))
    # Real conversation: 10 user messages
    (source_dir / "2024-12-23T14-00-00_real.jsonl").write_text(_make_session_jsonl(10))
    # Another real: 5 user messages
    (source_dir / "2024-12-23T15-00-00_medium.jsonl").write_text(_make_session_jsonl(5))

    config = StellineConfig(
        session_dir=session_dir,
        sources=[SourceConfig("test_source", "test_source", True)],
        min_session_messages=5,
    )

    tracker = SessionTracker(str(tmp_path / "test.db"))
    discovery = SessionDiscovery(config, tracker)
    unprocessed = discovery.discover_unprocessed()

    # Should only find sessions with 5+ user messages
    assert len(unprocessed) == 2
    ids = {s.session_id for s in unprocessed}
    assert "real" in ids
    assert "medium" in ids
    assert "agent-run" not in ids
    assert "short" not in ids


def test_discover_defaults_to_no_message_filter_when_min_is_zero(tmp_path):
    """When min_session_messages is 0, all sessions pass the filter."""
    from stelline.discovery import SessionDiscovery
    from stelline.config import StellineConfig, SourceConfig
    from stelline.tracker import SessionTracker

    session_dir = tmp_path / "sessions"
    source_dir = session_dir / "test_source"
    source_dir.mkdir(parents=True)

    (source_dir / "2024-12-23T12-00-00_single.jsonl").write_text(_make_session_jsonl(1))
    (source_dir / "2024-12-23T13-00-00_multi.jsonl").write_text(_make_session_jsonl(10))

    config = StellineConfig(
        session_dir=session_dir,
        sources=[SourceConfig("test_source", "test_source", True)],
        min_session_messages=0,
    )

    tracker = SessionTracker(str(tmp_path / "test.db"))
    discovery = SessionDiscovery(config, tracker)
    unprocessed = discovery.discover_unprocessed()

    assert len(unprocessed) == 2


def test_scan_shows_filtered_counts(tmp_path):
    """Scan stats should reflect the message filter."""
    from stelline.discovery import SessionDiscovery
    from stelline.config import StellineConfig, SourceConfig
    from stelline.tracker import SessionTracker

    session_dir = tmp_path / "sessions"
    source_dir = session_dir / "test_source"
    source_dir.mkdir(parents=True)

    (source_dir / "2024-12-23T12-00-00_agent.jsonl").write_text(_make_session_jsonl(1))
    (source_dir / "2024-12-23T13-00-00_real.jsonl").write_text(_make_session_jsonl(10))

    config = StellineConfig(
        session_dir=session_dir,
        sources=[SourceConfig("test_source", "test_source", True)],
        min_session_messages=5,
    )

    tracker = SessionTracker(str(tmp_path / "test.db"))
    discovery = SessionDiscovery(config, tracker)
    stats = discovery.get_source_stats()

    # Total should count ALL files, but unprocessed should reflect the filter
    assert stats["test_source"]["total"] == 2
    assert stats["test_source"]["unprocessed"] == 1  # only the real conversation