"""Tests for pipeline module."""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime


def test_acquires_and_releases_file_lock():
    """Pipeline should acquire file lock to prevent concurrent runs."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    from stelline.tracker import SessionTracker
    from stelline.context import ContextLoader
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config.db_path = f"{tmpdir}/stelline.db" 
        
        pipeline = StellinePipeline(config, tracker, context_loader)
        pipeline.LOCK_FILE = Path(tmpdir) / "stelline.lock"
        
        # Should acquire lock successfully
        assert pipeline.acquire_lock() == True
        assert pipeline.LOCK_FILE.exists()
        
        # Should release lock  
        pipeline.release_lock()


def test_prevents_concurrent_pipeline_runs():
    """Pipeline should prevent concurrent runs with file locking."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    from stelline.tracker import SessionTracker
    from stelline.context import ContextLoader
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config.db_path = f"{tmpdir}/stelline.db"
        
        pipeline1 = StellinePipeline(config, tracker, context_loader)
        pipeline2 = StellinePipeline(config, tracker, context_loader) 
        
        lock_file = Path(tmpdir) / "stelline.lock"
        pipeline1.LOCK_FILE = lock_file
        pipeline2.LOCK_FILE = lock_file
        
        # First pipeline should acquire lock
        assert pipeline1.acquire_lock() == True
        
        # Second pipeline should fail to acquire lock
        assert pipeline2.acquire_lock() == False
        
        pipeline1.release_lock()


def test_parses_session_transcript():
    """Pipeline should parse session file to clean transcript."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    with patch("stelline.pipeline.PiSessionParser") as mock_parser_cls:
        mock_parser = Mock()
        mock_parser.to_transcript.return_value = "USER: Hello\nASSISTANT: Hi there"
        mock_parser_cls.return_value = mock_parser
        
        transcript = pipeline._parse_session(Path("/test/session.jsonl"))
        
        assert transcript == "USER: Hello\nASSISTANT: Hi there"
        mock_parser.to_transcript.assert_called_once()


def test_handles_parse_failures():
    """Pipeline should handle parser failures gracefully."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    session_path = Path("/test/session.jsonl")
    
    with patch("stelline.pipeline.PiSessionParser") as mock_parser_cls:
        mock_parser = Mock()
        mock_parser.to_transcript.side_effect = FileNotFoundError("not found")
        mock_parser_cls.return_value = mock_parser
        
        with pytest.raises(FileNotFoundError):
            pipeline._parse_session(session_path)


def test_skips_tiny_sessions():
    """Pipeline should skip sessions with very short transcripts."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    from stelline.discovery import SessionFile
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    session_file = Mock()
    session_file.session_id = "tiny_123"
    session_file.source = "test"
    session_file.timestamp = datetime.now()
    session_file.path = Path("/test.jsonl")
    
    with patch.object(pipeline, "_parse_session", return_value="Hi"):  # Very short
        result = pipeline.process_session(session_file)
        
        assert result["status"] == "skipped"
        assert result["reason"] == "transcript too short"


def test_searches_existing_memories():
    """Pipeline should search for existing memories via context loader."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    from stelline.discovery import SessionFile
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    
    context_loader.search_existing_memories.return_value = [
        "Previously built parser functionality",
        "Fixed auth token handling"
    ]
    context_loader.load_project_context.return_value = "Active project: Stelline"
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    session_file = Mock()
    session_file.session_id = "test_123"
    session_file.source = "test"
    session_file.timestamp = datetime.now()
    session_file.path = Path("/test.jsonl")
    
    transcript = "USER: Help with parser\nASSISTANT: Fixed parser issue" + " and more text" * 50  # Make it long enough
    
    with patch.object(pipeline, "_parse_session", return_value=transcript), \
         patch.object(pipeline, "_get_memkoshi_instance") as mock_memkoshi:
        
        pipeline.process_session(session_file, dry_run=True)
        
        context_loader.search_existing_memories.assert_called_once_with(
            transcript, mock_memkoshi.return_value
        )


def test_builds_memory_creation_prompt():
    """Pipeline should build comprehensive prompt with three inputs."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    
    config = StellineConfig()
    config.max_transcript_chars = 1000
    tracker = Mock()
    context_loader = Mock()
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    transcript = "USER: Debug error\nASSISTANT: Fixed the bug"
    existing_memories = ["Previously fixed similar bug"]
    project_context = "Active projects: Stelline, VelociRAG"
    
    prompt = pipeline._build_prompt(transcript, existing_memories, project_context)
    
    assert "Stelline" in prompt
    assert "Previously fixed similar bug" in prompt
    assert "USER: Debug error" in prompt
    assert "SESSION TRANSCRIPT" in prompt


def test_truncates_long_transcripts():
    """Pipeline should truncate very long transcripts."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    
    config = StellineConfig()
    config.max_transcript_chars = 1000
    tracker = Mock()
    context_loader = Mock()
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    # Create transcript longer than limit
    long_transcript = "A" * 600 + "B" * 600  # 1200 chars
    
    prompt = pipeline._build_prompt(long_transcript, [], "")
    
    assert "[... TRUNCATED ...]" in prompt
    # Should contain truncated parts: keep = 1000//2 - 100 = 400 chars each
    assert "A" * 400 in prompt  # First 400 A's
    assert "B" * 400 in prompt  # Last 400 B's  
    # But not the full 600-char sequences
    assert "A" * 600 not in prompt
    assert "B" * 600 not in prompt


def test_dry_run_mode():
    """Pipeline should support dry run mode without calling LLM."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    from stelline.discovery import SessionFile
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    context_loader.search_existing_memories.return_value = ["mem1", "mem2"]
    context_loader.load_project_context.return_value = "context"
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    session_file = Mock()
    session_file.session_id = "test_123"
    session_file.source = "test"
    session_file.timestamp = datetime.now()
    session_file.path = Path("/test.jsonl")
    
    transcript = "USER: Test\nASSISTANT: Response" * 20  # Long enough
    
    with patch.object(pipeline, "_parse_session", return_value=transcript), \
         patch.object(pipeline, "_get_memkoshi_instance"):
        
        result = pipeline.process_session(session_file, dry_run=True)
        
        assert result["status"] == "dry_run"
        assert result["transcript_chars"] == len(transcript)
        assert result["existing_memories_count"] == 2
        assert "prompt_chars" in result


def test_calls_llm_and_stages_memories():
    """Pipeline should call LLM and stage memories in Memkoshi."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    from stelline.discovery import SessionFile
    from memkoshi.core.memory import Memory, MemoryCategory, MemoryConfidence
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    context_loader.search_existing_memories.return_value = []
    context_loader.load_project_context.return_value = ""
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    session_file = Mock()
    session_file.session_id = "test_123"  
    session_file.source = "test"
    session_file.timestamp = datetime.now()
    session_file.path = Path("/test.jsonl")
    
    # Mock memories to be extracted
    mock_memory = Memory(
        id="mem_12345678",
        category=MemoryCategory.EVENTS,
        topic="test-memory-topic",
        title="Test memory extraction",
        abstract="Test abstract for memory",
        content="Test content that is long enough to pass the quality gate threshold for minimum content length validation",
        confidence=MemoryConfidence.HIGH
    )
    
    transcript = "USER: Test\nASSISTANT: Response" * 20
    
    with patch.object(pipeline, "_parse_session", return_value=transcript), \
         patch.object(pipeline, "_get_memkoshi_instance") as mock_memkoshi_factory, \
         patch.object(pipeline, "_call_llm_and_parse", return_value=([mock_memory], [])) as mock_llm:
        
        mock_memkoshi = Mock()
        mock_memkoshi_factory.return_value = mock_memkoshi
        
        result = pipeline.process_session(session_file)
        
        # Should have called LLM
        mock_llm.assert_called_once()
        
        # Should have staged memory
        mock_memkoshi.storage.stage_memory.assert_called_once_with(mock_memory)
        
        # Should track successful processing
        tracker.record_session.assert_called_once()
        
        assert result["status"] == "success"
        assert result["memories_extracted"] == 1


def test_tracks_processing_errors():
    """Pipeline should track processing errors in database."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    from stelline.discovery import SessionFile
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    session_file = Mock()
    session_file.session_id = "error_123"
    session_file.source = "test"
    session_file.timestamp = datetime.now()
    session_file.path = Path("/test.jsonl")
    
    with patch.object(pipeline, "_parse_session", side_effect=RuntimeError("Parse failed")):
        result = pipeline.process_session(session_file)
        
        assert result["status"] == "failed"
        assert "Parse failed" in result["error"]
        
        # Should have tracked the failure
        tracker.record_session.assert_called_once()
        call_kwargs = tracker.record_session.call_args[1]
        assert call_kwargs["status"] == "failed"
        assert "Parse failed" in call_kwargs["error"]


def test_gets_memkoshi_instance_with_source_config():
    """Pipeline should get Memkoshi instance with correct storage path."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig, SourceConfig
    
    config = StellineConfig()
    config.memkoshi_storage = "~/.memkoshi-default"
    config.sources = [
        SourceConfig("test", "pattern", memkoshi_storage="~/.memkoshi-custom")
    ]
    
    tracker = Mock()
    context_loader = Mock()
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    with patch("memkoshi.Memkoshi") as mock_memkoshi_class:
        mock_instance = Mock()
        mock_memkoshi_class.return_value = mock_instance
        
        # Should use custom storage for configured source
        result = pipeline._get_memkoshi_instance("test")
        
        mock_memkoshi_class.assert_called_once()
        call_args = mock_memkoshi_class.call_args[1]
        storage_path = str(call_args["storage_path"])
        assert "memkoshi-custom" in storage_path
        
        mock_instance.init.assert_called_once()


def test_builds_prompt_with_instructions():
    """Pipeline should build prompt with clear memory creation instructions."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    
    config = StellineConfig()
    tracker = Mock() 
    context_loader = Mock()
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    prompt = pipeline._build_prompt("transcript", ["memory"], "context")
    
    # Should contain key instruction elements
    assert "transcript" in prompt.lower()
    assert "memory" in prompt
    assert "context" in prompt
    assert "memories" in prompt.lower()
    assert "granular" in prompt.lower() or "extract" in prompt.lower()


def test_records_session_metrics():
    """Pipeline should record detailed session processing metrics."""
    from stelline.pipeline import StellinePipeline
    from stelline.config import StellineConfig
    from stelline.discovery import SessionFile
    
    config = StellineConfig()
    tracker = Mock()
    context_loader = Mock()
    context_loader.search_existing_memories.return_value = []
    context_loader.load_project_context.return_value = ""
    
    pipeline = StellinePipeline(config, tracker, context_loader)
    
    session_file = Mock()
    session_file.session_id = "metrics_123"
    session_file.source = "test"
    session_file.timestamp = datetime(2024, 12, 23, 12, 0, 0)
    session_file.path = Path("/test.jsonl")
    
    transcript = "A" * 1000
    
    with patch.object(pipeline, "_parse_session", return_value=transcript), \
         patch.object(pipeline, "_get_memkoshi_instance") as mock_memkoshi_factory, \
         patch.object(pipeline, "_call_llm_and_parse", return_value=([Mock(), Mock()], [])):
        
        mock_memkoshi = Mock()
        mock_memkoshi_factory.return_value = mock_memkoshi
        
        pipeline.process_session(session_file)
        
        # Check recorded metrics
        call_kwargs = tracker.record_session.call_args[1]
        assert call_kwargs["session_id"] == "metrics_123"
        assert call_kwargs["source"] == "test"
        assert call_kwargs["memory_count"] == 2
        assert call_kwargs["transcript_chars"] == 1000
        assert call_kwargs["session_date"] == "2024-12-23T12:00:00"
        assert "duration_seconds" in call_kwargs