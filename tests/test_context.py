"""Tests for context module."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


def test_loads_project_context_files():
    """Context loader should load project context from configured files."""
    from stelline.context import ContextLoader
    from stelline.config import StellineConfig
    
    # Create temporary context files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock context files
        projects_file = tmpdir / "projects_active.md"
        projects_file.write_text("# Active Projects\n\n- Stelline v1.0\n- VelociRAG improvements")
        
        sessions_file = tmpdir / "sessions_recent.md"  
        sessions_file.write_text("# Recent Sessions\n\n- Built Stelline parser\n- Fixed auth bug")
        
        people_file = tmpdir / "people.md"
        people_file.write_text("# People\n\n- Haseeb: Main user")
        
        # Create config with custom paths
        config = StellineConfig()
        config.context.projects_active = str(projects_file)
        config.context.sessions_recent = str(sessions_file)
        config.context.people = str(people_file)
        
        loader = ContextLoader(config)
        context = loader.load_project_context()
        
        assert "Active Projects" in context
        assert "Stelline v1.0" in context
        assert "Recent Sessions" in context
        assert "Built Stelline parser" in context
        assert "People" in context
        assert "Haseeb: Main user" in context


def test_handles_missing_context_files_gracefully():
    """Context loader should handle missing context files gracefully."""
    from stelline.context import ContextLoader
    from stelline.config import StellineConfig
    
    config = StellineConfig()
    config.context.projects_active = "/nonexistent/path.md"
    config.context.sessions_recent = "/nonexistent/path2.md"
    config.context.people = "/nonexistent/path3.md"
    
    loader = ContextLoader(config)
    context = loader.load_project_context()
    
    # Should return empty string or handle gracefully
    assert isinstance(context, str)  # Should not crash


def test_searches_existing_memories_via_memkoshi():
    """Context loader should search existing memories using Memkoshi search."""
    from stelline.context import ContextLoader
    from stelline.config import StellineConfig
    
    # Mock Memkoshi instance
    mock_memkoshi = Mock()
    mock_memkoshi.search.search.return_value = [
        {"text": "Previously built parser for JSONL files"},
        {"content": "Fixed authentication token handling"},
        {"text": "Implemented session discovery logic"}
    ]
    
    config = StellineConfig()
    config.max_recall_memories = 20
    
    loader = ContextLoader(config)
    
    transcript = "Help me debug this parser issue with JSONL files"
    memories = loader.search_existing_memories(transcript, mock_memkoshi)
    
    # Should have called search with representative chunks
    mock_memkoshi.search.search.assert_called_once()
    call_args = mock_memkoshi.search.search.call_args[0]
    assert len(call_args) == 2  # query, top_k
    assert "debug this parser" in call_args[0] or "JSONL files" in call_args[0]
    assert call_args[1] == 20
    
    # Should return processed memory texts
    assert len(memories) == 3
    assert "Previously built parser" in memories[0]
    assert "Fixed authentication" in memories[1]
    assert "session discovery" in memories[2]


def test_builds_search_query_from_transcript_chunks():
    """Context loader should build search query from first 500 + last 300 chars."""
    from stelline.context import ContextLoader
    from stelline.config import StellineConfig
    
    mock_memkoshi = Mock()
    mock_memkoshi.search.search.return_value = []
    
    config = StellineConfig()
    loader = ContextLoader(config)
    
    # Create long transcript > 800 chars
    transcript = "A" * 500 + "B" * 400 + "C" * 300
    
    loader.search_existing_memories(transcript, mock_memkoshi)
    
    call_args = mock_memkoshi.search.search.call_args[0]
    query = call_args[0]
    
    # Should include first 500 chars
    assert "A" * 500 in query
    # Should include last 300 chars  
    assert "C" * 300 in query
    # Should NOT include middle section
    assert "B" * 400 not in query


def test_handles_memkoshi_search_errors():
    """Context loader should handle Memkoshi search errors gracefully."""
    from stelline.context import ContextLoader
    from stelline.config import StellineConfig
    
    mock_memkoshi = Mock()
    mock_memkoshi.search.search.side_effect = Exception("Search service down")
    
    config = StellineConfig()
    loader = ContextLoader(config)
    
    transcript = "Some session transcript"
    memories = loader.search_existing_memories(transcript, mock_memkoshi)
    
    # Should return empty list on error, not crash
    assert isinstance(memories, list)
    assert len(memories) == 0


def test_limits_recalled_memories_to_max_config():
    """Context loader should limit recalled memories to configured maximum."""
    from stelline.context import ContextLoader
    from stelline.config import StellineConfig
    
    mock_memkoshi = Mock()
    # Return more memories than the limit
    mock_memkoshi.search.search.return_value = [
        {"text": f"Memory {i}"} for i in range(25)
    ]
    
    config = StellineConfig()
    config.max_recall_memories = 5  # Set low limit
    
    loader = ContextLoader(config)
    transcript = "Test transcript"
    memories = loader.search_existing_memories(transcript, mock_memkoshi)
    
    # Should limit to configured max
    assert len(memories) == 5


def test_extracts_memory_text_from_different_formats():
    """Context loader should handle different Memkoshi result formats."""
    from stelline.context import ContextLoader
    from stelline.config import StellineConfig
    
    config = StellineConfig()
    loader = ContextLoader(config)
    
    # Test different memory result formats
    assert loader._extract_memory_text({"text": "memory text"}) == "memory text"
    assert loader._extract_memory_text({"content": "memory content"}) == "memory content"
    assert loader._extract_memory_text("string memory") == "string memory"
    assert loader._extract_memory_text({"other": "field"}) is None
    assert loader._extract_memory_text(None) is None


def test_deduplicates_recalled_memories():
    """Context loader should deduplicate identical recalled memories."""
    from stelline.context import ContextLoader
    from stelline.config import StellineConfig
    
    mock_memkoshi = Mock()
    mock_memkoshi.search.search.return_value = [
        {"text": "Same memory"},
        {"content": "Different memory"},
        {"text": "Same memory"},  # Duplicate
        {"content": "Another memory"},
        {"text": "Same memory"}   # Another duplicate
    ]
    
    config = StellineConfig()
    loader = ContextLoader(config)
    
    transcript = "Test transcript"
    memories = loader.search_existing_memories(transcript, mock_memkoshi)
    
    # Should have deduplicated
    assert len(memories) == 3  # Only unique memories
    assert "Same memory" in memories
    assert "Different memory" in memories
    assert "Another memory" in memories