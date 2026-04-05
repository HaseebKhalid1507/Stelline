"""Tests for parser module."""
import pytest
import json
import tempfile
from pathlib import Path


def test_parses_session_metadata():
    """Parser should extract session metadata from JSONL file."""
    from stelline.parsers.pi import PiSessionParser
    
    # Use the test fixture created in Phase 1
    fixture_path = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    
    parser = PiSessionParser()
    session_data = parser.parse_file(str(fixture_path))
    
    # Check metadata structure
    assert "meta" in session_data
    assert "messages" in session_data
    
    meta = session_data["meta"]
    assert meta["session_id"] == "test_session_123"
    assert meta["model"] == "claude-3-haiku-20240307"
    assert meta["cwd"] == "/home/haseeb"
    assert meta["timestamp"] is not None


def test_extracts_user_messages_from_jsonl():
    """Parser should extract user message text from JSONL."""
    from stelline.parsers.pi import PiSessionParser
    
    fixture_path = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    parser = PiSessionParser()
    session_data = parser.parse_file(str(fixture_path))
    
    # Find user messages
    user_messages = [msg for msg in session_data["messages"] if msg["role"] == "user"]
    
    assert len(user_messages) == 2
    assert user_messages[0]["text"] == "Help me debug this Python error"
    assert user_messages[1]["text"] == "Perfect, thanks for the fix!"


def test_extracts_assistant_text_content():
    """Parser should extract assistant text content parts."""
    from stelline.parsers.pi import PiSessionParser
    
    fixture_path = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    parser = PiSessionParser()
    session_data = parser.parse_file(str(fixture_path))
    
    # Find assistant messages
    assistant_messages = [msg for msg in session_data["messages"] if msg["role"] == "assistant"]
    
    assert len(assistant_messages) == 3
    
    # Check first assistant message has text content
    first_msg = assistant_messages[0]
    text_parts = [part for part in first_msg["parts"] if part["type"] == "text"]
    assert len(text_parts) == 1
    assert "I'd be happy to help debug the error" in text_parts[0]["text"]


def test_extracts_tool_calls():
    """Parser should extract tool calls with name and arguments."""
    from stelline.parsers.pi import PiSessionParser
    
    fixture_path = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    parser = PiSessionParser()
    session_data = parser.parse_file(str(fixture_path))
    
    # Find assistant messages with tool calls
    assistant_messages = [msg for msg in session_data["messages"] if msg["role"] == "assistant"]
    
    # First message should have read_file tool call
    tool_parts = [part for part in assistant_messages[0]["parts"] if part["type"] == "tool_call"]
    assert len(tool_parts) == 1
    
    tool_call = tool_parts[0]
    assert tool_call["name"] == "read_file"
    assert tool_call["arguments"]["path"] == "debug.py"
    assert tool_call["id"] == "call_1"
    
    # Second assistant message should have edit_file tool call
    tool_parts = [part for part in assistant_messages[1]["parts"] if part["type"] == "tool_call"]
    assert len(tool_parts) == 1
    
    tool_call = tool_parts[0]
    assert tool_call["name"] == "edit_file"
    assert "debug.py" in tool_call["arguments"]["path"]


def test_skips_thinking_blocks_by_default():
    """Parser should skip thinking blocks by default."""
    from stelline.parsers.pi import PiSessionParser
    
    # Create a test session with thinking blocks
    thinking_session = [
        {"type":"session","id":"test_123","timestamp":"2024-12-23T12:00:00Z"},
        {"type":"message","message":{
            "role":"assistant",
            "content":[
                {"type":"thinking","text":"Let me think about this..."},
                {"type":"text","text":"Here's my response"}
            ]
        },"timestamp":"2024-12-23T12:00:01Z"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for event in thinking_session:
            f.write(json.dumps(event) + '\n')
        temp_path = f.name
    
    try:
        parser = PiSessionParser()  # include_thinking defaults to False
        session_data = parser.parse_file(temp_path)
        
        assistant_msg = session_data["messages"][0]
        
        # Should only have text part, not thinking part
        assert len(assistant_msg["parts"]) == 1
        assert assistant_msg["parts"][0]["type"] == "text"
        assert assistant_msg["parts"][0]["text"] == "Here's my response"
    finally:
        Path(temp_path).unlink()


def test_includes_thinking_blocks_when_enabled():
    """Parser should include thinking blocks when enabled."""
    from stelline.parsers.pi import PiSessionParser
    
    # Create a test session with thinking blocks
    thinking_session = [
        {"type":"session","id":"test_123","timestamp":"2024-12-23T12:00:00Z"},
        {"type":"message","message":{
            "role":"assistant",
            "content":[
                {"type":"thinking","text":"Let me think about this..."},
                {"type":"text","text":"Here's my response"}
            ]
        },"timestamp":"2024-12-23T12:00:01Z"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for event in thinking_session:
            f.write(json.dumps(event) + '\n')
        temp_path = f.name
    
    try:
        parser = PiSessionParser(include_thinking=True)
        session_data = parser.parse_file(temp_path)
        
        assistant_msg = session_data["messages"][0]
        
        # Should have both thinking and text parts
        assert len(assistant_msg["parts"]) == 2
        assert assistant_msg["parts"][0]["type"] == "thinking"
        assert assistant_msg["parts"][0]["text"] == "Let me think about this..."
        assert assistant_msg["parts"][1]["type"] == "text"
        assert assistant_msg["parts"][1]["text"] == "Here's my response"
    finally:
        Path(temp_path).unlink()


def test_handles_malformed_jsonl_gracefully():
    """Parser should handle malformed JSONL gracefully."""
    from stelline.parsers.pi import PiSessionParser
    
    # Create JSONL with some malformed lines
    malformed_session = [
        '{"type":"session","id":"test_123","timestamp":"2024-12-23T12:00:00Z"}',
        'THIS IS NOT JSON',
        '',  # Empty line
        '{"broken_json":}',  # Malformed JSON
        '{"type":"message","message":{"role":"user","content":"Hello"}}'
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for line in malformed_session:
            f.write(line + '\n')
        temp_path = f.name
    
    try:
        parser = PiSessionParser()
        session_data = parser.parse_file(temp_path)
        
        # Should parse valid lines and skip malformed ones
        assert "meta" in session_data
        assert "messages" in session_data
        
        # Should have one valid user message
        user_messages = [msg for msg in session_data["messages"] if msg["role"] == "user"]
        assert len(user_messages) == 1
        assert user_messages[0]["text"] == "Hello"
    finally:
        Path(temp_path).unlink()


def test_generates_transcript_output():
    """Parser should generate clean transcript format."""
    from stelline.parsers.pi import PiSessionParser
    
    fixture_path = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    parser = PiSessionParser()
    transcript = parser.to_transcript(str(fixture_path))
    
    # Check basic transcript structure
    assert "USER:" in transcript
    assert "ASSISTANT:" in transcript
    assert "Help me debug this Python error" in transcript
    assert "division by zero" in transcript
    assert "Perfect, thanks for the fix!" in transcript
    
    # Should not contain raw JSON or timestamps
    assert "{" not in transcript
    assert "timestamp" not in transcript
    assert '"type":"message"' not in transcript


def test_transcript_output_excludes_tool_calls_by_default():
    """Transcript should exclude tool calls by default for clean reading."""
    from stelline.parsers.pi import PiSessionParser
    
    fixture_path = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    parser = PiSessionParser()
    transcript = parser.to_transcript(str(fixture_path))
    
    # Tool calls should not appear in clean transcript
    assert "read_file" not in transcript
    assert "edit_file" not in transcript
    assert "call_1" not in transcript
    
    # But human-readable tool effects should be mentioned in assistant text
    assert "division" in transcript


def test_transcript_output_includes_tool_calls_when_requested():
    """Transcript should include tool calls when specifically requested."""
    from stelline.parsers.pi import PiSessionParser
    
    fixture_path = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    parser = PiSessionParser(include_tools=True)
    transcript = parser.to_transcript(str(fixture_path), include_tools=True)
    
    # Tool calls should appear when explicitly requested
    assert "[TOOL: read_file]" in transcript or "read_file(" in transcript
    assert "[TOOL: edit_file]" in transcript or "edit_file(" in transcript