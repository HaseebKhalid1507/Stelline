"""Tests for LLM module."""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock


# --- Auth tests ---

def test_reads_auth_token_from_auth_json(tmp_path):
    """LLM client should read auth token from pi's auth.json."""
    from stelline.llm import LLMClient

    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "anthropic": {
            "type": "oauth",
            "access": "test_token_123",
            "expires": 9999999999999
        }
    }))

    client = LLMClient()
    with patch("stelline.llm.Path") as mock_path:
        mock_instance = Mock()
        mock_instance.expanduser.return_value = auth_file
        mock_path.return_value = mock_instance
        token = client._get_auth_token()

    assert token == "test_token_123"


def test_handles_missing_auth_file():
    """LLM client should handle missing auth file with clear error."""
    from stelline.llm import LLMClient

    client = LLMClient()
    with patch("stelline.llm.Path") as mock_path:
        mock_path.return_value.expanduser.return_value.exists.return_value = False
        with pytest.raises(RuntimeError, match="Pi auth file not found"):
            client._get_auth_token()


def test_handles_expired_oauth_token(tmp_path):
    """LLM client should detect expired OAuth token."""
    from stelline.llm import LLMClient
    import time

    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "anthropic": {
            "type": "oauth",
            "access": "expired",
            "expires": int(time.time() * 1000) - 1000
        }
    }))

    client = LLMClient()
    with patch("stelline.llm.Path") as mock_path:
        mock_instance = Mock()
        mock_instance.expanduser.return_value = auth_file
        mock_path.return_value = mock_instance
        with pytest.raises(RuntimeError, match="OAuth token expired"):
            client._get_auth_token()


# --- SSE backend tests ---

def _mock_sse_response(memories_json: str):
    """Create a mock SSE streaming response."""
    lines = [
        'event: message_start',
        f'data: {{"type":"message_start"}}',
        '',
        'event: content_block_start',
        f'data: {{"type":"content_block_start","index":0}}',
        '',
        'event: content_block_delta',
        f'data: {{"type":"content_block_delta","delta":{{"type":"text_delta","text":{json.dumps(memories_json)}}}}}',
        '',
        'event: message_stop',
        f'data: {{"type":"message_stop"}}',
    ]
    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    mock_resp.iter_lines = Mock(return_value=iter(lines))
    return mock_resp


def test_sse_streams_and_parses_memories():
    """SSE backend should stream response and parse into Memory objects."""
    from stelline.llm import LLMClient

    memories_json = json.dumps({
        "memories": [{
            "category": "events",
            "topic": "stelline",
            "title": "Built parser functionality",
            "abstract": "Implemented JSONL parser",
            "content": "Created PiSessionParser class",
            "confidence": "high",
            "source_quotes": ["debug this"],
            "related_topics": ["parsing"]
        }]
    })

    mock_resp = _mock_sse_response(memories_json)
    with patch("stelline.llm.requests.post", return_value=mock_resp):
        client = LLMClient(backend="sse")
        client._auth_token = "test_token"
        memories = client.extract_memories("test prompt")

    assert len(memories) == 1
    assert memories[0].title == "Built parser functionality"
    assert memories[0].category.value == "events"
    assert memories[0].confidence.value == "high"


def test_sse_sends_correct_headers():
    """SSE backend should send claude-code beta flag and streaming headers."""
    from stelline.llm import LLMClient

    mock_resp = _mock_sse_response('{"memories": []}')
    with patch("stelline.llm.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient(backend="sse")
        client._auth_token = "test_token"
        client.extract_memories("test")

        call_kwargs = mock_post.call_args[1]
        headers = call_kwargs["headers"]
        assert "claude-code-20250219" in headers["anthropic-beta"]
        assert "oauth-2025-04-20" in headers["anthropic-beta"]
        assert headers["accept"] == "text/event-stream"
        assert call_kwargs["json"]["stream"] is True
        assert call_kwargs["stream"] is True


def test_sse_uses_configured_model():
    """SSE backend should use configured model."""
    from stelline.llm import LLMClient

    mock_resp = _mock_sse_response('{"memories": []}')
    with patch("stelline.llm.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient(model="claude-3-opus-20240229", backend="sse")
        client._auth_token = "test_token"
        client.extract_memories("test")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "claude-3-opus-20240229"


def test_sse_includes_system_prompt():
    """SSE backend should include system prompt."""
    from stelline.llm import LLMClient

    mock_resp = _mock_sse_response('{"memories": []}')
    with patch("stelline.llm.requests.post", return_value=mock_resp) as mock_post:
        client = LLMClient(backend="sse")
        client._auth_token = "test_token"
        client.extract_memories("test")

        payload = mock_post.call_args[1]["json"]
        assert "system" in payload
        assert "Stelline" in payload["system"]
        assert "memories" in payload["system"].lower()


def test_sse_falls_back_to_pi_on_429():
    """SSE backend should fall back to pi on 429."""
    from stelline.llm import LLMClient
    import requests as req

    mock_resp = Mock()
    mock_resp.raise_for_status.side_effect = req.RequestException("429 Rate Limited")

    mock_pi_result = Mock()
    mock_pi_result.returncode = 0
    mock_pi_result.stdout = json.dumps({"memories": [{"category": "events", "topic": "t", "title": "Fallback worked", "abstract": "a", "content": "c", "confidence": "high"}]})

    with patch("stelline.llm.requests.post", return_value=mock_resp), \
         patch("stelline.llm.subprocess.run", return_value=mock_pi_result):
        client = LLMClient(backend="sse")
        client._auth_token = "test_token"
        memories = client.extract_memories("test")
        assert len(memories) == 1
        assert memories[0].title == "Fallback worked"


def test_sse_raises_non_429_errors():
    """SSE backend should raise on non-429 errors."""
    from stelline.llm import LLMClient
    import requests as req

    mock_resp = Mock()
    mock_resp.raise_for_status.side_effect = req.RequestException("500 Server Error")

    with patch("stelline.llm.requests.post", return_value=mock_resp):
        client = LLMClient(backend="sse")
        client._auth_token = "test_token"
        with pytest.raises((RuntimeError, req.RequestException)):
            client.extract_memories("test")


# --- Pi backend tests ---

def test_pi_backend_calls_subprocess():
    """Pi backend should call pi -p with correct args."""
    from stelline.llm import LLMClient

    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({
        "memories": [{
            "category": "events",
            "topic": "test",
            "title": "Test memory from pi",
            "abstract": "test",
            "content": "test content",
            "confidence": "high",
            "source_quotes": [],
            "related_topics": []
        }]
    })

    with patch("stelline.llm.subprocess.run", return_value=mock_result) as mock_run:
        client = LLMClient(backend="pi", model="claude-sonnet-4-6")
        memories = client.extract_memories("test prompt")

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "pi"
        assert "-p" in cmd
        assert "--no-session" in cmd
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd

    assert len(memories) == 1
    assert memories[0].title == "Test memory from pi"


def test_pi_backend_handles_failure():
    """Pi backend should raise on subprocess failure."""
    from stelline.llm import LLMClient

    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stderr = "pi crashed"

    with patch("stelline.llm.subprocess.run", return_value=mock_result):
        client = LLMClient(backend="pi")
        with pytest.raises(RuntimeError, match="pi -p failed"):
            client.extract_memories("test")


def test_pi_backend_handles_timeout():
    """Pi backend should raise on timeout."""
    from stelline.llm import LLMClient
    import subprocess

    with patch("stelline.llm.subprocess.run", side_effect=subprocess.TimeoutExpired("pi", 600)):
        client = LLMClient(backend="pi")
        with pytest.raises(RuntimeError, match="timed out"):
            client.extract_memories("test")


# --- Shared parsing tests ---

def test_extracts_json_from_markdown_code_block():
    """Should extract JSON from markdown code blocks."""
    from stelline.llm import LLMClient
    client = LLMClient()

    text = '```json\n{"memories": []}\n```'
    assert client._extract_json(text) == '{"memories": []}'


def test_extracts_raw_json():
    """Should handle raw JSON without wrapping."""
    from stelline.llm import LLMClient
    client = LLMClient()

    text = '{"memories": [{"title": "test"}]}'
    assert client._extract_json(text) == text


def test_generates_unique_memory_ids():
    """Should generate deterministic IDs based on title."""
    from stelline.llm import LLMClient
    client = LLMClient()

    mem1 = client._item_to_memory({"category": "events", "topic": "a", "title": "Memory One", "abstract": "x", "content": "x", "confidence": "high"})
    mem2 = client._item_to_memory({"category": "events", "topic": "b", "title": "Memory Two", "abstract": "x", "content": "x", "confidence": "high"})

    assert mem1.id != mem2.id
    assert mem1.id.startswith("mem_")
    assert len(mem1.id) == 12


def test_skips_malformed_memory_items():
    """Should skip items with missing required fields."""
    from stelline.llm import LLMClient
    client = LLMClient()

    assert client._item_to_memory({}) is None
    assert client._item_to_memory({"title": "ab"}) is None  # too short
    assert client._item_to_memory({"title": "Valid title", "category": "events", "topic": "t", "content": "c", "abstract": "a", "confidence": "high"}) is not None
