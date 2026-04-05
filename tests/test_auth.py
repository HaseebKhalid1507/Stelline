"""Tests for auth module."""
import json
import pytest
import time
from pathlib import Path


def test_reads_oauth_token_from_auth_json(tmp_path):
    """Should read OAuth token from auth.json file."""
    from stelline.auth import get_auth_token
    
    # Create test auth file with OAuth token
    auth_file = tmp_path / "auth.json"
    auth_data = {
        "anthropic": {
            "type": "oauth",
            "access": "test_oauth_token_123",
            "expires": int((time.time() + 3600) * 1000)  # Expires in 1 hour
        }
    }
    auth_file.write_text(json.dumps(auth_data))
    
    # Test reading token
    token = get_auth_token(str(auth_file))
    
    assert token == "test_oauth_token_123"


def test_detects_expired_token(tmp_path):
    """Should raise RuntimeError for expired OAuth token."""
    from stelline.auth import get_auth_token
    
    # Create test auth file with expired OAuth token
    auth_file = tmp_path / "auth.json"
    auth_data = {
        "anthropic": {
            "type": "oauth",
            "access": "expired_token_123",
            "expires": int((time.time() - 3600) * 1000)  # Expired 1 hour ago
        }
    }
    auth_file.write_text(json.dumps(auth_data))
    
    # Test expired token detection
    with pytest.raises(RuntimeError, match="OAuth token expired"):
        get_auth_token(str(auth_file))


def test_handles_missing_auth_file(tmp_path):
    """Should raise RuntimeError for missing auth file."""
    from stelline.auth import get_auth_token
    
    # Try to read from non-existent file
    missing_file = tmp_path / "nonexistent.json"
    
    with pytest.raises(RuntimeError, match="Pi auth file not found"):
        get_auth_token(str(missing_file))


def test_handles_invalid_json(tmp_path):
    """Should raise RuntimeError for invalid JSON in auth file."""
    from stelline.auth import get_auth_token
    
    # Create auth file with invalid JSON
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{ invalid json content [")
    
    with pytest.raises(RuntimeError, match="Invalid auth file format"):
        get_auth_token(str(auth_file))