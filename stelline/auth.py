"""Authentication handling for Stelline."""
import json
import time
from pathlib import Path


def get_auth_token(auth_path: str = "~/.pi/agent/auth.json") -> str:
    """Get Anthropic auth token from pi's auth.json."""
    auth_file = Path(auth_path).expanduser()
    
    try:
        with open(auth_file, 'r') as f:
            auth = json.load(f)
    except FileNotFoundError:
        raise RuntimeError("Pi auth file not found - run pi to authenticate")
    except json.JSONDecodeError:
        raise RuntimeError("Invalid auth file format")

    creds = auth.get("anthropic", {})
    
    if creds.get("type") == "oauth":
        # Check token expiry
        expires = creds.get("expires", 0)
        if expires < time.time() * 1000:
            raise RuntimeError("OAuth token expired - run pi to refresh")
        return creds.get("access", "")
    elif creds.get("type") == "api_key":
        return creds.get("key", "")
    
    raise RuntimeError("No anthropic credentials found in auth file")