"""Pi session parser - extracted from stelline-parse."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class PiSessionParser:
    """Parse pi JSONL session files into readable transcripts."""

    def __init__(self, include_thinking: bool = False, include_tools: bool = True):
        self.include_thinking = include_thinking
        self.include_tools = include_tools

    def parse_file(self, path: str) -> Dict[str, Any]:
        """Parse a JSONL session file. Returns structured session data."""
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        events = []
        with open(path, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Skip malformed lines

        if not events:
            raise ValueError(f"No valid events in {path}")

        # Extract metadata from first event
        meta = self._extract_metadata(events, path)

        # Process messages
        messages = []
        stats = {"user_messages": 0, "assistant_messages": 0,
                 "tool_calls": 0, "tool_results": 0, "thinking_blocks": 0,
                 "total_events": len(events)}

        for evt in events:
            if evt.get("type") != "message":
                continue

            msg = evt.get("message", {})
            role = msg.get("role", "unknown")
            content = msg.get("content")
            timestamp = evt.get("timestamp", "")

            if role == "user":
                stats["user_messages"] += 1
                text = self._extract_text(content)
                if text:
                    messages.append({"role": "user", "text": text, "ts": timestamp})

            elif role == "assistant":
                stats["assistant_messages"] += 1
                parts = self._extract_assistant_content(content, stats)
                if parts:
                    messages.append({"role": "assistant", "parts": parts, "ts": timestamp})

            elif role == "toolResult":
                stats["tool_results"] += 1
                text = self._extract_text(content)
                if text and self.include_tools:
                    messages.append({"role": "tool_result", "text": text, "ts": timestamp})

        meta["stats"] = stats
        return {"meta": meta, "messages": messages}

    def _extract_metadata(self, events: List[dict], path: Path) -> dict:
        """Extract session metadata from events."""
        meta = {
            "file": str(path),
            "session_id": None,
            "timestamp": None,
            "cwd": None,
            "model": None,
        }

        for evt in events:
            t = evt.get("type")
            if t == "session":
                meta["session_id"] = evt.get("id")
                meta["timestamp"] = evt.get("timestamp")
                meta["cwd"] = evt.get("cwd")
            elif t == "model_change" and not meta["model"]:
                meta["model"] = evt.get("modelId")

        return meta

    def _extract_text(self, content) -> str:
        """Extract text from content (string or block array)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return ""

    def _extract_assistant_content(self, content, stats: dict) -> List[dict]:
        """Extract structured content from assistant message."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        if not isinstance(content, list):
            return []

        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue

            btype = block.get("type")

            if btype == "text":
                text = block.get("text", "")
                if text.strip():
                    parts.append({"type": "text", "text": text})

            elif btype == "thinking":
                stats["thinking_blocks"] += 1
                if self.include_thinking:
                    text = block.get("text", block.get("thinking", ""))
                    if text:
                        parts.append({"type": "thinking", "text": text})

            elif btype == "toolCall":
                stats["tool_calls"] += 1
                if self.include_tools:
                    parts.append({
                        "type": "tool_call",
                        "name": block.get("name", "unknown"),
                        "arguments": block.get("arguments", {}),
                        "id": block.get("id", ""),
                    })

        return parts

    def to_transcript(self, path: str, include_tools: bool = False) -> str:
        """Generate clean transcript format from session file."""
        session_data = self.parse_file(path)
        lines = []
        
        for msg in session_data["messages"]:
            if msg["role"] == "user":
                lines.append(f"USER: {msg['text']}")
                
            elif msg["role"] == "assistant":
                text_parts = []
                for part in msg["parts"]:
                    if part["type"] == "text":
                        text_parts.append(part["text"])
                    elif part["type"] == "tool_call" and include_tools:
                        name = part["name"]
                        args = part["arguments"]
                        text_parts.append(f"[TOOL: {name}({args})]")
                        
                if text_parts:
                    lines.append(f"ASSISTANT: {' '.join(text_parts)}")
                    
            elif msg["role"] == "tool_result" and include_tools:
                lines.append(f"TOOL RESULT: {msg['text']}")
                
        return "\n\n".join(lines)