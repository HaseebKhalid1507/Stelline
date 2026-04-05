# Stelline

> *"If you have authentic memories, you have real human responses."*
> — Dr. Ana Stelline, Blade Runner 2049

Session intelligence for AI agents. Stelline reads conversation logs and crafts authentic memories — not data extraction, but memory making.

## What It Does

Stelline processes AI agent session transcripts and creates structured memories stored in [Memkoshi](https://github.com/HaseebKhalid1507/memkoshi). Each memory is grounded in the source transcript with exact quotes, importance scoring, and rich context.

```
Session transcript → Stelline → Memkoshi memories → searchable via VelociRAG
```

### The Write Path

Stelline is the **write path** for Memkoshi — it produces the memories that Memkoshi stores and [VelociRAG](https://github.com/HaseebKhalid1507/VelociRAG) searches. Together:

- **VelociRAG** — your agent can find things (read path)
- **Memkoshi** — your agent remembers (storage)
- **Stelline** — your agent learns from every conversation (write path)

## Features

- **Importance scoring** — 0.0-1.0 per memory, not flat. A config change (0.3) is not a strategic pivot (0.9).
- **Quality gate** — filters noise before staging. Operational trivia doesn't become memory.
- **Chunked processing** — large sessions split at message boundaries, processed in parts, deduped.
- **JSON truncation recovery** — salvages complete memories from partial LLM responses.
- **Context-aware** — searches existing memories before extracting, avoids duplicates.
- **VelociRAG-optimized** — memories structured for vector + keyword + graph search.
- **Scheduled harvesting** — systemd timer support for automated 2x daily processing.
- **Session tracking** — knows what it's processed, never double-processes.

## Install

```bash
pip install stelline
```

## Quick Start

### CLI

```bash
# Process a single session file
stelline harvest --file /path/to/session.jsonl

# Scan for unprocessed sessions
stelline scan

# Process all unprocessed sessions (up to batch size)
stelline harvest

# Check system status
stelline status

# View harvest history
stelline history
```

### Python API

```python
from stelline.config import StellineConfig
from stelline.discovery import SessionDiscovery
from stelline.pipeline import StellinePipeline
from stelline.tracker import SessionTracker
from stelline.context import ContextLoader

config = StellineConfig()
config.memkoshi_storage = "./my-memories"
config.session_dir = "~/.pi/agent/sessions"

tracker = SessionTracker(config.db_path)
context = ContextLoader(config)
pipeline = StellinePipeline(config, tracker, context)

# Process a session
from stelline.discovery import SessionFile
session = SessionFile.from_path(Path("session.jsonl"), "my-agent")
result = pipeline.process_session(session)

print(f"Extracted {result['memories_extracted']} memories")
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STELLINE_SOURCE` | `""` | Session source directory pattern |
| `STELLINE_CTX_PROJECTS` | `""` | Path to active projects context file |
| `STELLINE_CTX_SESSIONS` | `""` | Path to recent sessions context file |
| `STELLINE_CTX_PEOPLE` | `""` | Path to people context file |

### LLM Backend

Stelline uses Anthropic's API (SSE streaming) with automatic `pi -p` fallback:

```bash
# Auth via pi's OAuth token (~/.pi/agent/auth.json)
# Or set ANTHROPIC_API_KEY for direct API key auth
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Systemd Timer (Automated Harvesting)

```ini
# ~/.config/systemd/user/stelline-harvest.timer
[Timer]
OnCalendar=*-*-* 06,18:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

## Memory Quality

Stelline doesn't extract data — it crafts memories. The difference:

| Data Extraction | Memory Making |
|----------------|---------------|
| "Fixed the 429 bug" | "Months of mitmproxy deep-dives ended when a Reddit post revealed it was a string match on the system prompt. Haseeb typed 'YESSSSSSS' in all caps." |
| `importance: 0.5` (flat) | `importance: 0.95` (critical breakthrough) |
| Log entry | Something you'd remember |

Every memory includes:
- Exact quotes from the conversation
- Specific file paths, URLs, people named with roles
- Emotional context — breakthroughs, frustrations, reactions
- Importance score calibrated to actual significance

## How It Works

```
1. Discovery — finds unprocessed .jsonl session files
2. Parsing — converts JSONL events to clean transcript
3. Context — searches existing Memkoshi memories to avoid duplicates
4. Extraction — LLM crafts 6-10 memories per session with importance scoring
5. Quality Gate — filters by content length, importance threshold, category validity
6. Staging — memories staged in Memkoshi for review or auto-approval
7. Indexing — search index updated for next session's context lookup
```

## Architecture

```
stelline/
├── cli.py          # Click CLI (harvest, scan, status, history)
├── config.py       # Configuration with env var support
├── context.py      # Existing memory search + project context loading
├── discovery.py    # Session file discovery and filtering
├── llm.py          # LLM client (SSE + pi fallback, JSON salvage)
├── pipeline.py     # Core pipeline (quality gate, chunking, extraction)
├── tracker.py      # Processed session tracking (SQLite)
└── parsers/
    └── pi.py       # Pi session JSONL parser
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
# 86 passed
```

## License

MIT
