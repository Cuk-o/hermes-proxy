# Anthropic Token Saver Proxy

Local HTTP proxy between Claude Code CLI and the Anthropic API.
Translates Russian → English before sending, English → Russian after receiving.
Saves 40–60% of tokens (Russian uses ~1.75× more tokens than English).

## How it works

```
Claude Code CLI → proxy:8000 → api.anthropic.com
                     ↑
              RU→EN on request
              EN→RU on response
              (via Gemini free tier)
```

- Translates only the **last** user message (not full history — avoids re-translating cached context)
- Haiku model calls bypass translation entirely (sub-agent tool calls must stay in English)
- Restores Russian assistant messages back to English before forwarding (keeps Anthropic context in English)
- Injects `"MUST respond ONLY in English"` rule into the system prompt (appended after cache breakpoints to preserve cache hit rate)
- Rotates across 5 free-tier Gemini models on 429 rate limits
- LRU cache (500 entries) avoids redundant translation calls

## Prerequisites

- Python 3.10+
- [Google Gemini API key](https://aistudio.google.com/apikey) (free tier is enough)
- Claude Code CLI (`npm i -g @anthropic-ai/claude-code` or `npx`)

## Installation

```bash
# 1. Clone or download
git clone https://github.com/Cuk-o/hermes-proxy.git
cd hermes-proxy

# 2. Install dependencies
pip install aiohttp google-genai python-dotenv

# Or with venv (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install aiohttp google-genai python-dotenv
```

## Configuration

### Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click **"Create API key in new project"**
3. Copy the key

### Set up .env

Create `.env` in the project root:

```env
GEMINI_API_KEY=paste_your_key_here
```

Or copy from template:

```bash
cp .env.example .env
# Then edit .env with your actual key
```

Optional env vars:

| Variable | Default | Description |
|---|---|---|
| `PROXY_DEBUG` | `1` | Print translation logs (`0` = silent) |

### Anthropic API endpoint (config.json)

Create `config.json` in the project root to override defaults:

```json
{
  "anthropic_api_url": "https://api.anthropic.com",
  "proxy_host": "127.0.0.1",
  "proxy_port": 8000
}
```

All fields are optional — omit any to keep the default.

## Running

```bash
# Terminal 1: start proxy
python3 proxy.py

# Terminal 2: start Claude Code pointed at proxy
ANTHROPIC_BASE_URL=http://localhost:8000 claude
```

Proxy listens on `http://127.0.0.1:8000`.

### Permanent setup via Claude Code global config

Instead of passing `ANTHROPIC_BASE_URL` every time, add it to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000"
  }
}
```

Claude Code will pick it up automatically on every launch.

## Token savings stats

Printed on `Ctrl+C` (SIGINT) or SIGTERM:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 SESSION STATS  (12 requests)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INPUT:   RU ~4,200 → EN ~2,400  saved 1,800 (43%)
  OUTPUT:  EN ~3,100  (would be RU ~5,400)  saved 2,300 (43%)
  TOTAL:   9,600 → 5,500  saved 4,100 (43%)
────────────────────────────────────────────────────────────
  CACHE:   read=1,200  created=800
  GEMINI:  8 calls  |  CACHE HITS: 4
────────────────────────────────────────────────────────────
  💰 API billing:      ~43% fewer tokens
  ⏱  5h subscription:  ~51% more capacity (cache discount 1,080 tok)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Cumulative stats persist to `~/.tr_stats.json`.

## Gemini model rotation

Models tried in order (on 429, rotates to next):

| Model | Daily limit |
|---|---|
| `gemma-3-27b-it` | 14,400 RPD |
| `gemma-4-31b-it` | 1,500 RPD |
| `gemma-4-26b-it` | TBD |
| `gemini-3.1-flash-lite-preview` | 500 RPD |
| `gemini-2.5-flash-lite` | 20 RPD |

All free tier. Get key at [aistudio.google.com](https://aistudio.google.com/apikey).

## Troubleshooting

**`GEMINI_API_KEY not set`** — add key to `.env` or export in shell.

**Translation lag** — first request initializes Gemini client; normal. Subsequent requests use LRU cache.

**`All Gemini models exhausted`** — all free-tier daily limits hit. Wait until midnight UTC or add a paid key.

**`Connection reset`** — proxy crashed; restart `proxy.py`. Claude Code will reconnect automatically if `ANTHROPIC_BASE_URL` is still set.
