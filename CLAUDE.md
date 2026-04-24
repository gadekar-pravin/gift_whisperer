# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Gift Whisperer — a multi-turn AI agent (Flask + Google Gemini, default `gemini-flash-latest`) that finds personalized gift recommendations on Amazon India via RapidAPI.

## Setup

```bash
cp .env.example .env   # fill in keys for your chosen backend
uv sync                # install all dependencies (including dev)
uv run python app.py   # serves at http://127.0.0.1:5000
```

## Required Environment Variables

- `RAPIDAPI_KEY` — RapidAPI Real-Time Amazon Data key (free tier, ~500 requests/month)
- `GEMINI_BACKEND` — `"aistudio"` (default) or `"vertexai"`

**AI Studio backend** (default):
- `GEMINI_API_KEY` — Google AI Studio key (free tier)

**Vertex AI backend** (`GEMINI_BACKEND=vertexai`):
- `GOOGLE_CLOUD_PROJECT` — GCP project ID (required)
- `GOOGLE_CLOUD_LOCATION` — GCP region (optional, defaults to `us-central1`)
- Auth via Application Default Credentials (`gcloud auth application-default login`)

All loaded via `python-dotenv`. App exits on startup if required variables are missing.

## RapidAPI Quota

Free tier is ~500 requests/month. Avoid unnecessary `search_amazon_india` or `get_product_details` calls during development and testing. Prefer mocking or caching responses when possible.

## Architecture

- `app.py` — Flask server + manual agent loop (`run_agent`), max 12 turns. Logs full LLM conversation to `llm_log.txt`.
- `tools.py` — 4 tool functions + Gemini function declarations:
  1. `search_amazon_india` — real RapidAPI search
  2. `get_product_details` — real RapidAPI product details
  3. `calculate_value_score` — pure scoring (affordability 35%, rating 40%, review volume 25%)
  4. `compose_gift_card_message` — nested Gemini call for personalized card text
- `templates/index.html` — single-page dark-themed UI with agent reasoning display

The agent loop is intentionally manual (no LangChain) to satisfy course requirements.

## Formatting

Uses `ruff` for linting and formatting. Run `uv run ruff check .` and `uv run ruff format --check .` to verify.

## Dependency Management

Uses `uv` — never `pip`. Add packages with `uv add <pkg>` (or `uv add --dev <pkg>` for dev-only). Run scripts with `uv run`.

## Files to Never Commit

- `.env` (contains real API keys)
- `llm_log.txt` (generated logs)
- `.venv/` / `venv/` / `__pycache__/`
