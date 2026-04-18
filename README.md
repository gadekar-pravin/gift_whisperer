# 🎁 Gift Whisperer

A multi-turn AI agent that finds the perfect gift on Amazon India.
Built for the Agentic AI course assignment.

## What it does

You describe a recipient, occasion, and budget. The agent:

1. **Searches Amazon India** with keyword strategies tied to the recipient's interests (real API calls — no mocks).
2. **Pulls details** on the top candidates.
3. **Calculates an objective value-for-money score** for each finalist.
4. **Composes a personalized gift-card message** for the winner.
5. **Presents the final recommendation** with product link, price, and card text.

Agent reasoning, tool calls, and results stream live to the browser via SSE. The full LLM conversation is also written to `llm_log.txt` for submission.

## Assignment checklist

| Requirement                                 | Where it's satisfied                                          |
|---------------------------------------------|---------------------------------------------------------------|
| Query → LLM → Tool → Query → LLM → ...      | `app.py::run_agent_streaming()` — the manual loop             |
| Each query stores ALL past interaction      | `contents` list grows every turn with model + tool messages   |
| Display agent reasoning chain               | `templates/index.html` — thinking, tool calls, results, final |
| ≥ 3 custom tool functions                   | `tools.py` — 4 tools (search, details, scoring, message)      |
| Keep it simple                              | ~820 lines Python + 1 HTML file, React 18 loaded via CDN (no build step) |
| Submission: video + LLM logs                | `llm_log.txt` written automatically each run                  |

## The 4 tools

1. **`search_amazon_india`** — real RapidAPI call to Amazon India search
2. **`get_product_details`** — real RapidAPI call for deep product info by ASIN
3. **`calculate_value_score`** — pure calculation: affordability × rating × review volume
4. **`compose_gift_card_message`** — nested Gemini call to write a personalized card

## Features

- **SSE streaming** — agent turns stream live to the browser via the `/run-stream` endpoint; no waiting for the full loop to finish
- **File-based response cache** — repeated queries are served instantly from `.cache/`; clear via the UI button or `DELETE /cache`
- **3 themes** — ink (dark), vellum (warm parchment, default), paper (light)
- **Sidebar** — example prompts to get started + session history
- **Configurable model** — set `GEMINI_MODEL` in `.env` to override the default (`gemini-flash-latest`)

## Setup

Requires **Python ≥ 3.13** and [**uv**](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Get your API keys
#    - Gemini:   https://aistudio.google.com/app/apikey  (free)
#    - RapidAPI: https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-amazon-data
#                (sign up, subscribe to the Basic / Free plan)

# 3. Configure
cp .env.example .env
#  then open .env and paste both keys

# 4. Run
uv run python app.py

# 5. Open http://127.0.0.1:5000 in your browser
```

## Usage tips for the demo video

- The sidebar has pre-filled example prompts crafted to show different tool patterns. Pick one, hit **Find Gift**, and watch the turns stream in live.
- A full run makes roughly **4–7 RapidAPI calls** and **5–8 Gemini calls**. With the RapidAPI free tier (~500/month) you can do ~70 runs before hitting the ceiling.
- Repeated queries are served from cache. Use the **Clear Cache** button (or `curl -X DELETE http://127.0.0.1:5000/cache`) to force a fresh run.
- After the run, open `llm_log.txt` — you can copy-paste this directly as your "LLM logs" submission.

## File map

```
gift_whisperer/
├── app.py                  # Flask server + manual agent loop + SSE streaming + cache
├── tools.py                # 4 tool functions + Gemini function-declaration schemas
├── templates/
│   └── index.html          # React 18 SPA (CDN, no build step) — themes, sidebar, history
├── pyproject.toml           # Primary dependency spec (uv / PEP 621)
├── uv.lock                 # Locked dependency versions
├── .python-version          # Pins Python 3.13
├── requirements.txt         # Legacy pip requirements (kept for reference)
├── .env.example             # Template for API keys
├── .gitignore
├── .cache/                  # Response cache (git-ignored)
├── CLAUDE.md                # Claude Code project instructions
└── README.md
```

## Troubleshooting

- **`Missing GEMINI_API_KEY`** — you didn't create `.env` from `.env.example`.
- **403 from RapidAPI** — key is wrong or you haven't subscribed to the API on the RapidAPI dashboard.
- **Empty search results** — try different keywords or widen the price range. The free tier occasionally returns partial results.
- **Agent loops forever** — `MAX_TURNS = 12` in `app.py` is a hard safety cap. Bump or lower to taste.
- **Parse errors on `price`** — the agent sometimes fumbles `₹1,299.00 → 1299`. If so, check your system prompt or add an explicit parsing example.
- **Stale results** — the file-based cache may serve outdated responses. Clear it via the UI button or `DELETE /cache`.
