# 🎁 Gift Whisperer

A tiny multi-turn agent that finds the perfect gift on Amazon India.
Built for the Agentic AI course assignment.

## What it does

You describe a recipient, occasion, and budget. The agent:

1. **Searches Amazon India** with keyword strategies tied to the recipient's interests (real API calls — no mocks).
2. **Pulls details** on the top candidates.
3. **Calculates an objective value-for-money score** for each finalist.
4. **Composes a personalized gift-card message** for the winner.
5. **Presents the final recommendation** with product link, price, and card text.

Everything the agent does — reasoning, tool calls, tool results — is displayed live in the browser. The full LLM conversation is also written to `llm_log.txt` for submission.

## Assignment checklist

| Requirement                                 | Where it's satisfied                                          |
|---------------------------------------------|---------------------------------------------------------------|
| Query → LLM → Tool → Query → LLM → ...      | `app.py::run_agent()` — the manual loop                       |
| Each query stores ALL past interaction      | `contents` list grows every turn with model + tool messages   |
| Display agent reasoning chain               | `templates/index.html` — thinking, tool calls, results, final |
| ≥ 3 custom tool functions                   | `tools.py` — 4 tools (search, details, scoring, message)      |
| Keep it simple                              | ~400 lines of Python + 1 HTML file + 0 frameworks             |
| Submission: video + LLM logs                | `llm_log.txt` written automatically each run                  |

## The 4 tools

1. **`search_amazon_india`** — real RapidAPI call to Amazon India search
2. **`get_product_details`** — real RapidAPI call for deep product info by ASIN
3. **`calculate_value_score`** — pure calculation: affordability × rating × review volume
4. **`compose_gift_card_message`** — nested Gemini call to write a personalized card

## Setup

```bash
# 1. Create a virtual env and install deps
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Get your API keys
#    - Gemini:   https://aistudio.google.com/app/apikey  (free)
#    - RapidAPI: https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-amazon-data
#                (sign up, subscribe to the Basic / Free plan)

# 3. Configure
cp .env.example .env
#  then open .env and paste both keys

# 4. Run
python app.py

# 5. Open http://127.0.0.1:5000 in your browser
```

## Usage tips for the demo video

- The three pre-filled example queries are crafted to show different tool patterns. Pick one, hit **Find Gift**, and let the animation roll — each turn slides in with a 400ms delay so viewers can follow.
- A full run makes roughly **4–7 RapidAPI calls** and **5–8 Gemini calls**. With the RapidAPI free tier (~500/month) you can do ~70 runs before hitting the ceiling.
- After the run, open `llm_log.txt` — you can copy-paste this directly as your "LLM logs" submission.

## File map

```
gift_whisperer/
├── app.py                  # Flask server + manual agent loop + logging
├── tools.py                # 4 tool functions + Gemini function-declaration schemas
├── templates/
│   └── index.html          # Minimal dark-themed UI
├── .env.example            # Template for API keys
├── requirements.txt
└── README.md
```

## Troubleshooting

- **`Missing GEMINI_API_KEY`** — you didn't create `.env` from `.env.example`.
- **403 from RapidAPI** — key is wrong or you haven't subscribed to the API on the RapidAPI dashboard.
- **Empty search results** — try different keywords or widen the price range. The free tier occasionally returns partial results.
- **Agent loops forever** — `MAX_TURNS = 12` in `app.py` is a hard safety cap. Bump or lower to taste.
- **Parse errors on `price`** — the agent sometimes fumbles `₹1,299.00 → 1299`. If so, check your system prompt or add an explicit parsing example.

