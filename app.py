"""
Gift Whisperer — Agentic AI course assignment
A minimal multi-turn agent that finds gifts on Amazon India.

Architecture:
  User query
    → [Turn 1] Gemini decides which tool to call → executes tool → appends result
    → [Turn 2] Gemini sees history + result → calls next tool → appends result
    → ...
    → [Turn N] Gemini has enough info → returns final answer (no tool call)

Every turn's full context (query + all prior LLM responses + all prior tool
results) is sent to Gemini. This satisfies the assignment's context-accumulation
requirement.
"""

import hashlib
import os
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response
from google import genai
from google.genai import types
from dotenv import load_dotenv

from tools import (
    TOOL_REGISTRY,
    TOOL_DECLARATIONS,
    check_gemini_connectivity,
    check_rapidapi_connectivity,
    gemini_generate_with_retry,
    get_rapidapi_usage,
    validate_tool_args,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
MAX_TURNS = 12
LOG_FILE = "llm_log.txt"
CACHE_DIR = Path(".cache")

SYSTEM_INSTRUCTION = """You are **Gift Whisperer**, an AI agent that finds the perfect gift on Amazon India using the tools provided.

WORKFLOW (follow this order, but you may loop or skip):
  1. Read the user's message to extract: recipient name, age, interests, relationship, occasion, budget (INR).
  2. Call `search_amazon_india` with specific keywords tied to one of the recipient's interests. You SHOULD search multiple times with different keyword strategies (e.g. "premium yoga mat", "fantasy novel boxset") to give yourself real options.
  3. Pick the 2-3 most promising candidates across all searches. For each, call `get_product_details` to inspect reviews, features, and availability.
  4. For each serious candidate, call `calculate_value_score` to get an objective ranking. Parse the price (strip "₹" and commas) and the rating (e.g. "4.3 out of 5 stars" -> 4.3) carefully.
  5. Pick the single winner — highest value score AND best match for the recipient's interests.
  6. Call `compose_gift_card_message` exactly ONCE for the winner.
  7. Respond with the final recommendation (no more tool calls). Include: product title, price, URL, why you picked it, and the gift card message.

RULES:
  - Before every tool call, write a brief 1-2 sentence thought explaining WHY you're calling it. This is shown to the user as your reasoning.
  - Never exceed the user's budget.
  - If a search returns no products, try different keywords.
  - Keep the final response conversational and warm — not a JSON dump.
"""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit(
        "Missing GEMINI_API_KEY — copy .env.example to .env and fill it in."
    )
if not os.environ.get("RAPIDAPI_KEY"):
    raise SystemExit("Missing RAPIDAPI_KEY — copy .env.example to .env and fill it in.")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ---------------------------------------------------------------------------
# Connectivity pre-checks — fail fast with actionable errors
# ---------------------------------------------------------------------------
try:
    check_gemini_connectivity()
except RuntimeError as e:
    raise SystemExit(f"Startup check failed: {e}")

try:
    check_rapidapi_connectivity()
except RuntimeError as e:
    raise SystemExit(f"Startup check failed: {e}")


# ---------------------------------------------------------------------------
# Logging — writes a human-readable trace for assignment submission
# ---------------------------------------------------------------------------
def _part_to_str(part):
    if getattr(part, "text", None):
        return f"TEXT: {part.text.strip()}"
    if getattr(part, "function_call", None):
        fc = part.function_call
        try:
            args = dict(fc.args) if fc.args else {}
        except Exception:
            args = str(fc.args)
        return f"FUNCTION_CALL: {fc.name}({json.dumps(args, ensure_ascii=False)})"
    if getattr(part, "function_response", None):
        fr = part.function_response
        try:
            resp = dict(fr.response) if fr.response else {}
        except Exception:
            resp = str(fr.response)
        # Truncate huge tool responses in the log
        resp_str = json.dumps(resp, ensure_ascii=False)
        if len(resp_str) > 1200:
            resp_str = resp_str[:1200] + f"… [truncated, {len(resp_str)} chars total]"
        return f"FUNCTION_RESPONSE: {fr.name} -> {resp_str}"
    return "UNKNOWN_PART"


def _part_to_serializable(part):
    """Convert a types.Part into a JSON-safe dict for the frontend log panel."""
    if getattr(part, "text", None):
        return {"type": "text", "text": part.text.strip()}
    if getattr(part, "function_call", None):
        fc = part.function_call
        try:
            args = dict(fc.args) if fc.args else {}
        except Exception:
            args = str(fc.args)
        return {"type": "function_call", "name": fc.name, "args": args}
    if getattr(part, "function_response", None):
        fr = part.function_response
        try:
            resp = dict(fr.response) if fr.response else {}
        except Exception:
            resp = str(fr.response)
        resp_str = json.dumps(resp, ensure_ascii=False)
        truncated = len(resp_str) > 2000
        if truncated:
            resp_str = resp_str[:2000] + f"… [{len(resp_str)} chars total]"
        return {
            "type": "function_response",
            "name": fr.name,
            "response": resp_str,
            "truncated": truncated,
        }
    return {"type": "unknown"}


def _content_to_serializable(content):
    """Convert a types.Content into a JSON-safe dict."""
    return {
        "role": content.role,
        "parts": [_part_to_serializable(p) for p in (content.parts or [])],
    }


def log_session_start(query):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("GIFT WHISPERER — LLM LOG\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"Model:   {MODEL_NAME}\n")
        f.write(f"User Query:\n  {query}\n")
        f.write("=" * 80 + "\n")


def log_turn(turn_num, contents_sent, response):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'#' * 80}\n")
        f.write(f"# TURN {turn_num}  ({datetime.now().isoformat()})\n")
        f.write(f"{'#' * 80}\n")

        f.write("\n---- REQUEST: full accumulated context sent to Gemini ----\n")
        for i, c in enumerate(contents_sent):
            f.write(f"\n  [{i}] role={c.role}\n")
            for p in c.parts:
                f.write(f"      {_part_to_str(p)}\n")

        f.write("\n---- RESPONSE from Gemini ----\n")
        for cand in response.candidates:
            for p in cand.content.parts:
                f.write(f"  {_part_to_str(p)}\n")


# ---------------------------------------------------------------------------
# The manual agent loop — the heart of the assignment
# ---------------------------------------------------------------------------
def run_agent_streaming(user_query: str):
    """Generator that yields per-turn events for SSE streaming.

    Yields dicts with keys:
      kind='start'  — initial metadata (system_instruction, model)
      kind='turn'   — per-turn data (event dict + log_entry dict)
      kind='done'   — signals stream end
    """
    log_session_start(user_query)

    # The accumulating conversation — grows every turn.
    contents: list = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
        temperature=0.3,
    )

    yield {
        "kind": "start",
        "system_instruction": SYSTEM_INSTRUCTION,
        "model": MODEL_NAME,
    }

    for turn_num in range(1, MAX_TURNS + 1):
        # ---- Call Gemini with ALL accumulated context ----
        try:
            response = gemini_generate_with_retry(
                client,
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
        except Exception as e:
            yield {
                "kind": "turn",
                "event": {"type": "error", "error": f"Gemini call failed: {e}"},
                "log_entry": None,
            }
            yield {"kind": "done"}
            return

        log_turn(turn_num, contents, response)

        # ---- Parse response parts ----
        candidate = response.candidates[0]
        parts = candidate.content.parts or []

        # Snapshot log BEFORE contents is mutated further this turn
        log_entry = {
            "turn": turn_num,
            "request": [_content_to_serializable(c) for c in contents],
            "response": [_part_to_serializable(p) for p in parts],
        }

        thought_text = ""
        function_calls = []
        for part in parts:
            if getattr(part, "text", None):
                thought_text += part.text
            if getattr(part, "function_call", None):
                function_calls.append(part.function_call)

        # ---- No function calls → final answer ----
        if not function_calls:
            yield {
                "kind": "turn",
                "event": {
                    "type": "final",
                    "turn": turn_num,
                    "final_answer": thought_text.strip() or "(empty response)",
                },
                "log_entry": log_entry,
            }
            yield {"kind": "done"}
            return

        # ---- Append the model's turn (text + function_calls) to history ----
        contents.append(candidate.content)

        # ---- Execute each function call; collect results ----
        turn_event = {
            "type": "tool_turn",
            "turn": turn_num,
            "thinking": thought_text.strip(),
            "tool_calls": [],
        }
        function_response_parts = []

        for fc in function_calls:
            tool_name = fc.name
            try:
                tool_args = dict(fc.args) if fc.args else {}
            except Exception:
                tool_args = {}

            if tool_name not in TOOL_REGISTRY:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                tool_func = TOOL_REGISTRY[tool_name]
                validation_err = validate_tool_args(tool_func, tool_args)
                if validation_err:
                    result = {"error": validation_err}
                else:
                    try:
                        result = tool_func(**tool_args)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}

            turn_event["tool_calls"].append(
                {
                    "name": tool_name,
                    "args": tool_args,
                    "result": result,
                }
            )

            function_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
            )

        # ---- Append all tool results to history as a single user turn ----
        contents.append(types.Content(role="user", parts=function_response_parts))

        yield {
            "kind": "turn",
            "event": turn_event,
            "log_entry": log_entry,
        }

    # Safety fallback: max turns exceeded — include recent context for debugging
    recent_summary = []
    for c in contents[-4:]:
        for p in c.parts or []:
            if getattr(p, "text", None):
                snippet = p.text.strip()[:200]
                recent_summary.append(f"[{c.role}] {snippet}")
            if getattr(p, "function_call", None):
                recent_summary.append(f"[{c.role}] called {p.function_call.name}")
            if getattr(p, "function_response", None):
                recent_summary.append(
                    f"[{c.role}] result from {p.function_response.name}"
                )

    api_usage = get_rapidapi_usage()

    yield {
        "kind": "turn",
        "event": {
            "type": "error",
            "error": f"Agent exceeded {MAX_TURNS} turns without finalizing.",
            "diagnostics": {
                "total_turns": MAX_TURNS,
                "recent_activity": recent_summary,
                "rapidapi_usage": api_usage,
            },
        },
        "log_entry": None,
    }
    yield {"kind": "done"}


def run_agent(user_query: str) -> dict:
    """Non-streaming wrapper — collects all events from the streaming generator."""
    events: list = []
    log_entries: list = []
    for chunk in run_agent_streaming(user_query):
        if chunk["kind"] == "turn":
            events.append(chunk["event"])
            if chunk.get("log_entry"):
                log_entries.append(chunk["log_entry"])
    return _agent_result(events, log_entries)


def _agent_result(events, log_entries):
    """Bundle events and log data into the standard return dict."""
    return {
        "events": events,
        "log": {
            "system_instruction": SYSTEM_INSTRUCTION,
            "model": MODEL_NAME,
            "turns": log_entries,
        },
    }


# ---------------------------------------------------------------------------
# File-based response cache
# ---------------------------------------------------------------------------
CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(query: str) -> str:
    """Deterministic filename from normalized query text."""
    normalized = query.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _cache_get(query: str) -> dict | None:
    """Return cached response dict, or None if miss."""
    path = CACHE_DIR / f"{_cache_key(query)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _cache_put(query: str, response: dict) -> None:
    """Write response dict to cache."""
    path = CACHE_DIR / f"{_cache_key(query)}.json"
    try:
        path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # non-critical — next run will just re-fetch


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    body = request.json or {}
    query = body.get("query", "").strip()
    if not query:
        return jsonify({"turns": [{"type": "error", "error": "Empty query"}]})

    force = body.get("force", False)

    if not force:
        cached = _cache_get(query)
        if cached:
            cached["cached"] = True
            return jsonify(cached)

    result = run_agent(query)
    response = {"turns": result["events"], "log": result["log"]}
    _cache_put(query, response)
    response["cached"] = False
    return jsonify(response)


@app.route("/run-stream", methods=["POST"])
def run_stream():
    body = request.json or {}
    query = body.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"})

    force = body.get("force", False)

    if not force:
        cached = _cache_get(query)
        if cached:
            cached["cached"] = True

            def gen_cached():
                yield f"data: {json.dumps({'kind': 'cached', 'data': cached}, ensure_ascii=False)}\n\n"

            return Response(
                gen_cached(),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    def generate():
        all_events = []
        all_log_entries = []
        for chunk in run_agent_streaming(query):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            if chunk["kind"] == "turn":
                all_events.append(chunk["event"])
                if chunk.get("log_entry"):
                    all_log_entries.append(chunk["log_entry"])
            elif chunk["kind"] == "done":
                response_data = {
                    "turns": all_events,
                    "log": {
                        "system_instruction": SYSTEM_INSTRUCTION,
                        "model": MODEL_NAME,
                        "turns": all_log_entries,
                    },
                }
                _cache_put(query, response_data)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/cache", methods=["DELETE"])
def clear_cache():
    """Remove all cached responses."""
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return jsonify({"deleted": count})


if __name__ == "__main__":
    # debug=True is fine for a course demo; disable before any public deploy.
    app.run(host="127.0.0.1", port=5000, debug=True)
