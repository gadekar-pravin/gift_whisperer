"""
Four custom tools used by the Gift Whisperer agent.

  1. search_amazon_india     — REAL API call (RapidAPI / Real-Time Amazon Data)
  2. get_product_details     — REAL API call (RapidAPI / Real-Time Amazon Data)
  3. calculate_value_score   — pure calculation
  4. compose_gift_card_message — creative generation via a secondary Gemini call

Each tool has (a) the Python function, (b) a Gemini FunctionDeclaration schema.
"""

import inspect
import logging
import math
import os
import requests

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

RAPIDAPI_HOST = "real-time-amazon-data.p.rapidapi.com"
RAPIDAPI_BASE = f"https://{RAPIDAPI_HOST}"


def _get_runtime_config() -> dict[str, str]:
    """Read environment-backed configuration at call time."""
    return {
        "rapidapi_key": os.environ.get("RAPIDAPI_KEY", ""),
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
    }


# ---------------------------------------------------------------------------
# RapidAPI rate-limit tracking
# ---------------------------------------------------------------------------
_rapidapi_calls_made: int = 0
_rapidapi_limit: int | None = None  # from X-RateLimit-Limit header
_rapidapi_remaining: int | None = None  # from X-RateLimit-Remaining header


def get_rapidapi_usage() -> dict:
    """Return current RapidAPI usage stats for diagnostics."""
    return {
        "calls_made_this_session": _rapidapi_calls_made,
        "rate_limit": _rapidapi_limit,
        "rate_remaining": _rapidapi_remaining,
    }


def _track_rapidapi_headers(response: requests.Response) -> None:
    """Read rate-limit headers from a RapidAPI response and update tracking."""
    global _rapidapi_calls_made, _rapidapi_limit, _rapidapi_remaining
    _rapidapi_calls_made += 1

    limit = response.headers.get("X-RateLimit-Requests-Limit") or response.headers.get(
        "x-ratelimit-requests-limit"
    )
    remaining = response.headers.get(
        "X-RateLimit-Requests-Remaining"
    ) or response.headers.get("x-ratelimit-requests-remaining")

    if limit is not None:
        try:
            _rapidapi_limit = int(limit)
        except ValueError:
            pass
    if remaining is not None:
        try:
            _rapidapi_remaining = int(remaining)
        except ValueError:
            pass

    if _rapidapi_remaining is not None and _rapidapi_remaining <= 10:
        log.warning(
            "RapidAPI quota low: %d requests remaining (limit %s)",
            _rapidapi_remaining,
            _rapidapi_limit,
        )


def _check_rapidapi_quota() -> str | None:
    """Return an error string if we know quota is exhausted, else None."""
    if _rapidapi_remaining is not None and _rapidapi_remaining <= 0:
        return (
            f"RapidAPI quota exhausted (limit: {_rapidapi_limit}). "
            "Wait for the monthly reset or upgrade your plan."
        )
    return None


# ---------------------------------------------------------------------------
# Connectivity pre-checks (called once at startup)
# ---------------------------------------------------------------------------
def check_gemini_connectivity() -> None:
    """Verify Gemini API key works. Raises RuntimeError on failure."""
    config = _get_runtime_config()
    gemini_api_key = config["gemini_api_key"]
    gemini_model = config["gemini_model"]

    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    try:
        test_client = genai.Client(api_key=gemini_api_key)
        test_client.models.generate_content(
            model=gemini_model,
            contents="Say OK",
            config=types.GenerateContentConfig(
                max_output_tokens=5,
                temperature=0.0,
            ),
        )
    except Exception as e:
        raise RuntimeError(
            f"Gemini connectivity check failed (model={gemini_model}): {e}"
        ) from e


def check_rapidapi_connectivity() -> None:
    """Verify RapidAPI key works with a lightweight call. Raises RuntimeError on failure."""
    rapidapi_key = _get_runtime_config()["rapidapi_key"]

    if not rapidapi_key:
        raise RuntimeError("RAPIDAPI_KEY is not set.")
    url = f"{RAPIDAPI_BASE}/search"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    # Cheapest possible call — 1 result for a common query.
    params = {"query": "test", "country": "IN", "page": "1"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code == 403:
            raise RuntimeError(
                "RapidAPI returned 403 Forbidden — check that your RAPIDAPI_KEY is valid "
                "and you are subscribed to the Real-Time Amazon Data API."
            )
        if r.status_code == 429:
            raise RuntimeError(
                "RapidAPI returned 429 Too Many Requests — your free-tier quota may be "
                "exhausted. Wait for monthly reset or upgrade your plan."
            )
        r.raise_for_status()
        _track_rapidapi_headers(r)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"RapidAPI connectivity check failed: {e}") from e


# ---------------------------------------------------------------------------
# Tool argument pre-validation
# ---------------------------------------------------------------------------
def validate_tool_args(func, args: dict) -> str | None:
    """Validate args against the function's signature.

    Returns an error message string if validation fails, None if OK.
    """
    sig = inspect.signature(func)
    required = {
        name
        for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    accepted = set(sig.parameters.keys())

    missing = required - set(args.keys())
    unexpected = set(args.keys()) - accepted

    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing required: {', '.join(sorted(missing))}")
        if unexpected:
            parts.append(f"unexpected: {', '.join(sorted(unexpected))}")
        return f"Argument error for {func.__name__}: {'; '.join(parts)}"
    return None


# ===========================================================================
# Tool 1: search_amazon_india
# ===========================================================================
def search_amazon_india(
    keywords: str, max_price: float, min_price: float = 100
) -> dict:
    """
    Search Amazon India. Returns up to 5 products with ASIN/title/price/rating/URL.
    """
    quota_err = _check_rapidapi_quota()
    if quota_err:
        return {"error": quota_err}

    rapidapi_key = _get_runtime_config()["rapidapi_key"]
    url = f"{RAPIDAPI_BASE}/search"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    params = {
        "query": keywords,
        "country": "IN",
        "min_price": str(int(min_price)),
        "max_price": str(int(max_price)),
        "sort_by": "RELEVANCE",
        "page": "1",
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        _track_rapidapi_headers(r)
        r.raise_for_status()
        payload = r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Amazon search API request failed: {e}"}
    except ValueError as e:
        return {"error": f"Amazon search API returned non-JSON: {e}"}

    products = (payload.get("data") or {}).get("products", [])[:5]

    simplified = []
    for p in products:
        simplified.append(
            {
                "asin": p.get("asin"),
                "title": (p.get("product_title") or "")[:180],
                "price": p.get("product_price"),
                "original_price": p.get("product_original_price"),
                "rating": p.get("product_star_rating"),
                "num_ratings": p.get("product_num_ratings"),
                "url": p.get("product_url"),
                "is_prime": p.get("is_prime"),
            }
        )

    return {
        "keywords": keywords,
        "price_range": f"₹{int(min_price)}–₹{int(max_price)}",
        "result_count": len(simplified),
        "products": simplified,
    }


# ===========================================================================
# Tool 2: get_product_details
# ===========================================================================
def get_product_details(asin: str) -> dict:
    """
    Get deep details for one ASIN: description, bullets, availability.
    """
    quota_err = _check_rapidapi_quota()
    if quota_err:
        return {"error": quota_err}

    rapidapi_key = _get_runtime_config()["rapidapi_key"]
    url = f"{RAPIDAPI_BASE}/product-details"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    params = {"asin": asin, "country": "IN"}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        _track_rapidapi_headers(r)
        r.raise_for_status()
        payload = r.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Product details API request failed: {e}"}
    except ValueError as e:
        return {"error": f"Product details API returned non-JSON: {e}"}

    d = payload.get("data") or {}

    about = d.get("about_product") or []
    if isinstance(about, list):
        about = [str(b)[:200] for b in about[:6]]

    description = d.get("product_description") or ""
    if isinstance(description, str) and len(description) > 600:
        description = description[:600] + "…"

    return {
        "asin": asin,
        "title": (d.get("product_title") or "")[:180],
        "price": d.get("product_price"),
        "rating": d.get("product_star_rating"),
        "num_ratings": d.get("product_num_ratings"),
        "availability": d.get("product_availability"),
        "about_product": about,
        "description": description,
        "url": d.get("product_url"),
    }


# ===========================================================================
# Tool 3: calculate_value_score  (pure calculation, no API)
# ===========================================================================
def calculate_value_score(
    price: float, rating: float, num_ratings: int, budget_max: float
) -> dict:
    """
    Combine affordability, rating quality, and review volume into one score.
    """
    # 1) Affordability — peak at 75% of budget (good value without being cheap).
    if price is None or price <= 0:
        affordability = 0
    elif price > budget_max:
        affordability = 0
    else:
        pct = price / budget_max
        affordability = max(0, 100 * (1 - abs(pct - 0.75)))

    # 2) Rating quality — linear 0–5 → 0–100, unknown = 50.
    if rating is None or rating <= 0:
        rating_score = 50
    else:
        rating_score = min(100, (rating / 5.0) * 100)

    # 3) Review volume — log-scale, 1000 reviews = 100.
    if num_ratings is None or num_ratings <= 0:
        volume_score = 20
    else:
        volume_score = min(100, (math.log10(num_ratings + 1) / math.log10(1001)) * 100)

    overall = (affordability * 0.35) + (rating_score * 0.40) + (volume_score * 0.25)

    if overall >= 75:
        verdict = "Strong choice"
    elif overall >= 60:
        verdict = "Good"
    elif overall >= 45:
        verdict = "Mediocre"
    else:
        verdict = "Weak"

    return {
        "overall_score": round(overall, 1),
        "breakdown": {
            "affordability": round(affordability, 1),
            "rating_quality": round(rating_score, 1),
            "review_volume": round(volume_score, 1),
        },
        "weights": {
            "affordability": 0.35,
            "rating_quality": 0.40,
            "review_volume": 0.25,
        },
        "verdict": verdict,
    }


# ===========================================================================
# Tool 4: compose_gift_card_message  (nested LLM call)
# ===========================================================================
def compose_gift_card_message(
    recipient_name: str,
    occasion: str,
    product_title: str,
    relationship: str = "friend",
    tone: str = "warm",
) -> dict:
    """
    Use a secondary Gemini call to write a short personalized card message.
    """
    config = _get_runtime_config()
    gemini_api_key = config["gemini_api_key"]
    gemini_model = config["gemini_model"]

    if not gemini_api_key:
        return {"error": "GEMINI_API_KEY not configured"}

    prompt = f"""Write a short (3-4 sentences), {tone} handwritten-style gift card message.

Recipient:    {recipient_name}
Relationship: {relationship}
Occasion:     {occasion}
Gift:         {product_title}

Rules:
- Do NOT mention the price or where the gift was bought.
- Do NOT sound like marketing copy or a greeting card cliché.
- Feel personal and specific to the gift.
- End with a warm sign-off line like "Love," or "With you always," — the sender will add their name below.

Return ONLY the message text. No preamble, no quotes around it.
"""

    try:
        sub_client = genai.Client(api_key=gemini_api_key)
        response = sub_client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.8),
        )
        message = (response.text or "").strip()
    except Exception as e:
        return {"error": f"Message generation failed: {e}"}

    return {
        "recipient": recipient_name,
        "occasion": occasion,
        "relationship": relationship,
        "tone": tone,
        "message": message,
    }


# ===========================================================================
# Gemini function-calling schemas
# ===========================================================================
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_amazon_india",
        description=(
            "Search Amazon India for real products matching English keywords within a price range. "
            "Returns up to 5 products with ASIN, title, price (in ₹), rating, number of ratings, and URL. "
            "Call this multiple times with different keyword strategies to explore different gift ideas "
            "(e.g. one call for each major interest of the recipient)."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "keywords": types.Schema(
                    type="STRING",
                    description="Concrete English search keywords, e.g. 'premium yoga mat' or 'fantasy novel boxset'.",
                ),
                "max_price": types.Schema(
                    type="NUMBER",
                    description="Maximum price in INR. MUST NOT exceed the user's stated budget.",
                ),
                "min_price": types.Schema(
                    type="NUMBER",
                    description="Minimum price in INR. Defaults to 100 if not specified.",
                ),
            },
            required=["keywords", "max_price"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_product_details",
        description=(
            "Get detailed info for ONE Amazon India product by its ASIN. Returns description, "
            "feature bullets, availability, and full rating. Use this on your top 2-3 candidates "
            "AFTER you've searched — never call this before searching."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "asin": types.Schema(
                    type="STRING",
                    description="Amazon Standard Identification Number from a previous search result.",
                ),
            },
            required=["asin"],
        ),
    ),
    types.FunctionDeclaration(
        name="calculate_value_score",
        description=(
            "Compute a 0-100 value-for-money score combining affordability (vs budget), rating quality, "
            "and review volume. Call this on your final 2-3 candidates to pick the winner objectively. "
            "You MUST parse numbers from the strings returned by search — e.g. '₹1,299.00' → 1299, "
            "'4.3 out of 5 stars' → 4.3."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "price": types.Schema(
                    type="NUMBER", description="Product price in INR as a number."
                ),
                "rating": types.Schema(
                    type="NUMBER", description="Star rating out of 5, e.g. 4.3."
                ),
                "num_ratings": types.Schema(
                    type="INTEGER", description="Number of ratings/reviews."
                ),
                "budget_max": types.Schema(
                    type="NUMBER", description="User's maximum budget in INR."
                ),
            },
            required=["price", "rating", "num_ratings", "budget_max"],
        ),
    ),
    types.FunctionDeclaration(
        name="compose_gift_card_message",
        description=(
            "Compose a short personalized gift card message. Call this EXACTLY ONCE at the very end, "
            "after you have picked the winning product."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "recipient_name": types.Schema(
                    type="STRING", description="Recipient's first name."
                ),
                "occasion": types.Schema(
                    type="STRING",
                    description="e.g. 'birthday', 'wedding', 'anniversary'.",
                ),
                "product_title": types.Schema(
                    type="STRING", description="The chosen product's title."
                ),
                "relationship": types.Schema(
                    type="STRING",
                    description="e.g. 'sister', 'best friend', 'colleague'.",
                ),
                "tone": types.Schema(
                    type="STRING", description="'warm', 'playful', 'formal', etc."
                ),
            },
            required=["recipient_name", "occasion", "product_title"],
        ),
    ),
]


TOOL_REGISTRY = {
    "search_amazon_india": search_amazon_india,
    "get_product_details": get_product_details,
    "calculate_value_score": calculate_value_score,
    "compose_gift_card_message": compose_gift_card_message,
}
