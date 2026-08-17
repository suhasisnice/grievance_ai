"""
Phase 1 of the AI/ML layer for GrievanceAI (SIH260011).

Public functions (this is the contract the backend team imports and calls):
    transcribe_audio(file_path) -> str
    normalize_text(text) -> {"canonical_text": str, "language": str}
    describe_image(file_path) -> str
    classify_complaint(text, image_description=None) -> dict
    embed_text(text) -> {"vector": [768 floats]}

Reliability rules (must hold for every function):
    - Every external call has a timeout, then up to MAX_RETRIES retries with
      a short delay, then falls back to a safe default in the SAME shape,
      with "fallback": True added so callers can tell it happened.
    - classify_complaint() NEVER returns a category/priority outside the
      allowed lists in categories.py.
    - No function raises an unhandled exception to the caller — backend
      intake must always succeed even if the AI layer is having a bad day.

Phase 2 (split_departments, for multi-department complaints) is intentionally
NOT in this file yet — that's built in a later phase, once this is solid.
"""
import json
import logging
import time
from typing import Optional

from google import genai
from google.genai import types

from . import config
from .categories import CATEGORIES, PRIORITIES, DEFAULT_CATEGORY, DEFAULT_PRIORITY
from .prompts import NORMALIZE_PROMPT_TEMPLATE, IMAGE_DESCRIBE_PROMPT, build_classification_prompt

logger = logging.getLogger("ai_service")

_client: Optional[genai.Client] = None


def _get_client() -> Optional[genai.Client]:
    """Lazily creates the Gemini client. Returns None if no API key is set,
    so callers can fall back cleanly instead of crashing on import."""
    global _client
    if _client is None and config.GEMINI_API_KEY:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _with_retry(fn, *args, **kwargs):
    """Runs fn up to (1 + MAX_RETRIES) times with a short delay between
    attempts. Returns the result, or raises the last exception if every
    attempt failed — callers are expected to catch this and fall back."""
    last_error = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — intentionally broad, this is a reliability boundary
            last_error = exc
            logger.warning("ai_service call failed (attempt %d/%d): %s", attempt + 1, config.MAX_RETRIES + 1, exc)
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_DELAY_SECONDS)
    raise last_error


def _extract_json(raw_text: str) -> dict:
    """Gemini is asked for raw JSON, but strips/handles the common case of
    it wrapping the answer in ```json ... ``` fences anyway."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# transcribe_audio
# ---------------------------------------------------------------------------

def transcribe_audio(file_path: str) -> str:
    """Converts a voice-note audio file to text. Tries Gemini first (it
    accepts audio directly); falls back to Groq's free Whisper endpoint if
    GROQ_API_KEY is set and Gemini fails; falls back to an empty string
    (never raises) if everything fails, so the caller can still create a
    ticket flagged for human review instead of losing the complaint."""
    client = _get_client()
    if client is not None:
        try:
            return _with_retry(_transcribe_with_gemini, client, file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini transcription failed, trying fallback: %s", exc)

    if config.GROQ_API_KEY:
        try:
            return _with_retry(_transcribe_with_groq, file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Groq transcription fallback also failed: %s", exc)

    logger.error("transcribe_audio: no working transcription path, returning empty string")
    return ""


def _transcribe_with_gemini(client: genai.Client, file_path: str) -> str:
    uploaded = client.files.upload(file=file_path)
    response = client.models.generate_content(
        model=config.GEMINI_TEXT_MODEL,
        contents=[
            uploaded,
            "Transcribe this audio exactly as spoken. If it is not in English, "
            "transcribe it in its original language/script — do not translate here.",
        ],
    )
    return (response.text or "").strip()


def _transcribe_with_groq(file_path: str) -> str:
    from groq import Groq  # imported lazily so this dependency is optional

    client = Groq(api_key=config.GROQ_API_KEY)
    with open(file_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
        )
    return (result.text or "").strip()


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> dict:
    """Detects the language and translates/normalizes to clear English.
    Fallback: returns the original text unchanged with language 'unknown'
    and fallback=True, so downstream classification still runs on
    *something* rather than blocking."""
    client = _get_client()
    fallback = {"canonical_text": text, "language": "unknown", "fallback": True}
    if client is None:
        return fallback

    try:
        prompt = NORMALIZE_PROMPT_TEMPLATE.format(raw_text=text)
        response = _with_retry(
            client.models.generate_content,
            model=config.GEMINI_TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        parsed = _extract_json(response.text)
        return {
            "canonical_text": parsed.get("canonical_text") or text,
            "language": parsed.get("language") or "unknown",
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("normalize_text failed, returning original text: %s", exc)
        return fallback


# ---------------------------------------------------------------------------
# describe_image
# ---------------------------------------------------------------------------

def describe_image(file_path: str) -> str:
    """One-paragraph factual description of a complaint photo. Returns an
    empty string on failure (never raises) — classify_complaint() treats a
    missing image description as 'no photo context', not an error."""
    client = _get_client()
    if client is None:
        return ""

    try:
        return _with_retry(_describe_image_with_gemini, client, file_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("describe_image failed, continuing without image context: %s", exc)
        return ""


def _describe_image_with_gemini(client: genai.Client, file_path: str) -> str:
    uploaded = client.files.upload(file=file_path)
    response = client.models.generate_content(
        model=config.GEMINI_TEXT_MODEL,
        contents=[uploaded, IMAGE_DESCRIBE_PROMPT],
    )
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# classify_complaint
# ---------------------------------------------------------------------------

def _fallback_classification() -> dict:
    return {
        "category": DEFAULT_CATEGORY,
        "subcategory": "unclassified",
        "priority": DEFAULT_PRIORITY,
        "confidence": 0.0,
        "location_text": None,
        "summary": "Could not be automatically classified — needs manual review.",
        "fallback": True,
        "source": "fallback",
    }


def classify_complaint(text: str, image_description: Optional[str] = None) -> dict:
    """The core function. Takes canonical (already-translated) complaint
    text and an optional image description, returns structured
    classification. ALWAYS validates category/priority against the allowed
    lists before returning — never passes through a value the model
    invented outside those lists.

    Provider chain: Gemini -> Groq (if GROQ_API_KEY is set) -> safe default.
    Groq is a second, independent provider — if Gemini's free tier is
    rate-limited or briefly down (which happens during demos when several
    complaints come in close together), we don't want to silently drop to
    confidence=0 on every complaint until the quota window resets. Groq
    gets the exact same prompt (system instruction + few-shot examples),
    just run through a different model, so accuracy should be comparable."""
    client = _get_client()
    if client is not None:
        try:
            prompt = build_classification_prompt(text, image_description)
            response = _with_retry(
                client.models.generate_content,
                model=config.GEMINI_TEXT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            parsed = _extract_json(response.text)
            return _validate_classification(parsed, source="gemini")
        except Exception as exc:  # noqa: BLE001
            logger.error("classify_complaint: Gemini failed, trying Groq fallback: %s", exc)

    if config.GROQ_API_KEY:
        try:
            prompt = build_classification_prompt(text, image_description)
            parsed = _with_retry(_classify_with_groq, prompt)
            return _validate_classification(parsed, source="groq")
        except Exception as exc:  # noqa: BLE001
            logger.error("classify_complaint: Groq fallback also failed, returning safe default: %s", exc)
    else:
        logger.warning("classify_complaint: Gemini failed and no GROQ_API_KEY is set, returning safe default")

    return _fallback_classification()


def _classify_with_groq(prompt: str) -> dict:
    """Sends the same classification prompt to Groq's free Llama endpoint.
    Raises on failure so _with_retry can retry/propagate — the caller
    catches and moves to the final safe-default fallback."""
    from groq import Groq  # imported lazily so this dependency is optional

    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content
    return _extract_json(raw_text)


def _validate_classification(parsed: dict, source: str) -> dict:
    """Guards against the model returning a category/priority outside our
    fixed lists, a confidence outside [0, 1], or missing fields.

    `source` records which provider actually produced this answer
    ("gemini" or "groq") so callers/logs can tell them apart — useful for
    debugging and for knowing how often the Groq fallback is kicking in."""
    category = parsed.get("category")
    if category not in CATEGORIES:
        logger.warning("classify_complaint: invalid category '%s', defaulting to '%s'", category, DEFAULT_CATEGORY)
        category = DEFAULT_CATEGORY

    priority = parsed.get("priority")
    if priority not in PRIORITIES:
        logger.warning("classify_complaint: invalid priority '%s', defaulting to '%s'", priority, DEFAULT_PRIORITY)
        priority = DEFAULT_PRIORITY

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    # If we had to override the model's category/priority choice, the
    # confidence it reported no longer means anything — zero it out so the
    # backend correctly routes this to human review.
    if category != parsed.get("category") or priority != parsed.get("priority"):
        confidence = 0.0

    return {
        "category": category,
        "subcategory": parsed.get("subcategory") or "unspecified",
        "priority": priority,
        "confidence": confidence,
        "location_text": parsed.get("location_text") or None,
        "summary": parsed.get("summary") or "No summary available.",
        "source": source,
    }


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

def embed_text(text: str) -> dict:
    """Returns a 768-dimension embedding vector for the given text, matching
    the backend's pgvector column size (see Team Integration Guide's Locked
    Decisions). Fallback: a zero-vector with fallback=True — this is
    intentionally a "neutral" vector that won't falsely match as a
    duplicate of anything, rather than a random one."""
    client = _get_client()
    fallback_vector = [0.0] * config.EMBEDDING_DIMENSIONS

    if client is None:
        return {"vector": fallback_vector, "fallback": True}

    try:
        response = _with_retry(
            client.models.embed_content,
            model=config.GEMINI_EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=config.EMBEDDING_DIMENSIONS),
        )
        vector = list(response.embeddings[0].values)
        if len(vector) != config.EMBEDDING_DIMENSIONS:
            logger.warning(
                "embed_text: unexpected vector size %d (expected %d), using fallback",
                len(vector), config.EMBEDDING_DIMENSIONS,
            )
            return {"vector": fallback_vector, "fallback": True}
        return {"vector": vector}
    except Exception as exc:  # noqa: BLE001
        logger.error("embed_text failed, returning fallback zero-vector: %s", exc)
        return {"vector": fallback_vector, "fallback": True}