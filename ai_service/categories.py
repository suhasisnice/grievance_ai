"""
The ONLY allowed values for category and priority — this is the single
source of truth referenced in the Team Integration Guide's "Locked Decisions".
classify_complaint() validates its output against these lists before
returning anything to the backend. If the model returns something outside
these lists, it gets mapped to a safe default, never passed through.
"""

CATEGORIES = [
    "water_supply",
    "roads",
    "sanitation",
    "electricity",
    "streetlights",
    "drainage",
    "garbage",
    "parks",
    "other",
]

PRIORITIES = ["low", "medium", "high", "critical"]

DEFAULT_CATEGORY = "other"
DEFAULT_PRIORITY = "medium"

# Confidence below this means the backend should flag needs_human_review=true.
# (Enforcement of that flag happens in the backend, not here — this constant
# is just kept in sync for reference.)
HUMAN_REVIEW_CONFIDENCE_THRESHOLD = 0.6
