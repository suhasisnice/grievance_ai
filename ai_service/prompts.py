"""
All prompt text lives here — nowhere else in this package. If you or your
teammate refine wording with Gemini/ChatGPT, this is the only file that
should change.
"""
import json
from .examples import FEW_SHOT_EXAMPLES
from .categories import CATEGORIES, PRIORITIES

NORMALIZE_PROMPT_TEMPLATE = """You are normalizing a citizen's civic complaint for an Indian municipal grievance system.

The input text may be in English, Hindi, a regional Indian language, or Hinglish (code-mixed Hindi-English).

Task:
1. Detect the primary language of the input (use ISO-style short names like "english", "hindi", "hinglish", "kannada", etc.)
2. Rewrite the complaint in clear, simple English, preserving every factual detail (what happened, where, since when, who is affected). Do not add information that isn't in the original. Do not shorten or summarize — just translate/normalize.

Input complaint:
\"\"\"{raw_text}\"\"\"

Respond with ONLY a JSON object in this exact shape, no other text:
{{"canonical_text": "<the normalized English text>", "language": "<detected language>"}}
"""

IMAGE_DESCRIBE_PROMPT = """You are looking at a photo submitted alongside a civic complaint (e.g. a pothole, damaged pipe, garbage pile, broken streetlight, etc.) in an Indian city.

Describe factually, in one paragraph, what civic issue (if any) is visible in the image. Mention visible damage, hazards, or conditions relevant to a municipal department. Do not speculate about things not visible in the image. If the image does not show any civic issue, say so plainly.
"""

CLASSIFICATION_SYSTEM_INSTRUCTION = f"""Priority guidance:
- "critical": a person could be killed or seriously injured in the next few hours if nobody acts. Sparking or fallen live wires, open manholes, structural collapse, floodwater blocking hospital access.
- "high": significant ongoing disruption to many people, or a hazard that becomes dangerous if it is ignored for days. Whole-area outages, contaminated drinking water, disease-carrying conditions, an obstruction blocking a road.
- "medium": a real, valid civic issue that is degrading service or amenity, but nobody is in danger and the disruption is contained.
- "low": minor or cosmetic issues, routine maintenance requests, and vague complaints where no specific problem, place, or affected person is identified.

Priority is about CONSEQUENCE, not about how upset the citizen sounds. Angry
capital letters, insults, and exclamation marks carry no priority weight. A
calmly-worded live wire is critical; a furious complaint about street sweepers
is low.

Apply these tie-breakers, they override the general descriptions above:
- Damage to property, water, or money with no injury risk is NEVER critical. A burst pipeline wasting water and cutting supply is high, not critical — reserve critical for danger to people.
- Ordinary potholes and rough road surface on a residential or side road are medium. They become high only if the complaint states accidents, injuries, an arterial road, or vehicles becoming stuck.
- If a complaint names no specific location AND describes no harm to any person, it is low — even when the underlying issue is genuine. "The road is dusty" and "the sweepers do a bad job" are low.
- Dirty or unmaintained public facilities are medium. They are high only when the complaint states a resulting health impact, such as disease or contamination.
- If you find yourself weighing two adjacent levels, choose the LOWER one. Over-escalation floods officers with false alarms and makes the real emergencies harder to see.

Confidence guidance:
- Return a confidence score (0.0 to 1.0) for how likely it is that BOTH your category and your priority match what a trained municipal officer would choose.
- Judge the two separately, then return the LOWER of the two scores. Being sure of the department does not make you sure of the urgency — priority is the harder call and it decides the score.
- If you weighed two adjacent priority levels before picking one, your confidence MUST be at most 0.6. That disagreement is exactly what a human reviewer is there to settle.
- If the complaint is vague, off-topic, or could belong to either of two categories, stay below 0.5 rather than guessing confidently.
- Confidence measures how likely your LABEL is correct. It does not measure how clearly the text describes a problem, how serious the problem is, or how strongly the citizen feels. A vividly described complaint you are unsure how to rank is a LOW confidence answer.
- Do not inflate confidence just because you produced an answer — an answer can be your best guess while still being uncertain.
"""

CLASSIFICATION_JSON_INSTRUCTION_TEMPLATE = """
Now classify this new complaint. Respond with ONLY a JSON object in exactly this shape, no other text, no markdown fences:
{"category": "...", "subcategory": "...", "priority": "...", "confidence": 0.0, "location_text": "..." or null, "summary": "..."}

Complaint:
\"\"\"__COMPLAINT_TEXT__\"\"\"
__IMAGE_CONTEXT__
"""


def build_classification_prompt(complaint_text: str, image_description: str | None = None) -> str:
    """Assembles the full classification prompt: system instruction + few-shot
    examples + the new complaint to classify.

    Uses plain string substitution (not str.format) because the JSON example
    embedded in the template contains literal curly braces, which would
    collide with .format()'s placeholder syntax.
    """
    examples_block = "\n\n".join(
        f"Complaint: \"\"\"{ex['text']}\"\"\"\nOutput: {json.dumps(ex['output'])}"
        for ex in FEW_SHOT_EXAMPLES
    )
    image_context = f"\nAdditional context from an attached photo: {image_description}" if image_description else ""
    final_instruction = (
        CLASSIFICATION_JSON_INSTRUCTION_TEMPLATE
        .replace("__COMPLAINT_TEXT__", complaint_text)
        .replace("__IMAGE_CONTEXT__", image_context)
    )
    return CLASSIFICATION_SYSTEM_INSTRUCTION + "\n\n" + examples_block + "\n\n" + final_instruction
