"""
Few-shot examples used inside the classification prompt (see prompts.py).
These are baked into every classify_complaint() call so the model has
solved examples to pattern-match against, instead of classifying cold.

Keep this list to realistic, varied Indian civic complaints: a mix of
English, Hindi, and Hinglish (code-mixed), some vague, some clearly urgent,
and one that spans two departments at once. If you (or your teammate)
refine these with Gemini/ChatGPT, replace the whole list — don't hand-edit
piecemeal, since the prompt is tuned against the set as a whole.
"""

FEW_SHOT_EXAMPLES = [
    {
        "text": "Water pipe burst near the market on MG Road, water is flooding the street since this morning.",
        "output": {
            "category": "water_supply",
            "subcategory": "pipe_burst",
            "priority": "high",
            "confidence": 0.55,
            "location_text": "MG Road market",
            "summary": "Burst water pipe near MG Road market is flooding the street.",
        },
    },
    {
        "text": "Sadak me bahut bada gadda hai, do din pehle scooter gir gaya tha kisi ka.",
        "output": {
            "category": "roads",
            "subcategory": "pothole",
            "priority": "high",
            "confidence": 0.88,
            "location_text": None,
            "summary": "Large pothole reported; a scooter accident already occurred nearby.",
        },
    },
    {
        "text": "Streetlight bahut din se band hai humare gali mein, raat ko andhera rehta hai.",
        "output": {
            "category": "streetlights",
            "subcategory": "light_not_working",
            "priority": "medium",
            "confidence": 0.85,
            "location_text": None,
            "summary": "Streetlight has been non-functional for a while, causing darkness at night.",
        },
    },
    {
        "text": "garbage not collected for one week near our apartment, smell is very bad now",
        "output": {
            "category": "garbage",
            "subcategory": "collection_missed",
            "priority": "medium",
            "confidence": 0.9,
            "location_text": None,
            "summary": "Garbage hasn't been collected for a week, causing a bad smell near the apartment.",
        },
    },
    {
        "text": "power cut ho raha hai baar baar, transformer se spark bhi nikal raha tha kal raat",
        "output": {
            "category": "electricity",
            "subcategory": "transformer_fault",
            "priority": "critical",
            "confidence": 0.9,
            "location_text": None,
            "summary": "Frequent power cuts and a transformer sparked last night — possible fire hazard.",
        },
    },
    {
        "text": "drainage water is overflowing onto the road near the school gate every time it rains",
        "output": {
            "category": "drainage",
            "subcategory": "overflow",
            "priority": "high",
            "confidence": 0.87,
            "location_text": "near the school gate",
            "summary": "Drainage overflow near a school gate during rain — safety concern for children.",
        },
    },
    {
        "text": "park ka swing tuta hua hai bahut time se, bachche gir sakte hai",
        "output": {
            "category": "parks",
            "subcategory": "damaged_equipment",
            "priority": "medium",
            "confidence": 0.83,
            "location_text": None,
            "summary": "A broken swing in the park has been unrepaired for a while, risk to children.",
        },
    },
    {
        "text": "water problem",
        "output": {
            "category": "water_supply",
            "subcategory": "unspecified",
            "priority": "low",
            "confidence": 0.35,
            "location_text": None,
            "summary": "Citizen reported a water-related issue but did not provide details.",
        },
    },
    {
        "text": "Bohot dino se paani nahi aa raha humare area mein, poori building pareshan hai, please jaldi dekhiye",
        "output": {
            "category": "water_supply",
            "subcategory": "no_supply",
            "priority": "high",
            "confidence": 0.86,
            "location_text": None,
            "summary": "No water supply for several days, affecting the whole building.",
        },
    },
    {
        "text": "There is an open manhole right outside City Hospital's main gate, very dangerous at night, someone could fall in.",
        "output": {
            "category": "drainage",
            "subcategory": "open_manhole",
            "priority": "critical",
            "confidence": 0.93,
            "location_text": "City Hospital main gate",
            "summary": "Open, unmarked manhole outside City Hospital's main gate poses an immediate safety risk.",
        },
    },
    {
        "text": "sadak toot gayi hai kyunki niche ki paani ki pipe leak ho rahi thi, ab bada gadda ban gaya hai aur paani bhi bah raha hai",
        "output": {
            "category": "roads",
            "subcategory": "road_damage_from_leak",
            "priority": "critical",
            "confidence": 0.8,
            "location_text": None,
            "summary": "A leaking underground water pipe has damaged the road surface, creating a large pothole with active water flow — likely needs both Water Board and Roads Dept.",
        },
    },
    {
        "text": "streetlight ka pole jhuka hua hai, tez hawa mein gir sakta hai, please fix soon",
        "output": {
            "category": "streetlights",
            "subcategory": "leaning_pole",
            "priority": "high",
            "confidence": 0.84,
            "location_text": None,
            "summary": "A leaning streetlight pole risks falling in strong wind and needs urgent repair.",
        },
    },
    {
        "text": "not happy with the service overall, things could be better",
        "output": {
            "category": "other",
            "subcategory": "general_feedback",
            "priority": "low",
            "confidence": 0.3,
            "location_text": None,
            "summary": "Vague general feedback with no specific civic issue described.",
        },
    },
    {
        "text": "sabzi mandi ke paas kachra bahut jama ho gaya hai, machariyan bhi ho rahi hai, health issue ho sakta hai",
        "output": {
            "category": "garbage",
            "subcategory": "waste_accumulation",
            "priority": "high",
            "confidence": 0.88,
            "location_text": "sabzi mandi (vegetable market)",
            "summary": "Accumulated garbage near the vegetable market is attracting mosquitoes — a health risk.",
        },
    },
    {
        "text": "Electricity pole spark kar raha tha near the bus stand, log dar gaye, bahut dangerous lag raha tha",
        "output": {
            "category": "electricity",
            "subcategory": "sparking_pole",
            "priority": "critical",
            "confidence": 0.92,
            "location_text": "near the bus stand",
            "summary": "A sparking electricity pole near the bus stand alarmed bystanders — immediate safety hazard.",
        },
    },
    {
        "text": "the roads around our locality are in bad condition generally, nobody ever comes to look at them",
        "output": {
            "category": "roads",
            "subcategory": "general_condition",
            "priority": "low",
            "confidence": 0.4,
            "location_text": None,
            "summary": "General complaint about poor road condition; no specific place or damage identified.",
        },
    },
    {
        "text": "community hall ke bahar ka public toilet theek se saaf nahi hota, smell aata rehta hai",
        "output": {
            "category": "sanitation",
            "subcategory": "public_toilet_maintenance",
            "priority": "medium",
            "confidence": 0.58,
            "location_text": "outside the community hall",
            "summary": "Public toilet outside the community hall is not being cleaned regularly.",
        },
    },
]
