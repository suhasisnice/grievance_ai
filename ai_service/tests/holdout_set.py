"""
HELD-OUT TEST SET — 30 complaints the model has NEVER seen in its prompt.

This is the honest accuracy measurement for the classifier. run_examples.py
grades the model on the same 15 examples that are baked into its own prompt
(an open-book exam, which is why it scores 15/15). This file is the closed-book
exam.

RULES FOR THIS FILE:
  - NOTHING in here may ever be copied into examples.py. The moment a complaint
    appears in the prompt, it stops being a valid test and the score becomes
    meaningless. If you want more few-shot examples, write new ones.
  - Labels are human-reviewed, not model-generated. If you disagree with a
    label, change the label — don't change the model to match a bad label.
  - Entries tagged "borderline": True are cases where two labels are both
    defensible. They're scored separately so a coin-flip disagreement doesn't
    read as a model failure.

Generated with Gemini Pro, then reviewed by hand.
"""

HOLDOUT_SET = [
    # 1. Plain English - Critical (Open manhole near school - life danger)
    {
        "text": "EMERGENCY open manhole right outside St Marys primary school gate kids are running around someone will fall in cover it NOW",
        "expected_category": "drainage",
        "expected_priority": "critical",
        "note": "Open manhole at school entrance creates immediate fatal hazard for children."
    },
    # 2. Hinglish - Critical (Live wire hanging - life danger, landmark reference)
    {
        "text": "bhai Gupta sweets ke samne transformer se live wire latak raha hai sparks nikal rahe barish me koi mar jayega jaldi aao!!",
        "expected_category": "electricity",
        "expected_priority": "critical",
        "note": "Sparking exposed live electric wire in rain is an active life hazard; landmark-based location."
    },
    # 3. Kannada - Critical (Active waterlogging/flooding blocking hospital)
    {
        "text": "ಸರ್ಕಾರಿ ಆಸ್ಪತ್ರೆ ಎದುರು ಮಳೆ ನೀರು ಪೂರ್ತಿ ತುಂಬಿ ಆಂಬುಲೆನ್ಸ್ ಒಳಗೆ ಹೋಗೋಕೆ ಆಗ್ತಿಲ್ಲ ಕೂಡಲೇ ನೀರು ತೆರವು ಮಾಡಿ",
        "expected_category": "drainage",
        "expected_priority": "critical",
        "note": "Severe flooding blocking hospital access/ambulances represents life safety critical issue."
    },
    # 4. Pure Hindi - Critical (Bridge/culvert slab collapsed on main road)
    {
        "text": "मेन रोड का छोटा पुलिया अचानक धंस गया है बहुत बड़ा गड्ढा बन गया कोई भी गाड़ी अंदर गिर सकती है जान का खतरा है",
        "expected_category": "roads",
        "expected_priority": "critical",
        "note": "Sudden road/culvert structural collapse posing imminent threat of vehicular fatalities."
    },
    # 5. Tamil - High (Sewage contamination mixing into drinking water)
    {
        "text": "எங்க தெருவுல குடிநீர் குழாயில் சாக்கடை தண்ணி கலந்து வருது பயங்கர நாத்தம் குடிக்கவே முடியல 3 நாளா",
        "expected_category": "water_supply",
        "expected_priority": "high",
        "note": "Contaminated drinking water mixed with sewage causes serious ongoing public health outbreak risks.",
        "borderline": True,
        "borderline_note": "critical is equally defensible — sewage in the drinking line is an immediate health hazard, not a future one. Accept either.",
        "also_accept_priority": ["critical"],
    },
    # 6. Hinglish - High (Major water pipe burst flooding road)
    {
        "text": "main pipeline burst ho gaya near Reliance fresh pura paani sadak pe waste ho raha aur supply cut hai subah se",
        "expected_category": "water_supply",
        "expected_priority": "high",
        "note": "Major burst pipeline causing severe civic water disruption; landmark reference."
    },
    # 7. Plain English - High (Deep trenches dug up and left open across lane)
    {
        "text": "they dug whole 4th cross for gas pipeline 2 weeks ago and left open big ditches cars getting stuck daily fix the road",
        "expected_category": "roads",
        "expected_priority": "high",
        "note": "Dug up major access road causing ongoing disruption and vehicle entrapment."
    },
    # 8. Telugu - High (Rotting garbage dump near residential area causing fever/dengue)
    {
        "text": "మా కాలనీ లో చెత్త కుప్పలు పేరుకుపోయి వారం అయింది దోమలు విపరీతంగా పెరిగి డెంగ్యూ జ్వరాలు వస్తున్నాయి",
        "expected_category": "garbage",
        "expected_priority": "high",
        "note": "Uncollected decomposing garbage leading to active vector-borne disease outbreak."
    },
    # 9. Pure Hindi - High (Entire colony pitch dark for a week, female safety issue)
    {
        "text": "वार्ड 14 में पिछले एक हफ्ते से एक भी स्ट्रीट लाइट नहीं जल रही रात को महिलाओं और बच्चों का निकलना मुश्किल हो गया है",
        "expected_category": "streetlights",
        "expected_priority": "high",
        "note": "Systemic blackout of entire ward streetlights posing serious ongoing safety and security risks."
    },
    # 10. Hinglish - High (Drain overflow blocking market access)
    {
        "text": "Nala pura choke ho chuka hai sara ganda pani dukan ke andar ghus raha hai badboo se koi customer nahi aa raha",
        "expected_category": "drainage",
        "expected_priority": "high",
        "note": "Severe blocked sewer overflow entering commercial establishments causing major disruption."
    },
    # 11. Plain English - Low (Park maintenance, broken swing/benches)
    {
        "text": "municipal park children play area swings are broken and benches are damaged please send maintenance staff to repair",
        "expected_category": "parks",
        "expected_priority": "low",
        "note": "Minor amenity repair and routine park equipment maintenance.",
        "borderline": True,
        "borderline_note": "few-shot example #7 labels broken park swings as MEDIUM (children could fall). That example doesn't state an injury risk either. Expect the model to say medium — accept it.",
        "also_accept_priority": ["medium"],
    },
    # 12. Hinglish - Medium (Garbage truck not coming regularly)
    {
        "text": "kachra gadi alternate day bhi nahi aa rahi 3rd street me kripya driver ko regular aane bolo",
        "expected_category": "garbage",
        "expected_priority": "medium",
        "note": "Irregular doorstep garbage collection service; standard municipal grievance."
    },
    # 13. Plain English - Low (Streetlight blinking/faulty)
    {
        "text": "pole number 42 streetlight is flickering continuously like a strobe light please change the bulb",
        "expected_category": "streetlights",
        "expected_priority": "low",
        "note": "Single localized bulb defect; minor non-hazardous issue.",
        "borderline": True,
        "borderline_note": "few-shot example #3 labels a dead streetlight as MEDIUM. A flickering one is arguably the same class of fault. Accept either.",
        "also_accept_priority": ["medium"],
    },
    # 14. Pure Hindi - Medium (Low water pressure in locality)
    {
        "text": "सेक्टर 9 में पानी का प्रेशर बहुत कम आ रहा है पहली मंजिल पर भी पानी नहीं चढ़ रहा मोटर चलाने के बाद भी",
        "expected_category": "water_supply",
        "expected_priority": "medium",
        "note": "Low piped municipal water pressure affecting domestic supply."
    },
    # 15. Kannada - Medium (Potholes on residential cross road)
    {
        "text": "ನಮ್ಮ 2ನೇ ಕ್ರಾಸ್ ರಸ್ತೆಯಲ್ಲಿ ತುಂಬಾ ಗುಂಡಿಗಳು ಬಿದ್ದಿವೆ ದ್ವಿಚಕ್ರ ವಾಹನ ಸವಾರರಿಗೆ ತೊಂದರೆ ಆಗ್ತಿದೆ ಡಾಂಬರೀಕರಣ ಮಾಡಿ",
        "expected_category": "roads",
        "expected_priority": "medium",
        "note": "Standard pothole repair request on residential cross road."
    },
    # 16. Tamil - High (Public toilet unsanitary condition at bus stand)
    {
        "text": "பஸ் ஸ்டாண்ட் பக்கத்துல இருக்கிற பொது கழிப்பறை ரொம்ப அசுத்தமா இருக்கு ஒழுங்கா சுத்தம் பண்றதே இல்ல",
        "expected_category": "sanitation",
        "expected_priority": "high",
        "note": "Severe sanitation breakdown at a major transit hub (bus stand) impacting high public footfall.",
        "borderline": True,
        "borderline_note": "medium fits the stated rubric better — a dirty toilet is unpleasant but not 'major disruption' or 'dangerous soon'. Accept either.",
        "also_accept_priority": ["medium"],
    },
    # 17. Hinglish - Low (Overgrown weeds in neighborhood park)
    {
        "text": "behind Ganesh temple jo park hai waha grass bohot lambi ho gayi hai koi safai nahi karta evening walk impossible",
        "expected_category": "parks",
        "expected_priority": "low",
        "note": "Minor lawn maintenance and grass cutting request; landmark reference."
    },
    # 18. Telugu - Medium (Frequent voltage fluctuations)
    {
        "text": "మా వీధిలో రాత్రిపూట విపరీతమైన వోల్టేజ్ హెచ్చుతగ్గులు వస్తున్నాయి ఫ్రిజ్ టీవీ పాడైపోయేలా ఉంది",
        "expected_category": "electricity",
        "expected_priority": "medium",
        "note": "Power quality / voltage fluctuation grievance without immediate sparking/fire danger."
    },
    # 19. Plain English - Medium (Road median divider broken)
    {
        "text": "The concrete median divider on ring road near IOCL petrol pump has been damaged by some truck please reconstruct",
        "expected_category": "roads",
        "expected_priority": "medium",
        "note": "Damaged road median structure needing civil repair; landmark reference."
    },
    # 20. Hinglish - Medium (Public urinal cleaning required)
    {
        "text": "Vegetable market ke paas public urinal se bohot foul smell aa rahi hai disinfectent spray karwao",
        "expected_category": "sanitation",
        "expected_priority": "medium",
        "note": "Sanitation/hygiene maintenance near local public marketplace."
    },
    # 21. Plain English - High (Ambiguous: Garbage dumping vs Road blockage, no location)
    {
        "text": "someone dumped huge construction debris and broken tiles right in the middle of the road blocking traffic",
        "expected_category": "roads",
        "expected_priority": "high",
        "note": "Active traffic obstruction and collision hazard from dumped debris in carriageway.",
        "borderline": True,
        "borderline_note": "deliberately ambiguous — garbage is equally defensible since the root issue is illegal dumping. Accept either.",
        "also_accept_category": ["garbage"],
    },
    # 22. Hinglish - Medium (Ambiguous: Sanitation vs Drainage, foul smell/standing water)
    {
        "text": "gali me pura sadand faili hui hai ganda pani jama hai na safari wale aate na koi dekhne wala",
        "expected_category": "sanitation",
        "expected_priority": "medium",
        "note": "Valid civic sanitation/stagnant water issue without immediate acute emergency.",
        "borderline": True,
        "borderline_note": "deliberately ambiguous — standing dirty water is squarely drainage territory too. Accept either.",
        "also_accept_category": ["drainage"],
    },
    # 23. Pure Hindi - Medium (Ambiguous: Parks vs Streetlights)
    {
        "text": "नेहरू पार्क के अंदर की सारी लाइट्स पिछले महीने से बंद पड़ी हैं रात में वॉक करना असंभव हो गया है",
        "expected_category": "parks",
        "expected_priority": "medium",
        "note": "Ambiguous between parks department and streetlight maintenance; pertains to internal park infrastructure.",
        "borderline": True,
        "borderline_note": "deliberately ambiguous — the asset is a light, the owner is the parks dept. Accept either.",
        "also_accept_category": ["streetlights"],
    },
    # 24. Plain English - Low (No location mentioned, vague minor road issue)
    {
        "text": "road quality is very bad lots of dust everywhere please do something",
        "expected_category": "roads",
        "expected_priority": "low",
        "note": "Vague general complaint about dust/road quality with no specific location or clear actionable target."
    },
    # 25. Hinglish - Low (No location mentioned, minor water grievance)
    {
        "text": "paani ka timing change karo subah 4 baje kaun uthta hai paani bharne",
        "expected_category": "water_supply",
        "expected_priority": "low",
        "note": "Minor administrative timing grievance without any locality details or systemic breakdown."
    },
    # 26. Pure Hindi - Low (Vague street cleanliness, no location)
    {
        "text": "सड़क पर झाड़ू लगाने वाले समय पर नहीं आते सिर्फ खानापूर्ति करते हैं",
        "expected_category": "sanitation",
        "expected_priority": "low",
        "note": "Vague sweeping grievance lacking specific street, ward, or identifiable location details."
    },
    # 27. Plain English - Low (Angry/emotional rant, zero specific civic issue stated)
    {
        "text": "WORST MUNICIPAL CORPORATION EVER!!! WE PAY TAXES FOR WHAT??? TOTALLY USELESS STAFF CORRUPT OFFICERS SHAME ON YOU ALL",
        "expected_category": "other",
        "expected_priority": "low",
        "note": "Pure emotional venting/rant with no actionable civic issue or location stated."
    },
    # 28. Hinglish - Low (Angry rant against local authorities, no problem specified)
    {
        "text": "kya bakwas service hai tum logo ki phone uthate nahi complaint number leke so jate ho sharam aani chahiye sab chor hai",
        "expected_category": "other",
        "expected_priority": "low",
        "note": "Angry customer service complaint lacking any specific civic infrastructure category."
    },
    # 29. Kannada - Low (Generic anger/scolding, no department or issue named)
    {
        "text": "ನಿಮ್ಮ ಆಫೀಸ್‌ಗೆ ಎಷ್ಟು ಸಲ ಕಾಲ್ ಮಾಡಿದ್ರೂ ವೇಸ್ಟ್ ಯಾರು ಸರಿಯಾಗಿ ಕೆಲಸ ಮಾಡಲ್ಲ ಜನರಿಗೆ ಮೋಸ ಮಾಡ್ತಿದ್ದೀರಾ ಅಷ್ಟೇ",
        "expected_category": "other",
        "expected_priority": "low",
        "note": "General administrative frustration expressing anger without specifying any municipal issue."
    },
    # 30. Hinglish - Low (Very short / ultra-terse non-specific feedback)
    {
        "text": "bekar nagar nigam",
        "expected_category": "other",
        "expected_priority": "low",
        "note": "Three-word non-specific negative review with no civic problem, location, or department identified."
    }
]
