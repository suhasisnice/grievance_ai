"""
HELD-OUT TEST SET — 100 complaints the model has NEVER seen in its prompt.

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

Cases 1-30: original set, generated with Gemini Pro, reviewed by hand.
Cases 31-100: batch 1 from a teammate (70 cases), originally labeled with an
11-category scheme that didn't match CATEGORIES in categories.py (e.g.
"Water Supply & Sewage", "Public Health & Sanitation", "Town Planning &
Infrastructure"). Remapped case-by-case onto the locked 9-category taxonomy
using precedent from examples.py (open manholes -> drainage, toilets ->
sanitation, playground/park equipment -> parks); "other" used only where no
real municipal department among the 9 fits (spam, staff conduct, private
property disputes, noise pollution — all outside civic-engineering scope).
Priorities lowercased to match PRIORITIES. Original per-case reasoning kept
in "note".
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
    },

    # ---------------------------------------------------------------------
    # Cases 31-100: batch 1 from a teammate (70 cases). See the module
    # docstring above for the category-remapping methodology.
    # ---------------------------------------------------------------------

    # 31. Malayalam - Medium (Clean drinking water leaking and wasting from burst pipe in fron...)
    {
        "text": "ഞങ്ങളുടെ വീടിന് മുന്നിലെ പൈപ്പ് പൊട്ടി രണ്ട് ദിവസമായി ശുദ്ധജലം പാഴായി ഒഴുകുന്നു.",
        "expected_category": "water_supply",
        "expected_priority": "medium",
        "note": "Priority is Medium because it is persistent clean water wastage without immediate property inundation or structural threat.",
    },

    # 32. Marathi - Medium (Overflowing public garbage bin spreading waste and foul odor nea...)
    {
        "text": "शिवाजी नगर कमान जवळ कचऱ्याची कुंडी पूर्ण भरून रस्त्यावर कचरा पसरला आहे, दुर्गंधी सुटली आहे.",
        "expected_category": "garbage",
        "expected_priority": "medium",
        "note": "Priority is Medium as overflowing garbage causes foul odor and public nuisance without hazardous biohazard blockage.",
    },

    # 33. Bengali - Critical (Snapped live electrical wire lying on pooled water posing an imm...)
    {
        "text": "রাস্তার ধারের বৈদ্যুতিক তার ছিঁড়ে জলের উপর পড়ে আছে, যেকোনো সময় বড় দুর্ঘটনা ঘটতে পারে।",
        "expected_category": "electricity",
        "expected_priority": "critical",
        "note": "Priority is Critical due to immediate risk of electrocution and public fatality from a live conductor on water.",
    },

    # 34. Odia - High (Collapsed main drain causing heavy dirty water stagnation and co...)
    {
        "text": "ମୁଖ୍ୟ ଡ୍ରେନ ଭାଙ୍ଗିଯିବାରୁ ସବୁ ମଇଳା ପାଣି ରାସ୍ତାରେ ଜମି ରହିଛି, ଯାତାୟାତ ସମ୍ପୂର୍ଣ୍ଣ ବନ୍ଦ।",
        "expected_category": "drainage",
        "expected_priority": "high",
        "note": "Priority is High because dirty water overflow has completely blocked vehicular and pedestrian movement.",
    },

    # 35. English - Medium (Pedestrian footpath obstructed by abandoned construction debris ...)
    {
        "text": "Construction debris and concrete blocks left in the middle of pedestrian walking pathway.",
        "expected_category": "roads",
        "expected_priority": "medium",
        "note": "Priority is Medium because it blocks a walkway but does not immediately endanger vehicular traffic or lives.",
    },

    # 36. Hinglish - Low (Loudspeaker noise pollution late at night near Galaxy Banquet ha...)
    {
        "text": "Galaxy Banquet hall ke paas raat 1:30 baje tak loud speakers baj rahe hain, senior citizens cannot sleep.",
        "expected_category": "other",
        "expected_priority": "low",
        "note": "Priority is Low under civic grading since nighttime noise nuisance requires routine regulation rather than emergency dispatch.",
    },

    # 37. English - High (Cracked tree branch hanging dangerously over active school bus r...)
    {
        "text": "Large roadside gulmohar branch cracked and dangling precariously over the busy school bus route.",
        "expected_category": "parks",
        "expected_priority": "high",
        "note": "Priority is High (not Critical yet) because the branch is dangling with imminent risk of falling on a transit route, but hasn't collapsed yet.",
    },

    # 38. English - Low (Drop in tap water pressure with drain gurgling in 4th cross lane...)
    {
        "text": "Sudden sharp drop in water pressure across the entire 4th cross lane accompanied by gurgling noise in drains.",
        "expected_category": "water_supply",
        "expected_priority": "low",
        "borderline": True,
        "note": "Priority is Low and confidence is 0.41 due to vague dual-symptom description with no immediate contamination or overflow.",
    },

    # 39. English - Low (Commercial promotion for home painting and pest control services...)
    {
        "text": "Special discount on house painting and pest control services this weekend! Call 9876543210 for free inspection.",
        "expected_category": "other",
        "expected_priority": "low",
        "note": "Priority is Low because non-civic messages and spam are always categorized with lowest operational priority.",
    },

    # 40. Hindi - Critical (Uncovered manhole on an unlit street presenting severe and immed...)
    {
        "text": "बिना ढक्कन का खुला मैनहोल पड़ा है जहां कोई स्ट्रीट लाइट भी नहीं जल रही, बच्चे गिर सकते हैं।",
        "expected_category": "drainage",
        "expected_priority": "critical",
        "note": "Priority is Critical because an open manhole coupled with zero nighttime visibility poses immediate life-threatening fall risk.",
    },

    # 41. English - Medium (Damaged and uneven sidewalk pavers causing trip hazard outside S...)
    {
        "text": "Footpath paving tiles completely broken and uneven in front of SBI ATM on 80 Feet Road, senior citizens tripping.",
        "expected_category": "roads",
        "expected_priority": "medium",
        "note": "Priority is Medium because broken pedestrian pavers create a persistent tripping risk but not an emergency road blockage.",
    },

    # 42. English - Critical (Dangerous unbarricaded road cave-in occupying half the carriagew...)
    {
        "text": "Heavy road cave-in spanning half the lane near Metro Pillar 142, barricades missing.",
        "expected_category": "roads",
        "expected_priority": "critical",
        "note": "Priority is Critical due to deep unbarricaded structural cave-in on an active lane posing fatal collision risk.",
    },

    # 43. English - Low (Faded zebra crossings and lane markings needing repaint at high ...)
    {
        "text": "White lane markings and pedestrian zebra crossing faded completely at the busy high school intersection.",
        "expected_category": "roads",
        "expected_priority": "low",
        "note": "Priority is Low as faded paint is scheduled maintenance without immediate structural road disruption.",
    },

    # 44. English - Medium (Unauthorized unmarked speed bump outside Gate 3 causing vehicle ...)
    {
        "text": "Unscientific high speed breaker built by local residents without warning paint or signage outside gate 3.",
        "expected_category": "roads",
        "expected_priority": "medium",
        "note": "Priority is Medium because unpainted speed bumps damage vehicles and cause minor falls but no lane shutdown.",
    },

    # 45. Bengali - Medium (Eroded road bitumen with loose stones making vehicular driving d...)
    {
        "text": "রাস্তার পিচ উঠে গিয়ে পাথর বেরিয়ে পড়েছে, গাড়ি চালাতে খুব অসুবিধা হচ্ছে।",
        "expected_category": "roads",
        "expected_priority": "medium",
        "note": "Priority is Medium as eroded top surface impairs commute quality without forming sudden life-threatening craters.",
    },

    # 46. English - High (Unfilled utility trench left open across main lane for two weeks...)
    {
        "text": "Open trench dug up for utility cables left unfilled for two weeks across the main lane.",
        "expected_category": "roads",
        "expected_priority": "high",
        "note": "Priority is High because a wide transverse road trench forces sudden braking and damages fast traffic.",
    },

    # 47. English - Critical (Sharp protruding steel expansion joint on flyover slashing vehic...)
    {
        "text": "Iron expansion joint on the flyover bridge has jutted upwards by 3 inches, cutting vehicle tires.",
        "expected_category": "roads",
        "expected_priority": "critical",
        "note": "Priority is Critical because high-speed tire punctures on a flyover cause catastrophic rollover accidents.",
    },

    # 48. English - High (Slippery gravel on curved slope near temple arch causing repeate...)
    {
        "text": "Loose gravel scattered on curved road slope near temple arch, two-wheelers slipping during turns.",
        "expected_category": "roads",
        "expected_priority": "high",
        "note": "Priority is High due to high probability of active skidding accidents on a turning slope.",
    },

    # 49. Kannada - High (Severe sewage cross-contamination producing black foul-smelling ...)
    {
        "text": "ಕುಡಿಯುವ ನೀರಿನಲ್ಲಿ ಚರಂಡಿ ನೀರು ಮಿಶ್ರಣವಾಗಿ ಬರುತ್ತಿದೆ, ಕೆಟ್ಟ ವಾಸನೆ ಮತ್ತು ಕಪ್ಪು ಬಣ್ಣದ ನೀರು.",
        "expected_category": "water_supply",
        "expected_priority": "high",
        "also_accept_category": ["drainage"],
        "note": "Priority is High because contaminated municipal drinking supply creates an acute community disease risk.",
    },

    # 50. English - Low (Minor morning water loss from roadside sluice valve on 12th Cros...)
    {
        "text": "Water valve leak on 12th cross causing small continuous pool on curb side during morning supply hours.",
        "expected_category": "water_supply",
        "expected_priority": "low",
        "note": "Priority is Low because leakage is restricted to supply hours and creates negligible water accumulation.",
    },

    # 51. English - Critical (Major municipal water trunk line rupture causing high-pressure f...)
    {
        "text": "Main 600mm water distribution pipeline burst near bus terminus, 15-foot water jet flooding roadway.",
        "expected_category": "water_supply",
        "expected_priority": "critical",
        "note": "Priority is Critical due to catastrophic water wastage, rapid road flooding, and infrastructure erosion.",
    },

    # 52. English - High (Municipal sewer blockage causing indoor sewage reverse overflow ...)
    {
        "text": "Underground sewer line choked, black sewage backing up into ground floor toilet bowls in Apartment 4B.",
        "expected_category": "drainage",
        "expected_priority": "high",
        "note": "Priority is High because raw sewage backflow inside living quarters poses severe biological contamination.",
    },

    # 53. Malayalam - Medium (Chronic low tap water pressure preventing flow to upper floors f...)
    {
        "text": "കുടിവെള്ള പൈപ്പ് ലൈനിൽ പ്രഷർ വളരെ കുറവാണ്, രണ്ടാഴ്ചയായി ഒന്നാം നിലയിൽ വെള്ളം എത്തുന്നില്ല.",
        "expected_category": "water_supply",
        "expected_priority": "medium",
        "note": "Priority is Medium as non-critical water scarcity impairs domestic living but is not an emergency burst.",
    },

    # 54. English - High (Depressed sewer manhole frame creating sudden 6-inch road drop o...)
    {
        "text": "Manhole chamber frame sunken 6 inches below asphalt layer on fast lane opposite City Hospital.",
        "expected_category": "drainage",
        "expected_priority": "high",
        "note": "Priority is High due to sudden chassis impact and two-wheeler instability in front of hospital route.",
    },

    # 55. English - Critical (Corrosive chemical industrial effluent discharge with toxic vapo...)
    {
        "text": "Heavy industrial effluence leaking from underground drain chamber with strong acidic fumes.",
        "expected_category": "drainage",
        "expected_priority": "critical",
        "note": "Priority is Critical because toxic vapor and acidic liquid present immediate chemical hazard to citizens.",
    },

    # 56. English - Low (Mechanical handpump handle broken at public community borewell.)
    {
        "text": "Public borewell tap handle is broken near community hall, water flowing when pumped manually.",
        "expected_category": "water_supply",
        "expected_priority": "low",
        "note": "Priority is Low as it involves routine mechanical repair of a standalone neighborhood pump.",
    },

    # 57. Telugu - High (Unfenced high-voltage distribution transformer located immediate...)
    {
        "text": "విధుల్లోని ట్రాన్స్‌ఫార్మర్ చుట్టూ రక్షణ కంచె లేదు, పిల్లలు ఆడుకునే మైదానం పక్కనే ఉంది.",
        "expected_category": "electricity",
        "expected_priority": "high",
        "note": "Priority is High because open high-voltage equipment adjacent to play areas is an extreme safety liability.",
    },

    # 58. English - Critical (Open street feeder pillar box exposing live high-voltage busbars...)
    {
        "text": "Electricity junction box door hanging open with exposed live busbars reachable by pedestrians.",
        "expected_category": "electricity",
        "expected_priority": "critical",
        "note": "Priority is Critical due to immediate accidental electrocution risk to walking pedestrians.",
    },

    # 59. English - Medium (Severe voltage instability across Block D risking domestic elect...)
    {
        "text": "Frequent voltage fluctuations between 160V and 290V causing home appliances to shut down in Block D.",
        "expected_category": "electricity",
        "expected_priority": "medium",
        "note": "Priority is Medium as severe voltage instability causes economic damage but no immediate fire.",
    },

    # 60. English - Critical (Structurally broken concrete power pole tilting severely over pu...)
    {
        "text": "Cement electrical pole cracked at the base and tilting at 30 degrees across telephone cables.",
        "expected_category": "electricity",
        "expected_priority": "critical",
        "note": "Priority is Critical because imminent structural collapse of a power pole will drop live overhead conductors.",
    },

    # 61. English - Low (Minor surface rust on streetlight pole base requiring routine an...)
    {
        "text": "Streetlight pole base has slight rusted paint coating near Sector 7 park gate.",
        "expected_category": "streetlights",
        "expected_priority": "low",
        "note": "Priority is Low because superficial corrosion does not affect structural integrity.",
    },

    # 62. Tamil - Critical (Electrified lamppost leaking current during rains due to earthin...)
    {
        "text": "மழை பெய்யும்போது மின்கம்பத்தில் கை வைத்தால் கரண்ட் அடிக்கிறது, எர்த் பிரச்சனை உள்ளது.",
        "expected_category": "electricity",
        "expected_priority": "critical",
        "note": "Priority is Critical because an energized metallic pole in public space delivers lethal shocks.",
    },

    # 63. English - Medium (Power distribution lines tangled in overgrown tree branches need...)
    {
        "text": "Overhead distribution cables entangled in thick overgrown tree canopy behind market complex.",
        "expected_category": "electricity",
        "expected_priority": "medium",
        "note": "Priority is Medium as line entanglement creates intermittent tripping risk during strong winds.",
    },

    # 64. English - Low (Cracked meter display glass on pole 18 with no electrical safety...)
    {
        "text": "Billing meter on utility pole 18 has broken display glass, reading cannot be noted.",
        "expected_category": "electricity",
        "expected_priority": "low",
        "note": "Priority is Low since broken outer glass is an administrative metering defect.",
    },

    # 65. English - Critical (Illegal dumping of hazardous biomedical waste and sharps into pu...)
    {
        "text": "Hospital disposing used syringes, blood bags and bio-waste directly into public municipal dumpster.",
        "expected_category": "garbage",
        "expected_priority": "critical",
        "note": "Priority is Critical due to severe infectious biohazard exposure to scavengers and general public.",
    },

    # 66. English - High (Decomposing animal carcass on roadside near water tank requiring...)
    {
        "text": "Dead street dog carcass lying on side of road near water tank since yesterday afternoon.",
        "expected_category": "garbage",
        "expected_priority": "high",
        "note": "Priority is High because decomposing carcasses cause extreme biological stench and health hazards.",
    },

    # 67. English - Low (Missing lid on park twin-dustbin unit.)
    {
        "text": "Wet waste bin lid missing from public twin-bin stand near jogging park.",
        "expected_category": "garbage",
        "expected_priority": "low",
        "note": "Priority is Low as missing bin lid does not obstruct waste deposit or pose health emergencies.",
    },

    # 68. Malayalam - High (Illegal burning of plastic waste at compost facility generating ...)
    {
        "text": "കമ്പോസ്റ്റ് പ്ലാന്റിൽ നിന്ന് അസഹനീയമായ പുക വരുന്നു, പ്ലാസ്റ്റിക് മാലിന്യം കത്തിക്കുന്നു.",
        "expected_category": "garbage",
        "expected_priority": "high",
        "note": "Priority is High because open plastic incineration produces toxic carcinogens affecting local residents.",
    },

    # 69. English - Low (Uncollected dry garden and horticulture waste left outside bunga...)
    {
        "text": "Green garden clippings and trimmed hedge branches piled up outside bungalow 5 for 3 days.",
        "expected_category": "garbage",
        "expected_priority": "low",
        "note": "Priority is Low as dry garden clippings represent non-hazardous organic bulk waste.",
    },

    # 70. English - Medium (Sweepers improperly disposing road silt directly into stormwater...)
    {
        "text": "Sanitation sweepers dumping collected street dust directly into roadside storm drain openings.",
        "expected_category": "garbage",
        "expected_priority": "medium",
        "note": "Priority is Medium because silting drains leads to eventual monsoon waterlogging.",
    },

    # 71. English - High (Fish market vendors dumping raw decaying fish waste onto public ...)
    {
        "text": "Commercial fish market vendors throwing offal and rotten fish guts on public footpath every evening.",
        "expected_category": "garbage",
        "expected_priority": "high",
        "note": "Priority is High due to putrid organic decomposition, pest infestation, and severe health nuisance.",
    },

    # 72. English - Low (Preventive clearance requested for post office public waste bin ...)
    {
        "text": "Dustbin outside post office is 80% full, will overflow by tonight if not cleared.",
        "expected_category": "garbage",
        "expected_priority": "low",
        "note": "Priority is Low because the bin has not yet spilled onto the street.",
    },

    # 73. English - High (Broken drain concrete slab creating large fall opening near nurs...)
    {
        "text": "Cement drain slab broken, leaving a 3-foot wide exposed slit on pedestrian walkway near nursery school.",
        "expected_category": "drainage",
        "expected_priority": "high",
        "note": "Priority is High due to severe fall/injury hazard near a school area without active flood conditions.",
    },

    # 74. English - Medium (Clogged storm drain causing localized rainwater stagnation on Cr...)
    {
        "text": "Storm drain choked with plastic bottles causing knee-deep stagnant rainwater on Cross 7 after light drizzle.",
        "expected_category": "drainage",
        "expected_priority": "medium",
        "note": "Priority is Medium because waterlogging causes local transit slowdown without entering residential premises.",
    },

    # 75. English - Critical (Drainage canal wall collapse causing flash flooding into residen...)
    {
        "text": "Stormwater canal retaining wall collapsed during heavy rain; floodwater actively washing into basement parking.",
        "expected_category": "drainage",
        "expected_priority": "critical",
        "note": "Priority is Critical because active flood breach threatens structural foundation and human life.",
    },

    # 76. English - Low (Routine pre-monsoon desilting requested for silted roadside drai...)
    {
        "text": "Heavy silt accumulated in roadside ditch over summer; requires desilting before monsoon starts.",
        "expected_category": "drainage",
        "expected_priority": "low",
        "note": "Priority is Low because it represents preventive maintenance before onset of rains.",
    },

    # 77. Tamil - High (Natural stormwater discharge channel blocked by illegal construc...)
    {
        "text": "காற்றாற்று வெள்ள நீர் வடிகால் ஆக்கிரமிக்கப்பட்டு தடுப்பு சுவர் கட்டப்பட்டுள்ளது.",
        "expected_category": "drainage",
        "expected_priority": "high",
        "note": "Priority is High because obstruction of primary stormwater channels creates massive flood vulnerabilities.",
    },

    # 78. English - High (Stolen iron stormwater grating leaving dangerous road hole at Ga...)
    {
        "text": "Drain iron grill grate stolen overnight at corner of Gandhi Chowk, leaving an open rectangular hole.",
        "expected_category": "drainage",
        "expected_priority": "high",
        "note": "Priority is High due to severe vehicle tire trap and nighttime pedestrian fall risk.",
    },

    # 79. English - Medium (Stagnant stormwater drain acting as heavy mosquito breeding site...)
    {
        "text": "Stagnant green water in open roadside ditch breeding thousands of mosquitoes near primary clinic.",
        "expected_category": "drainage",
        "expected_priority": "medium",
        "note": "Priority is Medium due to vector-borne disease proliferation in public health proximity.",
    },

    # 80. English - Low (Minor wild weeds growing at stormwater outlet mouth into canal.)
    {
        "text": "Drain outlet pipe into main canal has minor vegetation overgrowth at mouth.",
        "expected_category": "drainage",
        "expected_priority": "low",
        "note": "Priority is Low as weed growth has not yet caused significant hydraulic backwater.",
    },

    # 81. English - High (Aggressive pack of stray dogs attacking commuters and school chi...)
    {
        "text": "Pack of aggressive stray dogs chasing two-wheelers and biting school children near gate 4 every morning.",
        "expected_category": "sanitation",
        "expected_priority": "high",
        "note": "Priority is High due to active physical bite injuries and severe rabies hazard to children.",
    },

    # 82. English - Low (Routine municipal anti-larval mosquito fogging requested for War...)
    {
        "text": "Chemical fogging for dengue mosquitoes not conducted in Ward 18 for over two months.",
        "expected_category": "sanitation",
        "expected_priority": "low",
        "note": "Priority is Low because routine periodic fogging request is an administrative sanitation schedule.",
    },

    # 83. English - Medium (Unmaintained public toilet overflowing onto sidewalk opposite ma...)
    {
        "text": "Public urinal opposite main bus stand overflowing with urine onto pavement, severe unbearable stink.",
        "expected_category": "sanitation",
        "expected_priority": "medium",
        "note": "Priority is Medium as severe public insanitary conditions degrade public space without biohazard lockdown.",
    },

    # 84. Marathi - Medium (Stray cattle squatting on main road causing severe traffic snarl...)
    {
        "text": "बेवारस जनावरे मुख्य रस्त्यावर बसल्यामुळे वाहतूक कोंडी होत आहे आणि अपघात घडत आहेत.",
        "expected_category": "roads",
        "expected_priority": "medium",
        "note": "Priority is Medium as cattle on roads cause transit delays and vehicular dodging without fatal pileups.",
    },

    # 85. English - High (Unlicensed roadside poultry slaughtering releasing biological bl...)
    {
        "text": "Illegal open meat stall slaughtering chickens on open footpath without health clearance or drainage.",
        "expected_category": "sanitation",
        "expected_priority": "high",
        "note": "Priority is High because unhygienic unregulated animal slaughtering generates acute disease outbreak risks.",
    },

    # 86. English - High (Large active bee hive on children swing frame creating immediate...)
    {
        "text": "Swarm of honeybees formed large hive on children playground swing set frame.",
        "expected_category": "parks",
        "expected_priority": "high",
        "note": "Priority is High due to severe sting attack threat directly on children's play equipment.",
    },

    # 87. English - Low (Park seating benches soiled with pigeon droppings requiring pres...)
    {
        "text": "Public park benches covered in wild pigeon droppings and bird feathers.",
        "expected_category": "parks",
        "expected_priority": "low",
        "note": "Priority is Low as soiled public park seating is a routine cleaning maintenance matter.",
    },

    # 88. English - Low (Small rodent carcass on footbridge staircase needing sweeper cle...)
    {
        "text": "Dead rat lying dried up on pedestrian overbridge stairs.",
        "expected_category": "garbage",
        "expected_priority": "low",
        "note": "Priority is Low as minor small-animal carcass on stairs requires routine sweep pickup.",
    },

    # 89. English - Medium (Shopkeeper built illegal concrete ramp extending 6 feet onto pub...)
    {
        "text": "Local sweet shop has extended permanent concrete ramp 6 feet onto the public road, blocking car passage.",
        "expected_category": "roads",
        "expected_priority": "medium",
        "note": "Priority is Medium because illegal permanent road construction restricts lane width and requires demolition.",
    },

    # 90. English - Critical (Uprooted giant tree crushing vehicle and completely blocking eme...)
    {
        "text": "Massive ancient peepal tree uprooted during cyclone, completely crushing car and blocking hospital road.",
        "expected_category": "roads",
        "expected_priority": "critical",
        "note": "Priority is Critical because emergency access to hospital is blocked and active property damage occurred.",
    },

    # 91. English - Low (Unauthorized promotional flex banners obstructing pedestrian foo...)
    {
        "text": "Illegal advertising flex banners erected across pedestrian footpath blocking sightlines at intersection.",
        "expected_category": "roads",
        "expected_priority": "low",
        "note": "Priority is Low as unauthorized hoarding removal is a standard municipal enforcement action.",
    },

    # 92. English - Low (Temporary hawker pushcart parked near park entrance during eveni...)
    {
        "text": "Street vendor pushing cart occupying corner spot near park gate during evening hours.",
        "expected_category": "roads",
        "expected_priority": "low",
        "note": "Priority is Low because transient hawkers create minor pedestrian friction without physical structures.",
    },

    # 93. English - Critical (Unsafe commercial basement excavation causing adjoining resident...)
    {
        "text": "Deep foundation excavation for commercial complex next door causing boundary wall of residential society to crack and tilt.",
        "expected_category": "other",
        "expected_priority": "critical",
        "note": "Priority is Critical due to structural destabilization threatening immediate building collapse.",
    },

    # 94. English - Low (Private garden shrubs protruding slightly over boundary wall int...)
    {
        "text": "Bougainvillea plant branches from private bungalow garden hanging 1 foot over compound wall into alley.",
        "expected_category": "other",
        "expected_priority": "low",
        "note": "Priority is Low as mild ornamental overgrowth causes minimal inconvenience.",
    },

    # 95. Gujarati - High (Broken underground sewer lid in street trapping vehicles and cau...)
    {
        "text": "શેરીમાં આવેલું ભૂગર્ભ ગટરનું ઢાંકણું તૂટી ગયું છે અને વાહનો તેમાં ફસાઈ રહ્યા છે.",
        "expected_category": "drainage",
        "expected_priority": "high",
        "note": "Priority is High because a broken sewer lid on an active street directly traps vehicles and injures drivers.",
    },

    # 96. Punjabi - Medium (All streetlights in lane non-functional for a week causing secur...)
    {
        "text": "ਗਲੀ ਦੀਆਂ ਸਾਰੀਆਂ ਲਾਈਟਾਂ ਇੱਕ ਹਫ਼ਤੇ ਤੋਂ ਬੰਦ ਪਈਆਂ ਹਨ, ਰਾਤ ਨੂੰ ਚੋਰੀ ਦਾ ਡਰ ਬਣਿਆ ਰਹਿੰਦਾ ਹੈ।",
        "expected_category": "streetlights",
        "expected_priority": "medium",
        "note": "Priority is Medium because complete dark street corridor increases neighborhood vulnerability over an extended period.",
    },

    # 97. English - Medium (Unpaved slushy trench left after water pipeline works trapping p...)
    {
        "text": "Water pipeline work completed 1 month ago but road was never restored; now muddy slush traps cars whenever water tanker passes.",
        "expected_category": "roads",
        "expected_priority": "medium",
        "also_accept_category": ["water_supply"],
        "borderline": True,
        "note": "Priority is Medium and confidence is 0.49 due to overlapping responsibility between Water Board excavation and PWD resurfacing.",
    },

    # 98. English - Low (Unsolicited commercial loan marketing spam text.)
    {
        "text": "Urgent loan approved up to 10 Lakhs with zero collateral! Call instant finance desk at 9123456780.",
        "expected_category": "other",
        "expected_priority": "low",
        "note": "Priority is Low because non-civic commercial promotional spam is assigned the minimum priority tier.",
    },

    # 99. English - Medium (Unexplained recurring subterranean rumbling noise near circular ...)
    {
        "text": "Strange loud rumbling noise heard underground every 10 minutes near the circular market square.",
        "expected_category": "other",
        "expected_priority": "medium",
        "borderline": True,
        "note": "Priority is Medium and confidence is 0.34 due to ambiguous physical source (metro boring vs pipeline cavitation vs geological).",
    },

    # 100. English - Low (Grievance regarding impolite municipal administrative staff at b...)
    {
        "text": "Municipal staff behavior was very rude when I visited office for birth certificate counter.",
        "expected_category": "other",
        "expected_priority": "low",
        "note": "Priority is Low because administrative staff demeanor is an internal governance issue with no emergency safety impact.",
    },
]
