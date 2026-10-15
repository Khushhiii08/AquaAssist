"""
AquaAssist: Response Catalog & Policy Advisory Renderer
Provides authoritative bilingual (English & Telugu) advisories across:
- OUT_OF_DOMAIN (True non-aquaculture queries)
- CHEMICAL_HAZARD (Lethal or violent chemical reactions)
- BLIND_DOSING_REFUSAL (Refusal to prescribe uncalibrated toxic chemicals without volume/diagnosis)
- REALTIME_DATA_LIMITATION (Offline boundaries: live village market prices, weather)
- UNINDEXED_CONDITION (Missing disease / treatment in verified manuals)
- EVIDENCE_MISMATCH (Retrieved evidence does not address query intent)
- SENSOR_CONFLICT (Contradictory measurements across reports)
- VAGUE_DIAGNOSTIC (Insufficient symptoms to formulate advice)
"""

from typing import Dict, Any

DECISION_RESPONSES: Dict[str, Dict[str, str]] = {
    "OUT_OF_DOMAIN": {
        "title": "Out of Domain / పరిధికి వెలుపల",
        "english": (
            "AquaAssist is an offline diagnostic assistant specialized exclusively in aquaculture, "
            "finfish/shrimp health management, and pond water limnology. "
            "Please submit a query related to pond management or aquatic health."
        ),
        "telugu": (
            "AquaAssist కేవలం ఆక్వాకల్చర్, చేపలు/రొయ్యల ఆరోగ్య నిర్వహణ మరియు చెరువు నీటి నాణ్యత సమస్యలకు "
            "మాత్రమే సహాయం చేయగలదు. దయచేసి చెరువు నిర్వహణ లేదా ఆక్వాకల్చర్ సంబంధిత ప్రశ్నలను అడగండి."
        )
    },

    "CHEMICAL_HAZARD": {
        "title": "Severe Chemical Hazard Warning / తీవ్రమైన రసాయన ప్రమాద హెచ్చరిక",
        "english": (
            "⚠️ **CRITICAL CHEMICAL HAZARD:**\n"
            "• **Do Not Mix Chemicals:** Never combine bleaching powder (calcium hypochlorite) with potassium "
            "permanganate (KMnO₄) or formalin. Mixing strong oxidizing and reducing agents produces violent exothermic "
            "reactions and releases toxic chlorine gas, causing immediate mass mortality and water deoxygenation.\n"
            "• **Immediate Action:** Run all pond aerators at full capacity. Do not apply uncalibrated chemicals to active ponds.\n"
            "• **Next Step:** Consult an accredited aquaculture officer or qualified pathologist for prescribed single-chemical schedules."
        ),
        "telugu": (
            "⚠️ **తీవ్రమైన రసాయన ప్రమాద హెచ్చరిక:**\n"
            "• **రసాయనాలను ఎప్పుడూ కలపవద్దు:** బ్లీచింగ్ పౌడర్ మరియు పొటాషియం పర్మాంగనేట్ (KMnO₄) వంటి తీవ్రమైన ఆక్సిడైజింగ్ రసాయనాలను "
            "కలపడం అత్యంత ప్రమాదకరం. ఇది విషపూరిత క్లోరిన్ వాయువును విడుదల చేసి, చెరువులోని ఆక్సిజన్‌ను హరించి చేపలు/రొయ్యల సామూహిక మరణానికి దారితీస్తుంది.\n"
            "• **తక్షణ చర్య:** వెంటనే అన్ని ఏరేటర్లను ఆన్ చేయండి. ఎటువంటి రసాయన మిశ్రమాలను చెరువులో వేయకండి.\n"
            "• **తదుపరి చర్య:** అర్హత కలిగిన మత్స్యశాఖ అధికారి లేదా నిపుణుడిని సంప్రదించండి."
        )
    },

    "BLIND_DOSING_REFUSAL": {
        "title": "Chemical Dosing Safeguard / రసాయన మోతాదు భద్రతా నిబంధన",
        "english": (
            "⚠️ **DOSAGE WITHHELD FOR SAFETY:**\n"
            "• **Risk of Toxicity:** Adding uncalibrated bleaching powder or disinfectant to ponds with dying stock "
            "can induce severe chemical gill burn, acute pH shock, and total collapse of the phytoplankton bloom.\n"
            "• **Required Diagnostic Data:** Safe therapeutic or disinfectant dosing requires exact water depth, volume, "
            "biomass, alkalinity, and identification of the causative pathogen (bacterial vs. parasitic vs. environmental hypoxia).\n"
            "• **Emergency First Aid:** Immediately measure Dissolved Oxygen (DO). If stock is gasping, turn on all aerators "
            "and suspend all feeding. Submit water and tissue samples to an accredited aquatic diagnostic lab."
        ),
        "telugu": (
            "⚠️ **భద్రతా కారణాల దృష్ట్యా మోతాదు నిలిపివేయబడింది:**\n"
            "• **రసాయన విషప్రభావ ప్రమాదం:** చనిపోతున్న చేపలు ఉన్న చెరువులో సరైన లెక్కలు లేకుండా బ్లీచింగ్ పౌడర్ వేయడం వల్ల "
            "చేపల మొప్పలు కాలిపోయి, pH మారిపోయి పరిస్థితి మరింత విషమిస్తుంది.\n"
            "• **అవసరమైన వివరాలు:** సురక్షితమైన మోతాదును నిర్ణయించడానికి చెరువు నీటి లోతు, విస్తీర్ణం, నీటి క్షారత (Alkalinity) మరియు వ్యాధి కారకం తెలియాలి.\n"
            "• **తక్షణ ప్రథమ చికిత్స:** వెంటనే ఆక్సిజన్ (DO) స్థాయిని పరీక్షించండి. ఏరేటర్లను పూర్తిస్థాయిలో ఆన్ చేయండి మరియు మేత ఇవ్వడం ఆపండి. "
            "సమీప ఆక్వా ల్యాబ్‌ను సంప్రదించండి."
        )
    },

    "REALTIME_DATA_LIMITATION": {
        "title": "Live Market Data Limitation / లైవ్ మార్కెట్ డేటా పరిమితి",
        "english": (
            "ℹ️ **OFFLINE TECHNICAL DATABASE BOUNDARY:**\n"
            "AquaAssist operates from verified scientific aquaculture manuals, limnological standards, and pathological guidelines. "
            "It does not maintain live internet connections to daily village-level commercial feed prices or commodity mandi tickers.\n"
            "• For today's verified feed prices, please consult your local feed distributor, cooperative society, or feed company dealer."
        ),
        "telugu": (
            "ℹ️ **ఆఫ్‌లైన్ సాంకేతిక డేటాబేస్ పరిమితి:**\n"
            "AquaAssist శాస్త్రీయ పెంపక మార్గదర్శకాలు మరియు నీటి నాణ్యత గ్రంథాల ఆధారంగా పనిచేస్తుంది. "
            "ఇది ప్రతిరోజూ మారే స్థానిక గ్రామీణ మేత ధరలు లేదా మార్కెట్ ధరల ప్రత్యక్ష సమాచారాన్ని కలిగి ఉండదు.\n"
            "• తాజా ఫీడ్ ధరల కోసం దయచేసి మీ స్థానిక డీలర్ లేదా ఫీడ్ సొసైటీని సంప్రదించండి."
        )
    },

    "UNINDEXED_CONDITION": {
        "title": "Treatment Unavailable in Scientific Manuals / సమాచారం అందుబాటులో లేదు",
        "english": (
            "AquaAssist could not locate an authoritative, scientifically validated treatment protocol for this specific "
            "condition in its current library.\n"
            "• **Safety Assurance:** The system strictly refuses to synthesize fabricated chemical treatments.\n"
            "• **Recommendation:** Provide the specific clinical lesions (e.g., red spots on belly, gill rot, white feces, shell erosion) "
            "or consult a certified aquatic veterinary clinician."
        ),
        "telugu": (
            "మా ప్రస్తుత శాస్త్రీయ నాలెడ్జ్ బేస్‌లో ఈ నిర్దిష్ట సమస్య లేదా వ్యాధికి సంబంధించి ధృవీకరించబడిన చికిత్సా వివరాలు లభించలేదు.\n"
            "• **భద్రతా నిబంధన:** తప్పుడు లేదా ధృవీకరించని రసాయన చికిత్సలను సిస్టమ్ సూచించదు.\n"
            "• **సలహా:** దయచేసి స్పష్టమైన బాహ్య లక్షణాలను (ఉదా: మొప్పల కుళ్ళు, ఎరుపు మచ్చలు, తెల్లటి మల విసర్జన) తెలపండి లేదా నిపుణులను సంప్రదించండి."
        )
    },

    "EVIDENCE_MISMATCH": {
        "title": "Insufficient Evidence / సరిపడని సమాచారం",
        "english": (
            "The retrieved scientific documentation covers related pond parameters (such as soil alkalinity or general limnology) "
            "but does not contain direct, conclusive evidence answering your specific inquiry.\n"
            "To prevent misleading advice, operational recommendations have been withheld."
        ),
        "telugu": (
            "సేకరించిన పత్రాలు చెరువుకు సంబంధించిన ఇతర అంశాలను (నేల క్షారత వంటివి) చర్చిస్తున్నాయి, కానీ మీ ప్రశ్నకు అవసరమైన ఖచ్చితమైన పరిష్కారాన్ని అందించడం లేదు.\n"
            "తప్పుడు సలహాలను నివారించడానికి సమాధానం నిలిపివేయబడింది."
        )
    },

    "CONFLICT_READINGS": {
        "title": "Contradictory Telemetry Readings / పరస్పర విరుద్ధమైన కొలతలు",
        "english": (
            "⚠️ **DISCREPANCY DETECTED:** You have reported conflicting readings for the same water parameter.\n"
            "• **Precautionary Protocol:** If either reading indicates dangerous hypoxia (DO < 3.0 mg/L), turn on aerators immediately as a precaution while verifying.\n"
            "• **Sensor Verification:** Calibrate digital probes with fresh buffer solution, inspect membrane caps for algal fouling, or perform a manual chemical titration test to verify the true reading."
        ),
        "telugu": (
            "⚠️ **కొలతలలో వ్యత్యాసం ఉంది:** మీరు ఒకే పారామీటర్‌కు సంబంధించి వేర్వేరు రీడింగులను నమోదు చేశారు.\n"
            "• **ముందస్తు జాగ్రత్త:** ఒకవేళ రీడింగులలో ఒకటి ఆక్సిజన్ తక్కువగా (< 3.0 mg/L) చూపిస్తుంటే, వెంటనే ఏరేటర్లను ఆన్ చేయండి.\n"
            "• **సెన్సార్ తనిఖీ:** మీ డిజిటల్ మీటర్‌ను రీ-క్యాలిబ్రేట్ చేయండి లేదా కెమికల్ టెస్ట్ కిట్ ద్వారా ఖచ్చితమైన స్థాయిని నిర్ధారించుకోండి."
        )
    },

    "VAGUE_DIAGNOSTIC": {
        "title": "Clarification Required / మరిన్ని వివరాలు అవసరం",
        "english": (
            "To accurately assess the condition of your pond, please provide specific observations:\n"
            "1. **Water Appearance:** What is the color and clarity (e.g., dark green bloom, brown, clear, turbid)?\n"
            "2. **Stock Behavior:** Are animals gasping at the surface early morning, sluggish, or off-feed?\n"
            "3. **Water Test Readings:** What are your current Dissolved Oxygen (DO), pH, and Ammonia levels?"
        ),
        "telugu": (
            "మీ చెరువు పరిస్థితిని సరిగ్గా అంచనా వేయడానికి మరికొన్ని వివరాలు తెలపండి:\n"
            "1. **నీటి రంగు:** చెరువు నీరు ఎలా ఉంది (ముదురు ఆకుపచ్చ, గోధుమ రంగు, లేదా బురద నీరా)?\n"
            "2. **జీవుల ప్రవర్తన:** చేపలు/రొయ్యలు ఉపరితలంపై తేలుతున్నాయా, ఈత మందగించిందా, లేదా మేత తినడం లేదా?\n"
            "3. **కొలతలు:** ప్రస్తుత ఆక్సిజన్ (DO), pH లేదా అమ్మోనియా రీడింగులు ఎంత ఉన్నాయి?"
        )
    }
}


def render_policy_response(decision_key: str) -> Dict[str, str]:
    """Formats and returns the standardized bilingual response payload."""
    payload = DECISION_RESPONSES.get(decision_key, DECISION_RESPONSES["OUT_OF_DOMAIN"])
    formatted_msg = (
        f"**Status: {payload['title']}**\n\n"
        f"--- English ---\n{payload['english']}\n\n"
        f"--- తెలుగు (Telugu) ---\n{payload['telugu']}"
    )
    return {
        "status": decision_key,
        "message": formatted_msg,
        "english": payload["english"],
        "telugu": payload["telugu"]
    }