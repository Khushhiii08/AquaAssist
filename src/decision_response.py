DECISION_RESPONSES = {
    "CLARIFY": {
        "message": (
            "📌 **Status: More Information Needed / మరింత సమాచారం అవసరం**\n\n"
            "--- English ---\n"
            "We noticed your observation, but we need a bit more detail. Could you please check your water test kit and provide your current Dissolved Oxygen (DO) and pH levels?\n\n"
            "--- తెలుగు (Telugu) ---\n"
            "సలహా: మాకు మరికొన్ని వివరాలు కావాలి. దయచేసి మీ వాటర్ టెస్ట్ కిట్ ద్వారా ప్రస్తుత ఆక్సిజన్ (DO) మరియు pH స్థాయిలను తనిఖీ చేసి తెలపండి."
        )
    },
    "ABSTAIN": {
        "message": (
            "🛑 **Status: Out of Domain / పరిధికి వెలుపల**\n\n"
            "--- English ---\n"
            "I can only assist with aquaculture, shrimp/fish health, and pond water quality issues. Please ask an aquaculture-related question.\n\n"
            "--- తెలుగు (Telugu) ---\n"
            "నేను ఆక్వాకల్చర్, రొయ్యలు/చేపల ఆరోగ్యం మరియు నీటి నాణ్యత సమస్యలకు మాత్రమే సహాయం చేయగలను. దయచేసి ఆక్వాకల్చర్ సంబంధిత ప్రశ్న అడగండి."
        )
    },
    "CONFLICT": {
        "message": (
            "⚠️ **Status: Safety Warning / హెచ్చరిక మరియు వైరుధ్యం**\n\n"
            "--- English ---\n"
            "Warning: Conflicting or high-risk measurements detected. Please re-test your water parameters or consult an expert before taking action.\n\n"
            "--- తెలుగు (Telugu) ---\n"
            "హెచ్చరిక: మీ ఇచ్చిన కొలతలలో వ్యత్యాసాలు ఉన్నాయి. దయచేసి నీటి నాణ్యతను మళ్లీ పరీక్షించండి లేదా నిపుణులను సంప్రదించండి."
        )
    }
}

def get_decision_response(decision_key):
    return DECISION_RESPONSES.get(decision_key, DECISION_RESPONSES["CLARIFY"])