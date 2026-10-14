def render_farmer_ux(result_dict):
    """
    Conversational Persona UX Layer (Bilingual: English & Telugu).
    Translates raw routing states into dual-language, plain-language advice.
    """
    decision = result_dict.get("decision")
    hypothesis = result_dict.get("hypothesis")
    
    print("\n" + "="*65)
    print("🌾 AQUAASSIST FIELD ADVISORY / అక్వా అసిస్ట్ క్షేత్ర సలహా")
    print("="*65)
    
    if decision == "ANSWER":
        best_chunk = result_dict["evidence"][0]["text"] if result_dict.get("evidence") else "Follow standard ICAR-CIBA protocols."
        
        print("📌 STATUS: RECOMMENDED ACTION READY / సిఫార్సు చేయబడిన చర్య సిద్ధంగా ఉంది\n")
        
        print("--- English ---")
        print(f"Observation: {hypothesis}")
        print(f"Advice: {best_chunk}\n")
        
        print("--- తెలుగు (Telugu) ---")
        print(f"పరిశీలన: {hypothesis}")
        print(f"సలహా: {best_chunk}\n")
        print("దయచేసి వెంటనే తగిన చర్యలు తీసుకోండి.")
        
    elif decision == "CLARIFY":
        print("📌 STATUS: MORE INFORMATION NEEDED / మరింత సమాచారం అవసరం\n")
        
        print("--- English ---")
        print(f"Observation: {hypothesis}")
        print("Advice: We need a bit more detail. Could you please check your water test kit and provide your current Dissolved Oxygen (DO) and pH levels?\n")
        
        print("--- తెలుగు (Telugu) ---")
        print(f"పరిశీలన: {hypothesis}")
        print("సలహా: మాకు మరికొన్ని వివరాలు కావాలి. దయచేసి మీ వాటర్ టెస్ట్ కిట్ ద్వారా ప్రస్తుత ఆక్సిజన్ (DO) మరియు pH స్థాయిలను తనిఖీ చేసి తెలపండి.\n")
        
    elif decision in ["ABSTAIN", "CONFLICT"]:
        print("📌 STATUS: SAFETY WARNING / హెచ్చరిక మరియు వైరుధ్యం\n")
        
        print("--- English ---")
        print("Warning: Conflicting or high-risk measurements detected. Please re-test your water parameters or consult an expert before taking action.\n")
        
        print("--- తెలుగు (Telugu) ---")
        print("హెచ్చరిక: మీ ఇచ్చిన కొలతలలో వ్యత్యాసాలు ఉన్నాయి. దయచేసి నీటి నాణ్యతను మళ్లీ పరీక్షించండి లేదా నిపుణులను సంప్రదించండి.")
        
    print("="*65)