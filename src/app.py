import os
import streamlit as st
import json
from gtts import gTTS
from response_generator import generate_answer
from diagnostic_engine import load_engine, run_diagnostic, detect_conflicting_measurements
from intent_extractor import extract_hypothesis
from decision_response import get_decision_response
from whisper_transcriber import transcribe_audio

def text_to_speech_file(text, filename="response_audio.mp3"):
    """Generates a stable audio file using gTTS for flawless browser playback."""
    try:
        os.makedirs("temp_audio", exist_ok=True)
        file_path = os.path.join("temp_audio", filename)
        
        # Clean text for speech (skip markdown symbols)
        clean_text = text.replace("*", "").replace("#", "").replace("⚠️", "")
        
        # Default to Telugu ('te') if Telugu script is present, otherwise English ('en')
        lang = 'te' if any('\u0c00' <= c <= '\u0c7f' for c in clean_text) else 'en'
        
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        tts.save(file_path)
        
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return file_path
        return None
    except Exception as e:
        print(f"[!] TTS Error: {e}")
        return None

st.set_page_config(page_title="AquaAssist", page_icon="🐟", layout="centered")

@st.cache_resource
def get_diagnostic_engine():
    return load_engine()

# --- CHATGPT-STYLE SIDEBAR HISTORY ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/fish.png", width=36)
    st.markdown("### AquaAssist")
    
    # Ensure sessions is stored as a list of dicts to avoid legacy KeyErrors
    if "sessions" not in st.session_state or not isinstance(st.session_state.sessions, list):
        st.session_state.sessions = [{"id": 0, "title": "New Chat", "messages": []}]
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = 0
        
    # "+ New Chat" button like ChatGPT
    if st.button("➕ New Chat", use_container_width=True):
        new_id = len(st.session_state.sessions)
        st.session_state.sessions.insert(0, {"id": new_id, "title": "New Chat", "messages": []})
        st.session_state.current_session_id = new_id
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Chat History**")
    
    # Find current session dictionary safely
    current_session = next((s for s in st.session_state.sessions if s["id"] == st.session_state.current_session_id), st.session_state.sessions[0])
    
    # Render history list as clean clickable buttons
    for session in st.session_state.sessions:
        is_current = (session["id"] == st.session_state.current_session_id)
        button_type = "primary" if is_current else "secondary"
        
        display_title = session["title"] if len(session["title"]) < 28 else session["title"][:25] + "..."
        
        if st.button(display_title, key=f"chat_s_{session['id']}", use_container_width=True, type=button_type):
            if session["id"] != st.session_state.current_session_id:
                st.session_state.current_session_id = session["id"]
                st.rerun()

    st.markdown("---")
    st.markdown("**System Status & Sync**")
    st.info("🟢 Mode: Local-First (Offline Ready)")
    
    from sync_manager import check_for_updates
    if st.button("🔄 Check Knowledge Updates", use_container_width=True):
        with st.spinner("Checking connectivity and sync manifest..."):
            has_update, msg = check_for_updates()
            if has_update:
                st.success(msg)
                # Here you can trigger your ingestion script or delta download
            else:
                st.info(msg)

# Link active messages list to the selected session
st.session_state.messages = current_session["messages"]

# --- HEADER ---
st.title("🐟 AquaAssist")
st.caption("Evidence-Aware Aquaculture Assistant — తెలుగు & English Support")

# Render active chat history cleanly (with persistent audio support across session switches)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            audio_path = text_to_speech_file(message["content"], filename=f"msg_{hash(message['content']) & 0xffffffff}.mp3")
            if audio_path:
                st.audio(audio_path, format="audio/mp3")

# --- UNIFIED CHAT INPUT ---
prompt = st.chat_input("Describe pond symptoms or record voice...", accept_audio=True)
farmer_query = None

if prompt:
    if prompt.text and prompt.text.strip():
        farmer_query = prompt.text.strip()
    elif prompt.audio is not None:
        audio_path = "temp_farmer_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(prompt.audio.getbuffer())
        with st.spinner("Transcribing local audio via Whisper..."):
            farmer_query = transcribe_audio(audio_path)

# --- PIPELINE EXECUTION ---
if farmer_query:
    # Auto-update the session title to the user's first query if it's still "New Chat"
    if current_session["title"] == "New Chat":
        current_session["title"] = farmer_query

    with st.chat_message("user"):
        st.markdown(farmer_query)
    st.session_state.messages.append({"role": "user", "content": farmer_query})
    
    recent_history = st.session_state.messages[-5:-1]
    context_string = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history])
    enriched_query = f"Previous Context:\n{context_string}\n\nFarmer's Current Statement: {farmer_query}" if context_string else farmer_query

    with st.chat_message("assistant"):
        if detect_conflicting_measurements(farmer_query):
            ui_response = "⚠️ **Safety Warning:** I detected contradictory water measurements in your input. Please recalibrate your sensors and provide the correct value."
            st.markdown(ui_response)
            
            audio_path = text_to_speech_file(ui_response, filename=f"msg_{hash(ui_response) & 0xffffffff}.mp3")
            if audio_path:
                st.audio(audio_path, format="audio/mp3")

            st.session_state.messages.append({"role": "assistant", "content": ui_response})
            
        else:
            with st.spinner("Analyzing telemetry & retrieving evidence..."):
                if any(char.isdigit() for char in farmer_query) and ("do" in farmer_query.lower() or "oxygen" in farmer_query.lower() or "ph" in farmer_query.lower()):
                    hypothesis = farmer_query
                else:
                    hypothesis = extract_hypothesis(enriched_query)
                
                embedding_model, nli_model, collection = get_diagnostic_engine()
                result = run_diagnostic(hypothesis, embedding_model, nli_model, collection)
                
                decision = result.get("decision", "CLARIFY")
                ui_response = ""

                if decision == "CLARIFY":
                    ui_response = get_decision_response("CLARIFY")["message"]
                elif decision == "ABSTAIN":
                    ui_response = get_decision_response("ABSTAIN")["message"]
                elif decision == "ANSWER":
                    raw_final_response = generate_answer(hypothesis, result["evidence"])
                    try:
                        parsed_data = json.loads(raw_final_response)
                        ui_response = parsed_data.get("recommended_action_telugu", raw_final_response)
                    except (json.JSONDecodeError, TypeError):
                        ui_response = raw_final_response

            st.markdown(ui_response)
            
            audio_path = text_to_speech_file(ui_response, filename=f"msg_{hash(ui_response) & 0xffffffff}.mp3")
            if audio_path:
                st.audio(audio_path, format="audio/mp3")

            st.session_state.messages.append({"role": "assistant", "content": ui_response})

            with st.expander("🛠️ Pipeline Debug Logs (For Engineering Team)"):
                st.write(f"**Enriched Query to Extractor:**\n{enriched_query}")
                st.write(f"**Extracted Hypothesis:**\n{hypothesis}")
                st.write("**NLI Verification Results:**")
                for chunk in result.get("evidence", []):
                    st.text(
                        f"ID: {chunk['id']} | Dist: {chunk['distance']:.4f}\n"
                        f"Entailment: {chunk['entailment']:.4f} | "
                        f"Neutral: {chunk['neutral']:.4f} | "
                        f"Contradiction: {chunk['contradiction']:.4f}"
                    )