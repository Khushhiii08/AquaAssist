import os
import re
import streamlit as st
import json
import shutil
import glob
from gtts import gTTS
from sync_manager import download_and_apply_ota_update
import subprocess
import sys
from response_generator import generate_answer
from diagnostic_engine import load_engine, run_diagnostic, detect_conflicting_measurements
from intent_extractor import extract_hypothesis
from decision_response import get_decision_response
from whisper_transcriber import transcribe_audio

def cleanup_temp_files():
    """Wipes old TTS and Whisper audio files on boot to prevent storage leaks."""
    # 1. Nuke and recreate the TTS folder
    if os.path.exists("temp_audio"):
        shutil.rmtree("temp_audio")
    os.makedirs("temp_audio", exist_ok=True)
    
    # 2. Delete any leftover Whisper WAV files in the root directory
    for wav_file in glob.glob("temp_farmer_audio*.wav"):
        try:
            os.remove(wav_file)
        except OSError:
            pass

# Run cleanup ONLY on the very first load of the session
if "cleanup_done" not in st.session_state:
    cleanup_temp_files()
    st.session_state.cleanup_done = True

def text_to_speech_file(text, filename="response_audio.mp3"):
    """Generates a stable audio file, isolating the target language to save time."""
    try:
        os.makedirs("temp_audio", exist_ok=True)
        file_path = os.path.join("temp_audio", filename)
        
        # Clean text for speech (skip markdown symbols)
        clean_text = text.replace("*", "").replace("#", "").replace("⚠️", "")
        
        # Isolate sentences that contain Telugu script
        telugu_lines = [
            line for line in clean_text.split('\n') 
            if any('\u0c00' <= c <= '\u0c7f' for c in line)
        ]
        
        # If Telugu is present, ONLY read the Telugu portion. Otherwise, read English.
        if telugu_lines:
            speech_text = " ".join(telugu_lines)
            lang = 'te'
        else:
            speech_text = clean_text
            lang = 'en'
            
        tts = gTTS(text=speech_text, lang=lang, slow=False)
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
    
    if st.button("🔄 Check Knowledge Updates", use_container_width=True):
        with st.spinner("Checking connectivity and syncing OTA updates..."):
            
            # For capstone demo, host a zipped version of the data/chroma_db folder 
            # on GitHub, Google Drive (direct link), or an AWS S3 bucket, and paste the URL here:
            CLOUD_DB_URL = "https://github.com/Khushhiii08/AquaAssist/releases/download/v1.1-data/chroma_db.zip"
            
            success, msg = download_and_apply_ota_update(CLOUD_DB_URL)
            
            if success:
                st.success(msg)
                # Force Streamlit to reload ChromaDB so it sees the new data
                get_diagnostic_engine.clear()
            else:
                st.error(msg)

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

def contains_telugu_script(text):
    """Checks if the typed string contains characters from the Telugu Unicode block."""
    return bool(re.search(r'[\u0C00-\u0C7F]', text))

if prompt:
    if prompt.text and prompt.text.strip():
        # Edge Hardware Guardrail: Reject typed Telugu to save translation compute
        if contains_telugu_script(prompt.text):
            st.warning("⚠️ For Telugu support, please click the 🎙️ microphone icon to record your voice. Typed text is currently English-only.")
        else:
            farmer_query = prompt.text.strip()
            
    elif prompt.audio is not None:
        audio_path = "temp_farmer_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(prompt.audio.getbuffer())
        with st.spinner("Transcribing and translating local audio..."):
            farmer_query = transcribe_audio(audio_path)

# --- PIPELINE EXECUTION ---
if farmer_query:
    # Auto-update the session title to the user's first query if it's still "New Chat"
    if current_session["title"] == "New Chat":
        current_session["title"] = farmer_query

    with st.chat_message("user"):
        st.markdown(farmer_query)
    st.session_state.messages.append({"role": "user", "content": farmer_query})

    with st.chat_message("assistant"):
        if detect_conflicting_measurements(farmer_query):
            english_warning = "⚠️ **Safety Warning:** I detected contradictory water measurements in your input. Please recalibrate your sensors and provide the correct value."
            telugu_warning = "**Telugu Advice:** ⚠️ **భద్రతా హెచ్చరిక:** మీ సమాచారంలో విరుద్ధమైన నీటి కొలతలు ఉన్నట్లు నేను గుర్తించాను. దయచేసి మీ సెన్సార్లను సరిచూసుకుని, సరైన విలువను అందించండి."
            
            ui_response = f"{english_warning}\n\n{telugu_warning}"
            st.markdown(ui_response)
            
            # The TTS engine will automatically isolate and speak the Telugu line
            audio_path = text_to_speech_file(ui_response, filename=f"msg_{hash(ui_response) & 0xffffffff}.mp3")
            if audio_path:
                st.audio(audio_path, format="audio/mp3")

            st.session_state.messages.append({"role": "assistant", "content": ui_response})
            
        else:
            with st.spinner("Analyzing telemetry & retrieving evidence..."):
                # Always run the deterministic parser strictly on the isolated current turn
                # Pass st.session_state.messages so the parser knows what was discussed previously
                hypothesis = extract_hypothesis(farmer_query, chat_history=st.session_state.messages)
                
                embedding_model, nli_model, collection = get_diagnostic_engine()
                result = run_diagnostic(hypothesis, embedding_model, nli_model, collection)
                
                decision = result.get("decision", "CLARIFY")
                ui_response = ""

                if decision == "CLARIFY":
                    ui_response = get_decision_response("CLARIFY")["message"]
                elif decision == "ABSTAIN":
                    ui_response = get_decision_response("ABSTAIN")["message"]
                elif decision == "ANSWER":
                    # 1. Get structured bilingual response directly from the metadata router
                    response_data = generate_answer(hypothesis, result["evidence"])
                    
                    # 2. Unpack the clean strings
                    english_text = response_data["english"]
                    telugu_text = response_data["telugu"]

                    # 3. Render the bilingual UX smoothly
                    # (Your text_to_speech_file function will automatically isolate the Telugu line!)
                    ui_response = f"**English Observation:** {english_text}\n\n**Telugu Advice:** {telugu_text}"

            st.markdown(ui_response)
            
            audio_path = text_to_speech_file(ui_response, filename=f"msg_{hash(ui_response) & 0xffffffff}.mp3")
            if audio_path:
                st.audio(audio_path, format="audio/mp3")

            st.session_state.messages.append({"role": "assistant", "content": ui_response})

            with st.expander("🛠️ Pipeline Debug Logs (For Engineering Team)"):
                st.write(f"**Raw Farmer Query:**\n{farmer_query}")
                st.write(f"**Structured Hypothesis:**\n{hypothesis}")
                st.write("**NLI Verification Results:**")
                for chunk in result.get("evidence", []):
                    st.text(
                        f"ID: {chunk['id']} | Dist: {chunk['distance']:.4f}\n"
                        f"Entailment: {chunk['entailment']:.4f} | "
                        f"Neutral: {chunk['neutral']:.4f} | "
                        f"Contradiction: {chunk['contradiction']:.4f}"
                    )