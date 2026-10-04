import os
import re
import streamlit as st
import json
import shutil
import glob
from gtts import gTTS
from sync_manager import download_and_apply_ota_update, get_latest_github_release_url
import subprocess
import sys
from response_generator import generate_answer
from diagnostic_engine import load_engine, run_diagnostic, detect_conflicting_measurements
from intent_extractor import extract_hypothesis
from decision_response import get_decision_response
from whisper_transcriber import transcribe_audio

def cleanup_temp_files():
    os.makedirs("temp_audio", exist_ok=True)
    for wav_file in glob.glob("temp_farmer_audio*.wav"):
        try:
            os.remove(wav_file)
        except OSError:
            pass

if "cleanup_done" not in st.session_state:
    cleanup_temp_files()
    st.session_state.cleanup_done = True

# --- TTS CACHING FIX ---
@st.cache_data(show_spinner=False, max_entries=50)
def generate_cached_tts(text):
    """Hashes the text input to prevent reprocessing identical alerts."""
    try:
        os.makedirs("temp_audio", exist_ok=True)
        # Unique filename based on text hash
        filename = f"audio_{hash(text) & 0xffffffff}.mp3"
        file_path = os.path.join("temp_audio", filename)
        
        clean_text = text.replace("*", "").replace("#", "").replace("⚠️", "")
        telugu_lines = [line for line in clean_text.split('\n') if any('\u0c00' <= c <= '\u0c7f' for c in line)]
        
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

# Define exactly once
MAGIC_URL = "https://github.com/Khushhiii08/AquaAssist/releases/latest/download/chroma_db.zip"
DB_PATH = "data/chroma_db"

# --- 1. STRICT OTA BOOT ENFORCER ---
# Runs immediately on page load, completely blocking the UI if the DB is missing
if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
    with st.spinner("Initial Boot: Extracting Knowledge Base via OTA from GitHub..."):
        success, msg = download_and_apply_ota_update(MAGIC_URL)
        if not success:
            st.error(f"Critical OTA Failure: Cannot boot without database. {msg}")
            st.stop() # Halts the app entirely

# --- 2. ENGINE WARM-UP ---
@st.cache_resource(show_spinner=False)
def get_diagnostic_engine():
    return load_engine()

# Call it immediately on boot so the LLM and Embedding models load into Mac memory 
# *before* the farmer types a message, making the first response instant.
get_diagnostic_engine() 

# --- INITIALIZE CONVERSATIONAL STATE ---
if "diagnostic_state" not in st.session_state:
    st.session_state.diagnostic_state = {"symptoms": set(), "pending_metric": None, "last_subject": None}

with st.sidebar:
    st.image("https://img.icons8.com/color/96/fish.png", width=36)
    st.markdown("### AquaAssist")
    
    if "sessions" not in st.session_state or not isinstance(st.session_state.sessions, list):
        st.session_state.sessions = [{"id": 0, "title": "New Chat", "messages": []}]
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = 0
        
    if st.button("➕ New Chat", use_container_width=True):
        new_id = len(st.session_state.sessions)
        st.session_state.sessions.insert(0, {"id": new_id, "title": "New Chat", "messages": []})
        st.session_state.current_session_id = new_id
        st.session_state.diagnostic_state = {"symptoms": set(), "pending_metric": None, "last_subject": None}
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Chat History**")
    
    current_session = next((s for s in st.session_state.sessions if s["id"] == st.session_state.current_session_id), st.session_state.sessions[0])
    
    for session in st.session_state.sessions:
        is_current = (session["id"] == st.session_state.current_session_id)
        button_type = "primary" if is_current else "secondary"
        display_title = session["title"] if len(session["title"]) < 28 else session["title"][:25] + "..."
        
        if st.button(display_title, key=f"chat_s_{session['id']}", use_container_width=True, type=button_type):
            if session["id"] != st.session_state.current_session_id:
                st.session_state.current_session_id = session["id"]
                st.rerun()

    # --- OTA UPDATE RESTORED HERE ---
    st.markdown("---")
    st.markdown("**System Status & Sync**")
    st.info("🟢 Mode: Local-First (Offline Ready)")
    
    if st.button("🔄 Check Knowledge Updates", use_container_width=True):
        with st.spinner("Downloading latest database directly from GitHub..."):
            
            # Uses the global MAGIC_URL defined at the top of the script
            success, msg = download_and_apply_ota_update(MAGIC_URL)
            
            if success:
                st.success(msg)
                get_diagnostic_engine.clear()
                get_diagnostic_engine() # Reload immediately
            else:
                st.error(msg)

st.session_state.messages = current_session["messages"]

st.title("🐟 AquaAssist")
st.caption("Evidence-Aware Aquaculture Assistant — తెలుగు & English Support")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            # Utilize the new caching mechanism
            audio_path = generate_cached_tts(message["content"])
            if audio_path:
                st.audio(audio_path, format="audio/mp3")

prompt = st.chat_input("Describe pond symptoms or record voice...", accept_audio=True)
farmer_query = None

def contains_telugu_script(text):
    return bool(re.search(r'[\u0C00-\u0C7F]', text))

if prompt:
    if prompt.text and prompt.text.strip():
        if contains_telugu_script(prompt.text):
            st.warning("⚠️ For Telugu support, please click the 🎙️ microphone icon to record your voice. Typed text is currently English-only.")
        else:
            farmer_query = prompt.text.strip()
            
    elif prompt.audio is not None:
        audio_path = "temp_farmer_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(prompt.audio.getbuffer())
            
        with st.spinner("Transcribing vernacular audio locally..."):
            transcribed_text = transcribe_audio(audio_path)
            if transcribed_text:
                farmer_query = transcribed_text
            else:
                st.error("Audio transcription failed. Please try speaking closer to the microphone.")

if farmer_query:
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
            
            audio_path = generate_cached_tts(ui_response)
            if audio_path:
                st.audio(audio_path, format="audio/mp3")

            st.session_state.messages.append({"role": "assistant", "content": ui_response})
            
        else:
            with st.spinner("Analyzing telemetry & retrieving evidence..."):
                hypothesis, updated_state = extract_hypothesis(
                    farmer_query, 
                    session_state=st.session_state.diagnostic_state
                )
                
                # --- NEW: BIOLOGICAL EMERGENCY SHORT-CIRCUIT ---
                # Bypass ChromaDB and the LLM entirely for lethal thresholds
                telemetry_val = None
                numeric_only = re.search(r'^\s*(\d+(\.\d+)?)\s*$', farmer_query)
                if numeric_only:
                    telemetry_val = float(numeric_only.group(1))
                else:
                    do_match = re.search(r'(?i)DO.*?(\d+(\.\d+)?)', hypothesis)
                    if do_match:
                        telemetry_val = float(do_match.group(1))

                if telemetry_val is not None and telemetry_val < 3.0:
                    english_text = f"CRITICAL EMERGENCY: Dissolved Oxygen is {telemetry_val} ppm. Shrimp are suffocating. Turn on all aerators immediately and halt all feeding."
                    telugu_text = f"అత్యవసర పరిస్థితి: ఆక్సిజన్ స్థాయి {telemetry_val} ppm. రొయ్యలు ఊపిరాడక ఇబ్బంది పడుతున్నాయి. వెంటనే అన్ని ఏరేటర్లను ఆన్ చేయండి మరియు ఆహారం ఇవ్వడం ఆపండి."
                    
                    ui_response = f"🚨 **English Action:** {english_text}\n\n🚨 **Telugu Action:** {telugu_text}"
                    
                    st.error("Lethal Water Parameter Detected!")
                    st.markdown(ui_response)
                    
                    audio_path = generate_cached_tts(ui_response)
                    if audio_path:
                        st.audio(audio_path, format="audio/mp3")
                        
                    # Save to chat history and immediately reload to halt standard RAG
                    st.session_state.messages.append({"role": "assistant", "content": ui_response})
                    
                    # --- NEW: WIPE STATE AFTER EMERGENCY ---
                    st.session_state.diagnostic_state = {"symptoms": set(), "pending_metric": None, "last_subject": None}
                    st.rerun() 
                # -------------------------------------------
                
                # If no emergency, proceed to standard RAG pipeline
                st.session_state.diagnostic_state = updated_state
                
                embedding_model, nli_model, collection = get_diagnostic_engine()
                result = run_diagnostic(hypothesis, embedding_model, nli_model, collection)
                
                decision = result.get("decision", "CLARIFY")
                
                # --- NEW: HARD TELEMETRY GATE ---
                # Only force clarification if it's a diagnostic symptom query missing numbers
                has_telemetry = bool(re.search(r'\d', farmer_query))
                is_symptom_query = "exhibits" in hypothesis or any(s in hypothesis.lower() for s in ["eating", "swimming", "dying", "gasping"])
                
                if not has_telemetry and is_symptom_query and decision == "ANSWER":
                    decision = "CLARIFY"
                # --------------------------------
                
                ui_response = ""
                report_content = ""

                if decision == "CLARIFY":
                    st.session_state.diagnostic_state["pending_metric"] = "DO"
                    ui_response = get_decision_response("CLARIFY")["message"]
                elif decision == "ABSTAIN":
                    ui_response = get_decision_response("ABSTAIN")["message"]
                elif decision == "ANSWER":
                    response_data = generate_answer(hypothesis, result["evidence"])
                    english_text = response_data["english"]
                    telugu_text = response_data["telugu"]

                    if "skip" in english_text.lower() or "skip" in telugu_text.lower():
                        english_text = "Critical observation detected, but standard evidence was filtered. Check aeration and water quality immediately."
                        telugu_text = "క్లిష్టమైన పరిస్థితి గుర్తించబడింది. దయచేసి వెంటనే ఏరేటర్లను ఆన్ చేసి నీటి నాణ్యతను తనిఖీ చేయండి."

                    ui_response = f"**English Observation:** {english_text}\n\n**Telugu Advice:** {telugu_text}"
                    
                    # --- NEW: Extract Metadata for Citation ---
                    top_chunk = result["evidence"][0]
                    source_doc = top_chunk["metadata"].get("source", "Unknown Manual")
                    page_num = top_chunk["metadata"].get("page", "N/A")
                    
                    # Render Academic Citation
                    st.caption(f"**Source:** {source_doc}, Page {page_num} (Distance: {top_chunk['distance']:.3f})")
                    
                    # Construct Export Log content
                    report_content = (
                        "=== AquaAssist Diagnostic Log ===\n"
                        f"System State: {updated_state}\n\n"
                        "--- English Action ---\n"
                        f"{english_text}\n\n"
                        "--- Telugu Action ---\n"
                        f"{telugu_text}\n\n"
                        f"Reference: {source_doc} (pg. {page_num})"
                    )

            st.markdown(ui_response)
            
            audio_path = generate_cached_tts(ui_response)
            if audio_path:
                st.audio(audio_path, format="audio/mp3")

            st.session_state.messages.append({"role": "assistant", "content": ui_response})
            
            # --- NEW: Export Diagnostic Button ---
            if report_content:
                st.download_button(
                    label="📥 Export Diagnostic Log",
                    data=report_content,
                    file_name="AquaAssist_Report.txt",
                    mime="text/plain"
                )

            with st.expander("🛠️ Pipeline Debug Logs (For Engineering Team)"):
                st.write(f"**Raw Farmer Query:**\n{farmer_query}")
                st.write(f"**Structured Hypothesis:**\n{hypothesis}")
                st.write("**NLI Verification Results:**")
                for chunk in result.get("evidence", []):
                    st.text(
                        f"ID: {chunk.get('id', 'N/A')} | Dist: {chunk.get('distance', 0.0):.4f}\n"
                        f"Entailment: {chunk.get('entailment', 0.0):.4f} | "
                        f"Neutral: {chunk.get('neutral', 0.0):.4f} | "
                        f"Contradiction: {chunk.get('contradiction', 0.0):.4f}"
                    )
                st.write(f"**Active State:** {st.session_state.diagnostic_state}")