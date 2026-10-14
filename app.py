import os
import re
import sys
import copy
import glob
import types
import asyncio
import hashlib
import tempfile
import threading
import functools
import concurrent.futures

import gradio as gr
import edge_tts

# The pipeline modules live in ./src; Hugging Face Spaces runs `python app.py` from the repo root.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
os.chdir(ROOT_DIR)

# --- ZEROGPU COMPATIBILITY ---
# `spaces` is only meaningful on Hugging Face ZeroGPU; fall back to a no-op decorator elsewhere.
try:
    import spaces
except ImportError:
    def _noop_gpu(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn
    spaces = types.SimpleNamespace(GPU=_noop_gpu)

from sync_manager import download_and_apply_ota_update
from response_generator import generate_answer
from diagnostic_engine import load_engine, run_diagnostic, detect_conflicting_measurements, NLI_MODEL_PATH
from intent_extractor import extract_hypothesis
from decision_response import get_decision_response
from whisper_transcriber import transcribe_audio

# Define GitHub Release URLs for both OTA assets
DB_MAGIC_URL = "https://github.com/Khushhiii08/AquaAssist/releases/latest/download/chroma_db.zip"
NLI_MAGIC_URL = "https://github.com/Khushhiii08/AquaAssist/releases/latest/download/deberta_aquaculture_router.zip"

DB_PATH = "data/chroma_db"
NLI_PATH = "data/deberta_aquaculture_router"
TTS_DIR = "temp_audio"

VOICE_TELUGU = "te-IN-ShrutiNeural"
VOICE_ENGLISH = "en-IN-NeerjaNeural"


def new_diagnostic_state():
    return {"symptoms": set(), "pending_metric": None, "last_subject": None}


def cleanup_temp_files():
    os.makedirs(TTS_DIR, exist_ok=True)
    for wav_file in glob.glob("temp_farmer_audio*.wav"):
        try:
            os.remove(wav_file)
        except OSError:
            pass


# --- TTS (edge-tts, event-loop safe) ---
def _synthesize_to_file(speech_text, voice, file_path):
    """
    Runs edge-tts synchronously. If an asyncio loop is already active in this thread
    (or save_sync refuses to run), the coroutine is executed on a separate thread instead.
    """
    try:
        asyncio.get_running_loop()
        loop_active = True
    except RuntimeError:
        loop_active = False

    if not loop_active:
        try:
            edge_tts.Communicate(text=speech_text, voice=voice).save_sync(file_path)
            return
        except RuntimeError:
            pass  # Fall through to the thread-based fallback

    def _run_in_fresh_loop():
        communicate = edge_tts.Communicate(text=speech_text, voice=voice)
        asyncio.run(communicate.save(file_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_run_in_fresh_loop).result()


@functools.lru_cache(maxsize=50)
def generate_cached_tts(text):
    """Caches on the text input to prevent reprocessing identical alerts."""
    try:
        os.makedirs(TTS_DIR, exist_ok=True)
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        file_path = os.path.join(TTS_DIR, f"audio_{digest}.mp3")

        clean_text = text
        for symbol in ("*", "#", "⚠️", "🚨", "📌"):
            clean_text = clean_text.replace(symbol, "")

        telugu_lines = []
        for line in clean_text.split("\n"):
            stripped = line.strip()
            # 1. Skip "--- తెలుగు (Telugu) ---" dividers and "Status: ... / ..." header lines
            if stripped.startswith("---") or stripped.lower().startswith("status:"):
                continue
            # 2. Keep lines with Telugu script, but strip leading English labels
            if any("\u0c00" <= c <= "\u0c7f" for c in stripped):
                cleaned_line = re.sub(r"(?i)^telugu\s+(advice|action)\s*:\s*", "", stripped).strip()
                if cleaned_line:
                    telugu_lines.append(cleaned_line)

        if telugu_lines:
            speech_text = " ".join(telugu_lines)
            voice = VOICE_TELUGU
        else:
            speech_text = clean_text
            voice = VOICE_ENGLISH

        if not speech_text.strip():
            return None

        _synthesize_to_file(speech_text, voice, file_path)

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return file_path
        return None
    except Exception as e:
        print(f"[!] TTS Error: {e}")
        return None


def synthesize_for_ui(text):
    """Wraps the cached TTS call so a cached-but-deleted file or a cached failure gets regenerated once."""
    path = generate_cached_tts(text)
    if path is None or not os.path.exists(path):
        generate_cached_tts.cache_clear()
        path = generate_cached_tts(text)
    return path


# --- 1. STRICT OTA BOOT ENFORCER ---
# Zero dependency on a pre-existing local data/chroma_db: if it is missing/empty, it must come from the GitHub release.
def ensure_knowledge_base():
    if not os.path.isdir(DB_PATH) or not os.listdir(DB_PATH):
        print("[*] Initial Boot: Extracting Knowledge Base via OTA from GitHub...")
        success, msg = download_and_apply_ota_update(DB_MAGIC_URL, target_dir=DB_PATH)
        if not success:
            raise RuntimeError(f"Critical OTA Failure: Cannot boot without database. {msg}")

def ensure_runtime_models():
    """
    Ensures the fine-tuned DeBERTa NLI router exists in ./models. 
    If missing, downloads and extracts deberta_aquaculture_router.zip via OTA from GitHub releases.
    """
    if not os.path.isdir(NLI_MODEL_PATH) or not os.listdir(NLI_MODEL_PATH):
        print("[*] Initial Boot: Extracting DeBERTa NLI Router via OTA from GitHub...")
        success, msg = download_and_apply_ota_update(NLI_MAGIC_URL, target_dir=NLI_MODEL_PATH)
        if not success:
            raise RuntimeError(
                f"Critical OTA Failure: NLI router model not found at '{NLI_MODEL_PATH}' and GitHub OTA download failed. {msg}"
            )

# --- 2. ENGINE WARM-UP: GLOBAL SINGLETON ---
_ENGINE = None
_ENGINE_LOCK = threading.RLock()


def _reset_chroma_client_cache():
    """Chroma caches clients per path; clear it so a freshly extracted DB is actually re-opened."""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def get_diagnostic_engine(force_reload=False):
    global _ENGINE
    with _ENGINE_LOCK:
        if force_reload or _ENGINE is None:
            _ENGINE = None
            if force_reload:
                _reset_chroma_client_cache()
            _ENGINE = load_engine()
        return _ENGINE


cleanup_temp_files()
ensure_knowledge_base()
ensure_runtime_models()
# Load the LLM/embedding/NLI models *before* the farmer types a message, making the first response instant.
get_diagnostic_engine()


def contains_telugu_script(text):
    return bool(re.search(r"[\u0C00-\u0C7F]", text))


# --- CORE DIAGNOSTIC & SAFETY PIPELINE ---
def _format_debug(farmer_query, hypothesis, evidence, state):
    lines = [f"Raw Farmer Query:\n{farmer_query}", "", f"Structured Hypothesis:\n{hypothesis}", "", "NLI Verification Results:"]
    for chunk in evidence or []:
        lines.append(
            f"ID: {chunk.get('id', 'N/A')} | Dist: {chunk.get('distance', 0.0):.4f}\n"
            f"Entailment: {chunk.get('entailment', 0.0):.4f} | "
            f"Neutral: {chunk.get('neutral', 0.0):.4f} | "
            f"Contradiction: {chunk.get('contradiction', 0.0):.4f}"
        )
    lines += ["", f"Active State: {state}"]
    return "\n".join(lines)


@spaces.GPU
def run_aquaassist_pipeline(farmer_query, audio_path, diag_state):
    """
    Main diagnostic inference function (transcription + hypothesis + retrieval + NLI + decision).
    Pure with respect to its arguments: the updated diagnostic_state is returned, never mutated in place,
    so it behaves identically inside a ZeroGPU worker process.
    """
    out = {
        "ok": True,
        "query": farmer_query,
        "response": "",
        "report": "",
        "citation": "",
        "debug": "",
        "state": diag_state,
        "toast": None,
        "error": None,
    }

    if not farmer_query and audio_path:
        transcribed_text = transcribe_audio(audio_path)
        if not transcribed_text:
            out["ok"] = False
            out["error"] = "Audio transcription failed. Please try speaking closer to the microphone."
            return out
        farmer_query = transcribed_text
        out["query"] = farmer_query

    if detect_conflicting_measurements(farmer_query):
        english_warning = "⚠️ **Safety Warning:** I detected contradictory water measurements in your input. Please recalibrate your sensors and provide the correct value."
        telugu_warning = "**Telugu Advice:** ⚠️ **భద్రతా హెచ్చరిక:** మీ సమాచారంలో విరుద్ధమైన నీటి కొలతలు ఉన్నట్లు నేను గుర్తించాను. దయచేసి మీ సెన్సార్లను సరిచూసుకుని, సరైన విలువను అందించండి."
        out["response"] = f"{english_warning}\n\n{telugu_warning}"
        out["debug"] = _format_debug(farmer_query, "(conflict gate triggered - hypothesis extraction skipped)", [], diag_state)
        return out

    hypothesis, updated_state = extract_hypothesis(farmer_query, session_state=diag_state)
    updated_state = copy.deepcopy(updated_state)

    # --- BIOLOGICAL EMERGENCY SHORT-CIRCUIT ---
    # Bypass ChromaDB and the LLM entirely for lethal thresholds
    telemetry_val = None
    numeric_only = re.search(r"^\s*(\d+(\.\d+)?)\s*$", farmer_query)
    if numeric_only:
        telemetry_val = float(numeric_only.group(1))
    else:
        do_match = re.search(r"(?i)DO.*?(\d+(\.\d+)?)", hypothesis)
        if do_match:
            telemetry_val = float(do_match.group(1))

    if telemetry_val is not None and telemetry_val < 3.0:
        english_text = f"CRITICAL EMERGENCY: Dissolved Oxygen is {telemetry_val} ppm. Shrimp are suffocating. Turn on all aerators immediately and halt all feeding."
        telugu_text = f"అత్యవసర పరిస్థితి: ఆక్సిజన్ స్థాయి {telemetry_val} ppm. రొయ్యలు ఊపిరాడక ఇబ్బంది పడుతున్నాయి. వెంటనే అన్ని ఏరేటర్లను ఆన్ చేయండి మరియు ఆహారం ఇవ్వడం ఆపండి."

        out["response"] = f"🚨 **English Action:** {english_text}\n\n🚨 **Telugu Action:** {telugu_text}"
        out["toast"] = "Lethal Water Parameter Detected!"
        # --- WIPE STATE AFTER EMERGENCY ---
        out["state"] = new_diagnostic_state()
        out["debug"] = _format_debug(farmer_query, hypothesis, [], out["state"])
        return out

    # If no emergency, proceed to standard RAG pipeline
    diag_state = updated_state

    embedding_model, nli_model, collection = get_diagnostic_engine()
    result = run_diagnostic(hypothesis, embedding_model, nli_model, collection)

    decision = result.get("decision", "CLARIFY")

    # --- HARD TELEMETRY GATE ---
    # Only force clarification if it's a diagnostic symptom query missing numbers
    has_telemetry = bool(re.search(r"\d", farmer_query))
    is_symptom_query = "exhibits" in hypothesis or any(s in hypothesis.lower() for s in ["eating", "swimming", "dying", "gasping"])

    if not has_telemetry and is_symptom_query and decision == "ANSWER":
        decision = "CLARIFY"

    ui_response = ""

    if decision == "CLARIFY":
        diag_state["pending_metric"] = "DO"
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

        # --- Extract Metadata for Citation ---
        source_doc, page_num = "Unknown Manual", "N/A"
        if result.get("evidence"):
            top_chunk = result["evidence"][0]
            source_doc = top_chunk["metadata"].get("source", "Unknown Manual")
            page_num = top_chunk["metadata"].get("page", "N/A")
            out["citation"] = f"> 📖 **Source:** {source_doc}, Page {page_num} (Distance: {top_chunk['distance']:.3f})"

        # Construct Export Log content
        out["report"] = (
            "=== AquaAssist Diagnostic Log ===\n"
            f"System State: {updated_state}\n\n"
            "--- English Action ---\n"
            f"{english_text}\n\n"
            "--- Telugu Action ---\n"
            f"{telugu_text}\n\n"
            f"Reference: {source_doc} (pg. {page_num})"
        )

    out["response"] = ui_response
    out["state"] = diag_state
    out["debug"] = _format_debug(farmer_query, hypothesis, result.get("evidence", []), diag_state)
    return out


# --- SESSION HELPERS ---
def _session_label(session):
    title = session["title"]
    return title if len(title) < 28 else title[:25] + "..."


def _get_session(sessions, session_id):
    return next((s for s in sessions if s["id"] == session_id), sessions[0])


def _radio_update(sessions, current_id):
    return gr.update(choices=[(_session_label(s), s["id"]) for s in sessions], value=current_id)


def _write_report(report_content):
    report_dir = tempfile.mkdtemp(dir=TTS_DIR)
    report_path = os.path.join(report_dir, "AquaAssist_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return report_path


# Output order shared by every handler that touches the conversation:
# chatbot, sessions, current_session_id, diagnostic_state, text_in, audio_in, tts_out, export_btn, debug_box, session_radio


def handle_submit(text, audio_path, sessions, current_id, diag_state):
    unchanged = (gr.update(),) * 10
    text = (text or "").strip()

    if text:
        if contains_telugu_script(text):
            gr.Warning("For Telugu support, please click the 🎙️ microphone icon to record your voice. Typed text is currently English-only.")
            return unchanged
        query, audio_for_pipeline = text, None
    elif audio_path:
        query, audio_for_pipeline = None, audio_path
    else:
        gr.Warning("Describe the pond symptoms or record your voice first.")
        return unchanged

    result = run_aquaassist_pipeline(query, audio_for_pipeline, diag_state)

    if not result["ok"]:
        gr.Warning(result["error"])
        return unchanged
    if result["toast"]:
        gr.Warning(result["toast"])

    farmer_query = result["query"]
    session = _get_session(sessions, current_id)
    if session["title"] == "New Chat":
        session["title"] = farmer_query

    # TTS is generated from the plain response; the citation is display-only.
    audio_out = synthesize_for_ui(result["response"])
    display = result["response"] + (f"\n\n{result['citation']}" if result["citation"] else "")

    session["messages"].append({"role": "user", "content": farmer_query})
    session["messages"].append({"role": "assistant", "content": display})
    session["diagnostic_state"] = result["state"]

    if result["report"]:
        export_update = gr.update(value=_write_report(result["report"]), visible=True)
    else:
        export_update = gr.update(value=None, visible=False)

    return (
        list(session["messages"]),
        sessions,
        current_id,
        result["state"],
        "",
        None,
        audio_out,
        export_update,
        result["debug"],
        _radio_update(sessions, current_id),
    )


def handle_new_chat(sessions, current_id, diag_state):
    _get_session(sessions, current_id)["diagnostic_state"] = diag_state
    new_id = max(s["id"] for s in sessions) + 1
    fresh_state = new_diagnostic_state()
    sessions.insert(0, {"id": new_id, "title": "New Chat", "messages": [], "diagnostic_state": copy.deepcopy(fresh_state)})
    return (
        [],
        sessions,
        new_id,
        fresh_state,
        "",
        None,
        None,
        gr.update(value=None, visible=False),
        "",
        _radio_update(sessions, new_id),
    )


def handle_select_session(selected_id, sessions, current_id, diag_state):
    if selected_id is None or selected_id == current_id:
        return (gr.update(),) * 10
    _get_session(sessions, current_id)["diagnostic_state"] = diag_state
    target = _get_session(sessions, selected_id)
    restored_state = copy.deepcopy(target.get("diagnostic_state") or new_diagnostic_state())
    return (
        list(target["messages"]),
        sessions,
        target["id"],
        restored_state,
        "",
        None,
        None,
        gr.update(value=None, visible=False),
        "",
        _radio_update(sessions, target["id"]),
    )


def handle_ota_update():
    success, msg = download_and_apply_ota_update(DB_MAGIC_URL, target_dir=DB_PATH)
    if not success:
        gr.Warning(msg)
        return f"❌ {msg}"
    try:
        get_diagnostic_engine(force_reload=True)  # Reload immediately
    except Exception as e:
        gr.Warning(f"Knowledge base updated, but the engine failed to reload: {e}")
        return f"❌ Knowledge base updated, but the engine failed to reload: {e}"
    gr.Info(msg)
    return f"✅ {msg}"


# --- UI ---
with gr.Blocks(title="AquaAssist", theme=gr.themes.Soft()) as demo:
    sessions_state = gr.State([{"id": 0, "title": "New Chat", "messages": [], "diagnostic_state": new_diagnostic_state()}])
    current_session_id = gr.State(0)
    diagnostic_state = gr.State(new_diagnostic_state())

    with gr.Sidebar():
        gr.Markdown("### 🐟 AquaAssist")
        new_chat_btn = gr.Button("➕ New Chat", variant="primary")
        gr.Markdown("**Chat History**")
        session_radio = gr.Radio(choices=[("New Chat", 0)], value=0, label="Sessions", show_label=False, interactive=True)
        gr.Markdown("---\n**System Status & Sync**")
        gr.Markdown("🟢 Mode: Local-First (Offline Ready)")
        ota_btn = gr.Button("🔄 Check Knowledge Updates")
        ota_status = gr.Markdown("")

    gr.Markdown("# 🐟 AquaAssist")
    gr.Markdown("Evidence-Aware Aquaculture Assistant — తెలుగు & English Support")

    chatbot = gr.Chatbot(type="messages", label="Conversation", height=460)
    tts_out = gr.Audio(label="Spoken Response", autoplay=True, interactive=False, type="filepath")

    with gr.Row():
        text_in = gr.Textbox(placeholder="Describe pond symptoms...", label="Type (English)", lines=2, scale=3)
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎙️ Record / upload voice (Telugu or English)", scale=2)
    send_btn = gr.Button("Send", variant="primary")

    export_btn = gr.DownloadButton("📥 Export Diagnostic Log", visible=False)

    with gr.Accordion("🛠️ Pipeline Debug Logs (For Engineering Team)", open=False):
        debug_box = gr.Textbox(label="Pipeline trace", lines=12, interactive=False)

    conversation_outputs = [chatbot, sessions_state, current_session_id, diagnostic_state, text_in, audio_in, tts_out, export_btn, debug_box, session_radio]
    submit_inputs = [text_in, audio_in, sessions_state, current_session_id, diagnostic_state]

    send_btn.click(handle_submit, submit_inputs, conversation_outputs)
    text_in.submit(handle_submit, submit_inputs, conversation_outputs)
    new_chat_btn.click(handle_new_chat, [sessions_state, current_session_id, diagnostic_state], conversation_outputs)
    # `.input` fires only on user selection, not when handlers update the radio programmatically.
    session_radio.input(handle_select_session, [session_radio, sessions_state, current_session_id, diagnostic_state], conversation_outputs)
    ota_btn.click(handle_ota_update, None, ota_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[os.path.abspath(TTS_DIR)])