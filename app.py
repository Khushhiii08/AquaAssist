"""
AquaAssist: Production Gradio Web Application & Orchestrator
Integrated Neuro-Symbolic Pipeline for Aquaculture Limnology & Diagnostics.
Features:
- Pure State Flow & Conversational Anaphora Resolution (T19)
- Multi-Tier Policy & Chemical Safety Gates (T7, T8, T9, T10, T12)
- Species Preservation & Structured Telemetry Evaluation (T4, T11)
- Non-blocking Typed Telugu & Voice Translation (T13-T18)
- Safe Pointer-Based Database Synchronization (T20)
"""

import os
import re
import sys
import copy
import glob
import types
import torch
import asyncio
import hashlib
import tempfile
import threading
import functools
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple

import gradio as gr
import edge_tts

# Prevent CPU thread contention on Hugging Face Spaces 2 vCPU tier
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)

# Path initialization
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.chdir(ROOT_DIR)

# Safe ZeroGPU initialization
try:
    import spaces
except ImportError:
    def _noop_gpu(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn
    spaces = types.SimpleNamespace(GPU=_noop_gpu)

# Subsystem Imports
from src.intent_extractor import extract_hypothesis
from src.decision_router import route_query_policy
from src.decision_response import render_policy_response
from src.diagnostic_engine import (
    load_engine,
    run_diagnostic,
    evaluate_telemetry_health_and_conflicts,
    NLI_MODEL_PATH
)
from src.response_generator import generate_answer
from src.whisper_transcriber import transcribe_audio
from src.sync_manager import download_and_apply_ota_update, get_active_db_path

# Static Configurations
DB_MAGIC_URL = "https://github.com/Khushhiii08/AquaAssist/releases/latest/download/chroma_db.zip"
NLI_MAGIC_URL = "https://github.com/Khushhiii08/AquaAssist/releases/latest/download/deberta_aquaculture_router.zip"
TTS_DIR = os.path.join(ROOT_DIR, "temp_audio")
VOICE_TELUGU = "te-IN-ShrutiNeural"
VOICE_ENGLISH = "en-IN-NeerjaNeural"

os.makedirs(TTS_DIR, exist_ok=True)


# --- 1. CONVERSATIONAL STATE & COREFERENCE RESOLUTION ---
def new_diagnostic_state() -> Dict[str, Any]:
    """Initializes pure, typed diagnostic state."""
    return {
        "symptoms": set(),
        "species": None,
        "habitat": None,
        "telemetry": {},
        "pending_metrics": None,
        "last_query": ""
    }


def resolve_conversational_coreference(current_query: str, session_history: List[Dict[str, str]], state: Dict[str, Any]) -> str:
    q_low = current_query.strip().lower()
    
    # Invariant: If the farmer is reporting numerical readings or calibration updates, DO NOT rewrite as an elliptical question!
    has_telemetry_reading = any(char.isdigit() for char in q_low) and any(
        u in q_low for u in ["mg/l", "ppm", "ppt", "ph", "reads", "reading", "probe", "meter"]
    )
    if has_telemetry_reading:
        return current_query

    pronoun_markers = {"that", "it", "this", "these", "those", "about that", "do now", "what now", "remedy", "fix it"}
    words = q_low.split()
    is_elliptical = len(words) <= 6 or any(p in q_low for p in pronoun_markers)

    if not is_elliptical or not session_history:
        return current_query

    species = state.get("species") or "fish"
    active_params = list(state.get("telemetry", {}).keys())
    symptoms = list(state.get("symptoms", []))

    # Formulate a clean clinical search prompt
    topics = []
    if "DO" in active_params:
        topics.append("low dissolved oxygen aeration emergency hypoxia treatment")
    if symptoms:
        topics.extend(symptoms)

    if topics:
        return f"{species} pond management: {' '.join(topics)}"

    return f"{species} pond water quality management and corrective measures"

# --- 2. AUDIO SYNTHESIS & TTS ---
def _synthesize_to_file(speech_text: str, voice: str, file_path: str):
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
            pass

    def _run_in_fresh_loop():
        communicate = edge_tts.Communicate(text=speech_text, voice=voice)
        asyncio.run(communicate.save(file_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_run_in_fresh_loop).result()


@functools.lru_cache(maxsize=50)
def generate_cached_tts(text: str) -> Optional[str]:
    try:
        os.makedirs(TTS_DIR, exist_ok=True)
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        file_path = os.path.join(TTS_DIR, f"audio_{digest}.mp3")

        clean_text = text
        for symbol in ("*", "#", "`", "_"):
            clean_text = clean_text.replace(symbol, "")

        telugu_lines = []
        for line in clean_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("---") or stripped.lower().startswith("status:"):
                continue
            if any("\u0c00" <= c <= "\u0c7f" for c in stripped):
                cleaned_line = re.sub(r"(?i)^(telugu|సాంకేతిక సలహా|సలహా)\s*:\s*", "", stripped).strip()
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
        return file_path if os.path.exists(file_path) and os.path.getsize(file_path) > 0 else None
    except Exception as e:
        print(f"[!] TTS Error: {e}")
        return None


def synthesize_for_ui(text: str) -> Optional[str]:
    path = generate_cached_tts(text)
    if path is None or not os.path.exists(path):
        generate_cached_tts.cache_clear()
        path = generate_cached_tts(text)
    return path


# --- 3. RUNTIME BOOTSTRAPPING & OTA SAFETY ---
def ensure_knowledge_base():
    active_db = get_active_db_path()
    if not os.path.isdir(active_db) or not os.listdir(active_db):
        print("[*] Initial Boot: Downloading Knowledge Base via OTA...")
        success, msg = download_and_apply_ota_update(DB_MAGIC_URL)
        if not success:
            raise RuntimeError(f"Critical Boot Failure: Database missing. {msg}")


def ensure_runtime_models():
    if not os.path.isdir(NLI_MODEL_PATH) or not os.listdir(NLI_MODEL_PATH):
        print("[*] Initial Boot: Downloading DeBERTa Router via OTA...")
        success, msg = download_and_apply_ota_update(NLI_MAGIC_URL, target_dir=NLI_MODEL_PATH)
        if not success:
            raise RuntimeError(f"Critical Boot Failure: NLI model missing. {msg}")

_ENGINE = None
_ENGINE_LOCK = threading.RLock()


def get_diagnostic_engine(force_reload=False):
    global _ENGINE
    with _ENGINE_LOCK:
        if force_reload or _ENGINE is None:
            _ENGINE = None
            _ENGINE = load_engine()
        return _ENGINE


ensure_knowledge_base()
ensure_runtime_models()
get_diagnostic_engine()


# --- 4. THE UNIFIED PIPELINE ---
def _format_debug(
    farmer_query: str,
    hypothesis: str,
    evidence: List[Dict[str, Any]],
    state: Dict[str, Any],
    stage_info: Optional[Dict[str, Any]] = None
) -> str:
    stage_info = stage_info or {}
    policy_flag = stage_info.get("policy", "CLEARED (Passed Tiers 1-4)")
    health_flag = stage_info.get("health", "NORMAL (No Fatal Telemetry)")
    
    lines = [
        "============================================================",
        "          AQUAASSIST NEURO-SYMBOLIC PIPELINE TRACE          ",
        "============================================================",
        f" [1] RAW INPUT: \"{farmer_query}\"",
        f" [2] CONVERSATIONAL REWRITE: \"{hypothesis}\"",
        "------------------------------------------------------------",
        f" [3] SAFETY POLICY GATE : {policy_flag}",
        f" [4] TELEMETRY HEALTH   : {health_flag}",
        "------------------------------------------------------------",
        " [5] EXTRACTED SYMBOLIC STATE:",
        f"     • Species Detected : {state.get('species')}",
        f"     • Habitat          : {state.get('habitat')}",
        f"     • Telemetry Matrix : {state.get('telemetry')}",
        f"     • Symptoms Active  : {list(state.get('symptoms') or [])}",
        "------------------------------------------------------------",
        " [6] CHROMADB TOP-K EVIDENCE & DeBERTa NLI ENTAILMENT:"
    ]
    
    if evidence:
        for idx, chunk in enumerate(evidence, 1):
            lines.append(
                f"     [{idx}] Doc: {chunk.get('metadata', {}).get('source', 'Manual')} (p. {chunk.get('metadata', {}).get('page', '?')})\n"
                f"         Vector Dist: {chunk.get('distance', 0.0):.4f} | "
                f"Entailment: {chunk.get('entailment', 0.0):.3f} | "
                f"Contradiction: {chunk.get('contradiction', 0.0):.3f}"
            )
    else:
        lines.append("     (No external retrieval triggered / Intercepted or Fast-Path)")
        
    lines.append("============================================================")
    return "\n".join(lines)


@spaces.GPU
def run_aquaassist_pipeline(
    farmer_query: Optional[str],
    audio_path: Optional[str],
    diag_state: Optional[Dict[str, Any]],
    session_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Unified Neuro-Symbolic Pipeline. Pure with respect to its arguments.
    """
    if diag_state is None:
        diag_state = new_diagnostic_state()

    out = {
        "ok": True,
        "query": farmer_query or "",
        "response": "",
        "report": "",
        "citation": "",
        "debug": "",
        "state": copy.deepcopy(diag_state),
        "toast": None,
        "error": None
    }

    # Step 1: Voice Input Normalization
    if not farmer_query and audio_path:
        transcribed = transcribe_audio(audio_path)
        if not transcribed:
            out["ok"] = False
            out["error"] = "Audio transcription failed. Please check microphone input and speak clearly."
            return out
        farmer_query = transcribed
        out["query"] = farmer_query

    if not farmer_query:
        out["ok"] = False
        out["error"] = "No input provided."
        return out

    # Step 2: Conversational Anaphora Resolution (T19)
    resolved_query = resolve_conversational_coreference(farmer_query, session_history or [], diag_state)
    
    # Step 3: Policy & Safety Router (Tier 1 to Tier 4)
    # Check policy on actual user input to avoid false triggers from coreference expansions
    policy_status, policy_key = route_query_policy(farmer_query)
    if policy_status == "POLICY_INTERCEPT" and policy_key:
        rendered = render_policy_response(policy_key)
        out["response"] = rendered["message"]
        out["debug"] = _format_debug(farmer_query, f"Policy Gate: {policy_key}", [], diag_state)
        return out

    # -------------------------------------------------------------------------
    # Step 4: Linguistic Intent & Telemetry Extraction (Multi-Turn Safe)
    # -------------------------------------------------------------------------
    # 1. Parse the RAW farmer input to extract telemetry belonging ONLY to this turn.
    #    (Using resolved_query here causes historical numbers like 1.8 and 7.5 to re-enter!)
    search_prompt, fresh_turn_state = extract_hypothesis(farmer_query, session_state=copy.deepcopy(diag_state))
    fresh_telemetry = fresh_turn_state.get("telemetry", {})

    # 2. Check if the farmer actually provided new readings in this turn
    #    Compare against what was previously stored in diag_state
    old_telemetry = diag_state.get("telemetry", {})
    has_fresh_telemetry = bool(fresh_telemetry and fresh_telemetry != old_telemetry)

    # 3. Update the state
    updated_state = fresh_turn_state
    if has_fresh_telemetry:
        # User supplied fresh readings: clear any old conflict lists and update with new readings
        updated_state["telemetry"] = fresh_telemetry
        updated_state["fresh_telemetry_in_turn"] = True
    else:
        updated_state["fresh_telemetry_in_turn"] = False

    species = updated_state.get("species")
    out["state"] = updated_state

   # -------------------------------------------------------------------------
    # Step 5: Telemetry Health, Conflict & Emergency Gate (T4, T11, T14)
    # -------------------------------------------------------------------------
    is_critical, decision_flag, detail = evaluate_telemetry_health_and_conflicts(updated_state)
    if is_critical and decision_flag:
        # Only fire acute alarm if current turn provided a fresh lethal reading
        if decision_flag == "EMERGENCY_HYPOXIA" and has_fresh_telemetry:
            do_data = updated_state.get("telemetry", {}).get("DO", [])
            do_val = do_data[-1].get("value", 2.0) if (isinstance(do_data, list) and do_data and isinstance(do_data[-1], dict)) else 2.0
            organism_en = "Fish are" if species == "fish" else ("Shrimp are" if species == "shrimp" else "Pond stock is")
            organism_te = "చేపలు" if species == "fish" else ("రొయ్యలు" if species == "shrimp" else "జీవులు")

            msg_en = (
                f"🚨 **CRITICAL EMERGENCY: Dissolved Oxygen is {do_val} mg/L.**\n"
                f"{organism_en} in acute respiratory distress (hypoxia suffocation). "
                f"Turn on all paddlewheel aerators immediately and halt all feeding."
            )
            msg_te = (
                f"🚨 **అత్యవసర పరిస్థితి: ఆక్సిజన్ స్థాయి {do_val} mg/L కు పడిపోయింది.**\n"
                f"{organism_te} ఊపిరాడక ఇబ్బంది పడుతున్నాయి. వెంటనే అన్ని ఏరేటర్లను ఆన్ చేయండి మరియు మేత ఇవ్వడం ఆపండి."
            )
            out["response"] = f"{msg_en}\n\n---\n{msg_te}"
            out["toast"] = "Lethal Hypoxia Detected!"
            out["debug"] = _format_debug(farmer_query, search_prompt, [], updated_state)
            return out

        elif decision_flag in ("CONFLICT_READINGS", "SENSOR_CONFLICT"):
            rendered = render_policy_response("CONFLICT_READINGS")
            out["response"] = f"**{detail}**\n\n{rendered['message']}"
            out["debug"] = _format_debug(farmer_query, f"Conflict: {detail}", [], updated_state)
            return out
    
    # -------------------------------------------------------------------------
    # Step 5b: Optimal Telemetry Confirmation Fast-Path (Fixes T11 Turn 2 Abstention)
    # -------------------------------------------------------------------------
    telemetry = updated_state.get("telemetry", {})
    symptoms = updated_state.get("symptoms", set())
    has_symptoms = bool(symptoms)

    do_records = telemetry.get("DO", [])
    if not isinstance(do_records, list):
        do_records = [do_records]
    valid_do = [r["value"] for r in do_records if isinstance(r, dict) and r.get("value") is not None]

    # If Dissolved Oxygen is optimal (>= 5.0 mg/L) with zero pathological signs, confirm immediately
    if valid_do and min(valid_do) >= 5.0 and not has_symptoms and not is_critical:
        do_val = valid_do[-1]
        species_target = species or "finfish and shrimp"
        organism_te = "చేపలు మరియు రొయ్యల" if not species else ("చేపల" if species == "fish" else "రొయ్యల")

        msg_en = (
            f"✅ **OPTIMAL PARAMETER CONFIRMED: Dissolved Oxygen is {do_val} mg/L.**\n\n"
            f"• **Status:** The reported DO level is in the safe, optimal range (> 5.0 mg/L) for {species_target} grow-out ponds.\n"
            f"• **Operational Protocol:** Continue routine water quality monitoring. Maintain regular aerator schedules during night and early morning hours when biological oxygen demand peaks."
        )
        msg_te = (
            f"✅ **సరైన కొలత నిర్ధారించబడింది: కరిగిన ఆక్సిజన్ (DO) {do_val} mg/L.**\n\n"
            f"• **స్థితి:** ఈ ఆక్సిజన్ స్థాయి {organism_te} చెరువులకు అత్యంత అనుకూలమైన మరియు సురక్షితమైన పరిధిలో (> 5.0 mg/L) ఉంది.\n"
            f"• **సాంకేతిక సలహా:** సాధారణ రోజువారీ పర్యవేక్షణను కొనసాగించండి. ఆక్సిజన్ వినియోగం ఎక్కువగా ఉండే రాత్రి మరియు తెల్లవారుజాము వేళల్లో షెడ్యూల్ ప్రకారం ఏరేటర్లను నిర్వహించండి."
        )

        out["response"] = f"{msg_en}\n\n---\n{msg_te}"
        out["report"] = (
            "=== AquaAssist Diagnostic Log ===\n"
            f"Query: {farmer_query}\n"
            f"Target Organism: {species_target}\n"
            f"Parameters: DO = {do_val} mg/L (Safe/Optimal)\n\n"
            f"{msg_en}\n\n{msg_te}"
        )
        out["debug"] = _format_debug(farmer_query, f"Fast-Path: Optimal DO Confirmation ({do_val} mg/L)", [], updated_state)
        return out

    # Step 6: ChromaDB Retrieval & Grounded NLI Verification
    embedding_model, nli_model, collection = get_diagnostic_engine()
    # Pass target_parameter so retrieval isolation activates
    active_param = "DO" if "DO" in updated_state.get("telemetry", {}) else None
    
    diag_result = run_diagnostic(
        search_prompt, 
        updated_state, 
        embedding_model, 
        nli_model, 
        collection,
        target_parameter=active_param
    )
    decision = diag_result["decision"]
    evidence = diag_result["evidence"]

    # Step 7: Response Routing based on Evidence Verification
    if decision == "ABSTAIN":
        rendered = render_policy_response("UNINDEXED_CONDITION")
        out["response"] = rendered["message"]
        out["debug"] = _format_debug(farmer_query, search_prompt, evidence, updated_state)
        return out
    elif decision == "CLARIFY":
        rendered = render_policy_response("VAGUE_DIAGNOSTIC")
        out["response"] = rendered["message"]
        out["debug"] = _format_debug(farmer_query, search_prompt, evidence, updated_state)
        return out

    # Step 8: Dynamic Grounded Generation (Phase 4)
    answer_dict = generate_answer(resolved_query, evidence, species=species)
    english_text = answer_dict["english"]
    telugu_text = answer_dict["telugu"]

    out["response"] = (
        f"**Technical Observation:**\n{english_text}\n\n"
        f"---\n**సాంకేతిక సలహా (Telugu):**\n{telugu_text}"
    )

    source_doc, page_num = "Aquaculture Manual", "N/A"
    if evidence:
        top_meta = evidence[0].get("metadata", {})
        source_doc = top_meta.get("source", "Aquaculture Manual")
        page_num = top_meta.get("page", "N/A")
        out["citation"] = f"> 📚 **Source:** {source_doc}, Page {page_num} (Distance: {evidence[0]['distance']:.3f})"

    out["report"] = (
        "=== AquaAssist Diagnostic Log ===\n"
        f"Query: {farmer_query}\n"
        f"Target Organism: {species}\n"
        f"Parameters: {updated_state.get('telemetry')}\n\n"
        "--- English Action ---\n"
        f"{english_text}\n\n"
        "--- Telugu Action ---\n"
        f"{telugu_text}\n\n"
        f"Reference: {source_doc} (pg. {page_num})"
    )

    out["debug"] = _format_debug(farmer_query, search_prompt, evidence, updated_state)
    return out


# --- 5. UI CALLBACK HANDLERS ---
def _session_label(session: Dict[str, Any]) -> str:
    title = session["title"]
    return title if len(title) < 28 else title[:25] + "..."


def _get_session(sessions: List[Dict[str, Any]], session_id: int) -> Dict[str, Any]:
    return next((s for s in sessions if s["id"] == session_id), sessions[0])


def _radio_update(sessions: List[Dict[str, Any]], current_id: int):
    return gr.update(choices=[(_session_label(s), s["id"]) for s in sessions], value=current_id)


def _write_report(report_content: str) -> str:
    report_dir = tempfile.mkdtemp(dir=TTS_DIR)
    report_path = os.path.join(report_dir, "AquaAssist_Diagnostic_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return report_path


def handle_submit(text, audio_path, sessions, current_id, diag_state):
    unchanged = (gr.update(),) * 10
    text = (text or "").strip()

    if text:
        # Typed Telugu and English are both accepted directly
        query, audio_for_pipeline = text, None
    elif audio_path:
        query, audio_for_pipeline = None, audio_path
    else:
        gr.Warning("Describe the pond symptoms or record your voice first.")
        return unchanged

    session = _get_session(sessions, current_id)
    history = list(session.get("messages", []))

    # Pass conversation history so anaphora and pronouns can be resolved
    result = run_aquaassist_pipeline(query, audio_for_pipeline, diag_state, session_history=history)

    if not result["ok"]:
        gr.Warning(result["error"])
        return unchanged
    if result["toast"]:
        gr.Warning(result["toast"])

    farmer_query = result["query"]
    if session["title"] == "New Chat":
        session["title"] = farmer_query

    audio_out = synthesize_for_ui(result["response"])
    display = result["response"] + (f"\n\n{result['citation']}" if result["citation"] else "")

    session["messages"].append({"role": "user", "content": farmer_query})
    session["messages"].append({"role": "assistant", "content": display})
    session["diagnostic_state"] = copy.deepcopy(result["state"])

    export_update = (
        gr.update(value=_write_report(result["report"]), visible=True)
        if result["report"] else gr.update(value=None, visible=False)
    )

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
    sessions.insert(0, {
        "id": new_id,
        "title": "New Chat",
        "messages": [],
        "diagnostic_state": copy.deepcopy(fresh_state)
    })
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
    success, msg = download_and_apply_ota_update(DB_MAGIC_URL)
    if not success:
        gr.Warning(msg)
        return f"❌ {msg}"
    try:
        get_diagnostic_engine(force_reload=True)
    except Exception as e:
        gr.Warning(f"Knowledge base updated, but engine reload failed: {e}")
        return f"⚠️ Knowledge base updated, but engine reload failed: {e}"
    gr.Info(msg)
    return f"✅ {msg}"


# --- 6. USER INTERFACE (GRADIO BLOCKS) ---
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
        gr.Markdown("🌐 Mode: Edge Neuro-Symbolic (Offline Ready)")
        ota_btn = gr.Button("🔄 Check Knowledge Updates")
        ota_status = gr.Markdown("")

    gr.Markdown("# 🌊 AquaAssist: Offline Aquaculture Diagnostic Assistant")
    gr.Markdown("Grounded Limnological Diagnosis & Decision Support — Telugu (తెలుగు) & English")

    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(type="messages", label="Diagnostic Consultation", height=460)
            tts_out = gr.Audio(label="Spoken Response (వాణి సలహా)", autoplay=True, interactive=False, type="filepath")
            with gr.Row():
                text_in = gr.Textbox(placeholder="Describe pond symptoms, parameters (e.g. DO, pH), or ask questions in English or Telugu...", label="Query / ప్రశ్న", lines=2, scale=3)
                audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Record / Upload Voice (Telugu or English)", scale=2)
            send_btn = gr.Button("Submit Analysis / విశ్లేషించండి", variant="primary")
            export_btn = gr.DownloadButton("📄 Export Diagnostic Log", visible=False)

        with gr.Column(scale=5):
            with gr.Accordion("🔍 Engineering Verification & NLI Traces", open=True):
                debug_box = gr.Textbox(label="Pipeline Trace", lines=24, interactive=False)

    conversation_outputs = [chatbot, sessions_state, current_session_id, diagnostic_state, text_in, audio_in, tts_out, export_btn, debug_box, session_radio]
    submit_inputs = [text_in, audio_in, sessions_state, current_session_id, diagnostic_state]

    send_btn.click(handle_submit, submit_inputs, conversation_outputs)
    text_in.submit(handle_submit, submit_inputs, conversation_outputs)
    new_chat_btn.click(handle_new_chat, [sessions_state, current_session_id, diagnostic_state], conversation_outputs)
    session_radio.input(handle_select_session, [session_radio, sessions_state, current_session_id, diagnostic_state], conversation_outputs)
    ota_btn.click(handle_ota_update, None, ota_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[os.path.abspath(TTS_DIR)])