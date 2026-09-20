import streamlit as st

from response_generator import generate_answer
from diagnostic_engine import load_engine, run_diagnostic
from intent_extractor import extract_hypothesis
from decision_response import get_decision_response
from whisper_transcriber import transcribe_audio


st.set_page_config(
    page_title="AquaAssist",
    page_icon="🐟",
    layout="wide"
)


@st.cache_resource
def get_diagnostic_engine():
    return load_engine()


st.title("🐟 AquaAssist")
st.subheader("Evidence-Aware Aquaculture Assistant")

st.write(
    "Enter a farmer's observation below to begin the diagnostic process."
)

st.subheader("Input")

input_mode = st.radio(
    "Choose input method",
    ["Text", "Audio"],
    horizontal=True
)

farmer_query = ""

if input_mode == "Text":

    farmer_query = st.text_area(
        "Farmer Query",
        placeholder=(
            "Example: My shrimp are not eating and the pond water "
            "has become dark green."
        ),
        height=120
    )

else:

    audio_file = st.file_uploader(
        "Upload farmer audio",
        type=["wav", "mp3", "m4a", "ogg"]
    )

    if audio_file is not None:

        audio_path = f"temp_farmer_audio.{audio_file.name.split('.')[-1]}"

        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        with st.spinner("Transcribing audio locally..."):
            farmer_query = transcribe_audio(audio_path)

        st.write("**Transcription:**")
        st.write(farmer_query)


if st.button("🔬 Run Diagnosis"):

    if not farmer_query.strip():
        st.warning("Please enter a farmer query.")

    else:

        with st.spinner("Running diagnostic pipeline..."):

            # Step 1: Convert farmer query into a declarative observation
            hypothesis = extract_hypothesis(farmer_query)

            # Step 2: Load retrieval + NLI engine
            embedding_model, nli_model, collection = (
                get_diagnostic_engine()
            )

            # Step 3: Retrieve evidence and run NLI verification
            result = run_diagnostic(
                hypothesis,
                embedding_model,
                nli_model,
                collection
            )

        # ---------------------------------------------------------
        # Extracted observation
        # ---------------------------------------------------------

        st.divider()

        st.subheader("🧠 Extracted Observation")
        st.write(hypothesis)

        # ---------------------------------------------------------
        # Retrieved evidence
        # ---------------------------------------------------------

        st.subheader("📚 Retrieved Evidence")

        for i, chunk in enumerate(result["evidence"], 1):

            with st.expander(
                f"Evidence {i} — {chunk['id']}"
            ):

                st.write(chunk["text"])

                st.caption(
                    f"Topic: {chunk['metadata'].get('topic')} | "
                    f"Species: {chunk['metadata'].get('species')}"
                )

                st.write(
                    f"Retrieval distance: "
                    f"{chunk['distance']:.4f}"
                )

        # ---------------------------------------------------------
        # NLI verification
        # ---------------------------------------------------------

        st.subheader("🔬 NLI Verification")

        for chunk in result["evidence"]:

            st.write(f"**{chunk['id']}**")

            st.write(
                f"Contradiction: {chunk['contradiction']:.4f} | "
                f"Entailment: {chunk['entailment']:.4f} | "
                f"Neutral: {chunk['neutral']:.4f}"
            )

        # ---------------------------------------------------------
        # Decision
        # ---------------------------------------------------------

        decision = result["decision"]

        st.subheader("🚦 Decision")

        if decision == "ANSWER":

            st.success("ANSWER")

            with st.spinner(
                "Generating evidence-grounded Telugu response..."
            ):

                final_response = generate_answer(
                    hypothesis,
                    result["evidence"]
                )

            st.subheader("🗣️ AquaAssist Response")

            st.code(
                final_response,
                language="json"
            )

        elif decision == "CLARIFY":

            st.warning("CLARIFY")

            response = get_decision_response("CLARIFY")

            st.write(response["message"])

        else:

            st.error("ABSTAIN")

            response = get_decision_response("ABSTAIN")

            st.write(response["message"])