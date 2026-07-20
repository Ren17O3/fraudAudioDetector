import base64
import requests
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------
API_URL = "http://0.0.0.0:8000/api/voice-detection"

st.set_page_config(
    page_title="Fraud Audio Detector",
    page_icon="🎙️",
    layout="centered"
)

# -----------------------------
# Session State
# -----------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "error" not in st.session_state:
    st.session_state.error = None


# -----------------------------
# Helper Functions
# -----------------------------
def clear_results():
    """Clear previous prediction whenever a new file is uploaded."""
    st.session_state.result = None
    st.session_state.error = None


# -----------------------------
# UI
# -----------------------------
st.title("🎙️ Fraud Audio Detector")
st.write("Detect whether an uploaded audio recording is Human or AI Generated.")

api_key = st.text_input(
    "API Key",
    type="password"
)

language = st.selectbox(
    "Language",
    [
        "English",
        "Hindi",
        "Tamil",
        "Malayalam",
        "Telugu"
    ]
)

audio = st.file_uploader(
    "Upload Audio",
    type=["mp3"],
    accept_multiple_files=False,
    key="audio_file",
    on_change=clear_results
)

if audio is not None:
    st.audio(audio)

# -----------------------------
# Detect
# -----------------------------
if st.button("Detect", use_container_width=True):

    # Clear previous output before every prediction
    clear_results()

    if not api_key:
        st.session_state.error = "Please enter your API Key."

    elif audio is None:
        st.session_state.error = "Please upload an audio file."

    else:

        audio_bytes = audio.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": audio_b64
        }

        headers = {
            "x-api-key": api_key
        }

        try:

            with st.spinner("Analyzing audio..."):

                response = requests.post(
                    API_URL,
                    json=payload,
                    headers=headers,
                    timeout=60
                )

            if response.status_code == 200:
                st.session_state.result = response.json()

            else:
                try:
                    err = response.json()

                    st.session_state.error = (
                        err.get("detail")
                        or err.get("message")
                        or "Request failed."
                    )

                except Exception:
                    st.session_state.error = (
                        f"HTTP {response.status_code}"
                    )

        except requests.exceptions.ConnectionError:
            st.session_state.error = "Unable to connect to the FastAPI server."

        except requests.exceptions.Timeout:
            st.session_state.error = "Request timed out."


# -----------------------------
# Display Error
# -----------------------------
if st.session_state.error:
    st.error(st.session_state.error)


# -----------------------------
# Display Result
# -----------------------------
if st.session_state.result:

    result = st.session_state.result

    st.success("Analysis Complete")

    st.divider()

    st.subheader("Classification")

    if result["classification"] == "HUMAN":
        st.success("✅ HUMAN")
    else:
        st.error("⚠️ AI GENERATED")

    st.subheader("Confidence Score")

    confidence = result["confidenceScore"]

    st.progress(confidence)

    st.metric(
        "Confidence",
        f"{confidence:.2%}"
    )