import base64
import io
import numpy as np
import soundfile as sf
import librosa
import binascii
import os


def detect_audio_input_type(input_value: str) -> str:
    """
    Returns one of:
    - 'wav_path'
    - 'base64_mp3'
    - 'base64_wav'
    Raises ValueError if unknown.
    """

    # 1️⃣ File path check (WAV on disk)
    if os.path.isfile(input_value):
        return "wav_path"

    # 2️⃣ Try Base64 decode
    try:
        audio_bytes = base64.b64decode(input_value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Input is neither a file path nor valid Base64")

    # 3️⃣ Inspect decoded bytes
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:12]:
        return "base64_wav"

    if (
        audio_bytes.startswith(b"ID3")
        or audio_bytes[:2] in [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"]
    ):
        return "base64_mp3"

    raise ValueError("Unsupported Base64 audio format")





def preprocess_audio_base64(
    audio_base64: str,
    target_sr: int = 16000
):
    """
    Compliant preprocessing:
    - Decodes Base64 MP3
    - Converts to waveform
    - Converts to mono if needed
    - Resamples to 16kHz
    DOES NOT modify audio content
    """

    # 1. Base64 → bytes
    audio_bytes = base64.b64decode(audio_base64)

    # 2. Bytes → audio waveform
    audio_buffer = io.BytesIO(audio_bytes)
    waveform, sr = librosa.load(audio_buffer, sr=target_sr, mono=True)



    # 5. Ensure float32 (model requirement)
    waveform = waveform.astype(np.float32)

    return waveform, target_sr




def preprocess_audio_wav(
    wav_path: str,
    target_sr: int = 16000
):
    """
    Preprocess WAV input:
    - Loads WAV from disk
    - Converts to mono if needed
    - Resamples to 16kHz
    DOES NOT modify audio content
    """

    # 1. Load WAV
    waveform, sr = sf.read(wav_path)

    # 2. Stereo → mono
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)

    # 3. Resample
    if sr != target_sr:
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=target_sr
        )

    return waveform.astype(np.float32), target_sr

def preprocess_audio_auto(input_value: str):
    input_type = detect_audio_input_type(input_value)

    if input_type == "wav_path":
        return preprocess_audio_wav(input_value)

    if input_type == "base64_mp3":
        return preprocess_audio_base64(input_value)

    if input_type == "base64_wav":
        # optional: if you ever allow this
        return preprocess_audio_base64(input_value)

    raise ValueError("Unsupported input type")
