"""
TTS Module — Text-to-Speech response generator
"""

from gtts import gTTS
import tempfile
import base64
import os


def generate_tts_base64(text: str) -> str:
    """
    Convert text to speech and return as base64 encoded MP3.
    Returns empty string on failure (non-critical feature).
    """
    filename = None
    try:
        tts = gTTS(text=text, lang="en")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            filename = f.name

        tts.save(filename)

        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        return b64

    except Exception:
        return ""

    finally:
        if filename and os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass
