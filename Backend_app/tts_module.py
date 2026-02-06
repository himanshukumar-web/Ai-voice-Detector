from gtts import gTTS
import tempfile
import base64
import os

def generate_tts_base64(text):

    tts = gTTS(text=text, lang="en")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        filename = f.name

    tts.save(filename)

    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    os.remove(filename)

    return b64
