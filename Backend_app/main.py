from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64

from fastapi.middleware.cors import CORSMiddleware

from Backend_app.enhanced_detector import detect_voice
from Backend_app.tts_module import generate_tts_base64


# ✅ FIRST create app
app = FastAPI(title="AI Voice Detection System")


# ✅ THEN add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VoiceRequest(BaseModel):
    audio_base64: str


class VoiceResponse(BaseModel):
    classification: str
    confidence: float
    language_detected: str
    explanation: str
    audio_response_base64: str


@app.get("/")
def home():
    return {"status": "AI Voice Detection API running"}


@app.post("/analyze", response_model=VoiceResponse)
def analyze(req: VoiceRequest):

    try:
        base64.b64decode(req.audio_base64)
    except:
        raise HTTPException(status_code=400, detail="Invalid base64")

    result = detect_voice(req.audio_base64)

    verdict_text = (
        f"This voice sample is classified as "
        f"{'AI generated' if result['classification']=='AI_GENERATED' else 'human generated'}. "
        f"Detected language is {result['language_detected']}. "
        f"Confidence score is {int(result['confidence']*100)} percent."
    )

    audio_b64 = generate_tts_base64(verdict_text)

    return {
        "classification": result["classification"],
        "confidence": result["confidence"],
        "language_detected": result["language_detected"],
        "explanation": result["explanation"],
        "audio_response_base64": audio_b64
    }
