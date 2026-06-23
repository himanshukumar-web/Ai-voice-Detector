"""
AI Voice Detector — FastAPI Backend
=====================================
Endpoints:
  POST /analyze    — Analyze a single audio file
  POST /batch      — Analyze multiple audio files
  GET  /history    — Get analysis history
  GET  /history/{id} — Get specific analysis result
  DELETE /history  — Clear history
  GET  /health     — Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import base64
import uuid
from datetime import datetime

from Backend_app.enhanced_detector import detect_voice
from Backend_app.tts_module import generate_tts_base64


# ─── App Setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Voice Detection System",
    description="Detect AI-generated vs human voice using ensemble ML and advanced audio features",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For deployment, we allow all origins. You can restrict this later to your Vercel URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── In-memory history store ────────────────────────────────────────────
analysis_history: list[dict] = []


# ─── Request / Response Models ──────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    filename: Optional[str] = Field(None, description="Original filename")

class BatchRequest(BaseModel):
    files: list[AnalyzeRequest] = Field(..., description="List of audio files to analyze")

class FeatureResponse(BaseModel):
    pitch_mean: float = 0
    pitch_std: float = 0
    pitch_range: float = 0
    spectral_centroid: float = 0
    spectral_flatness: float = 0
    spectral_bandwidth: float = 0
    spectral_rolloff: float = 0
    zcr: float = 0
    rms_energy: float = 0
    rms_std: float = 0
    voiced_ratio: float = 0
    silence_ratio: float = 0
    duration: float = 0
    tempo: float = 0
    mfcc_1_mean: float = 0
    mfcc_2_mean: float = 0
    mfcc_3_mean: float = 0
    mfcc_variability: float = 0
    delta_smoothness: float = 0
    # AI-discriminative features
    pitch_jitter: float = 0
    pitch_entropy: float = 0
    harmonic_ratio: float = 0
    rms_cv: float = 0
    spectral_flux: float = 0
    mfcc_temporal_change: float = 0
    spectral_crest_factor: float = 0

class AnalyzeResponse(BaseModel):
    id: str
    timestamp: str
    filename: Optional[str]
    classification: str
    confidence: float
    language_detected: str
    explanation: str
    features: FeatureResponse
    waveform: list[float]
    audio_response_base64: Optional[str] = None


# ─── Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "status": "running",
        "service": "AI Voice Detection System v2.0",
        "endpoints": ["/analyze", "/batch", "/history", "/health"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "history_count": len(analysis_history),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Analyze a single audio file for AI voice detection."""

    # Validate base64
    try:
        base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    # Run detection
    try:
        result = detect_voice(req.audio_base64)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {str(e)}")

    # Generate TTS response
    audio_response_b64 = None
    try:
        verdict_text = (
            f"This voice is classified as "
            f"{'AI generated' if result['classification'] == 'AI_GENERATED' else 'human generated'}. "
            f"Confidence is {int(result['confidence'] * 100)} percent. "
            f"Language detected: {result['language_detected']}."
        )
        audio_response_b64 = generate_tts_base64(verdict_text)
    except Exception:
        pass  # TTS is non-critical

    # Build response
    entry_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()

    response = {
        "id": entry_id,
        "timestamp": timestamp,
        "filename": req.filename,
        "classification": result["classification"],
        "confidence": result["confidence"],
        "language_detected": result["language_detected"],
        "explanation": result["explanation"],
        "features": result["features"],
        "waveform": result["waveform"],
        "audio_response_base64": audio_response_b64,
    }

    # Store in history
    analysis_history.insert(0, response)

    # Keep max 50 entries
    if len(analysis_history) > 50:
        analysis_history.pop()

    return response


@app.post("/analyze-stream")
def analyze_stream(req: AnalyzeRequest):
    """Lightweight endpoint for real-time stream analysis. No TTS, no history."""

    try:
        base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    try:
        result = detect_voice(req.audio_base64)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {str(e)}")

    return {
        "classification": result["classification"],
        "confidence": result["confidence"],
        "language_detected": result["language_detected"],
        "explanation": result["explanation"],
    }


@app.post("/batch")
def batch_analyze(req: BatchRequest):
    """Analyze multiple audio files in batch."""

    if len(req.files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    results = []
    for i, file_req in enumerate(req.files):
        try:
            result = detect_voice(file_req.audio_base64)
            entry_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().isoformat()

            entry = {
                "id": entry_id,
                "timestamp": timestamp,
                "filename": file_req.filename or f"file_{i+1}",
                "classification": result["classification"],
                "confidence": result["confidence"],
                "language_detected": result["language_detected"],
                "explanation": result["explanation"],
                "features": result["features"],
                "waveform": result["waveform"],
                "audio_response_base64": None,
            }
            results.append(entry)
            analysis_history.insert(0, entry)

        except Exception as e:
            results.append({
                "id": None,
                "filename": file_req.filename or f"file_{i+1}",
                "error": str(e),
            })

    # Keep max 50 entries
    while len(analysis_history) > 50:
        analysis_history.pop()

    return {
        "total": len(req.files),
        "successful": sum(1 for r in results if r.get("id")),
        "results": results,
    }


@app.get("/history")
def get_history():
    """Get all analysis history."""
    return {
        "count": len(analysis_history),
        "results": analysis_history,
    }


@app.get("/history/{entry_id}")
def get_history_entry(entry_id: str):
    """Get a specific analysis result by ID."""
    for entry in analysis_history:
        if entry.get("id") == entry_id:
            return entry
    raise HTTPException(status_code=404, detail="Entry not found")


@app.delete("/history")
def clear_history():
    """Clear all analysis history."""
    analysis_history.clear()
    return {"status": "cleared", "count": 0}
