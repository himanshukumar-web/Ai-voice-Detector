import numpy as np
import librosa
import soundfile as sf
import io
import base64


def detect_language_simple(y, sr):
    """
    Lightweight heuristic language identification
    (Only for hackathon demo – stable, fast, no ML dependency)
    """

    # Formant proxy (spectral envelope shape)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = np.mean(np.var(mfcc, axis=1))

    # Very rough acoustic tendencies (NOT linguistic accuracy)
    # but gives separation for demo use
    if spec_centroid < 1800 and mfcc_var > 120:
        return "Hindi"

    if spec_centroid < 2000 and zcr < 0.06:
        return "Tamil"

    if spec_centroid > 2400 and mfcc_var < 90:
        return "English"

    if zcr > 0.085:
        return "Telugu"

    return "Malayalam"


def detect_voice(audio_b64):

    audio_bytes = base64.b64decode(audio_b64)
    y, sr = sf.read(io.BytesIO(audio_bytes))

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    y = y.astype(np.float32)

    # -----------------------------
    # Language (simple & stable)
    # -----------------------------
    language = detect_language_simple(y, sr)

    # -----------------------------
    # Feature extraction
    # -----------------------------
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    f0, _, _ = librosa.pyin(y, fmin=60, fmax=350, sr=sr)
    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        pitch_std = 0.0
    else:
        pitch_std = float(np.std(f0))

    centroid_mean = float(np.mean(centroid))
    flatness_mean = float(np.mean(flatness))
    zcr_mean = float(np.mean(zcr))

    # -----------------------------
    # Conservative heuristic AI detection
    # -----------------------------

    score = 0
    reasons = []

    if pitch_std < 12:
        score += 1
        reasons.append("very smooth pitch contour")

    if flatness_mean > 0.25:
        score += 1
        reasons.append("flat spectral distribution")

    if centroid_mean > 3200:
        score += 1
        reasons.append("high frequency emphasis")

    if zcr_mean < 0.045:
        score += 1
        reasons.append("low temporal variability")

    is_ai = score >= 3

    confidence = min(0.92, 0.50 + 0.08 * score)

    if not reasons:
        reasons.append("natural pitch and spectral variations")

    return {
        "classification": "AI_GENERATED" if is_ai else "HUMAN_GENERATED",
        "confidence": round(float(confidence), 2),
        "language_detected": language,
        "explanation": "Indicators: " + ", ".join(reasons)
    }
