import sys, os
from pathlib import Path
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
import pickle
import base64

# Add parent directory to path to allow importing Backend_app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from Backend_app.tts_module import generate_tts_base64
from Backend_app.enhanced_detector import preprocess_audio, extract_features

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def get_segments(y, sr, segment_len_sec=10):
    segment_samples = int(segment_len_sec * sr)
    segments = []
    total_len = len(y)
    
    start = 0
    while start + segment_samples <= total_len:
        seg = y[start:start + segment_samples]
        if np.max(np.abs(seg)) > 0.05:
            segments.append(seg)
        start += segment_samples
    return segments

def main():
    X = []
    y_labels = []

    # 1. Humans (LibriSpeech examples)
    print("Extracting human speech features (LibriSpeech)...")
    libri_list = ["libri1", "libri2", "libri3"]
    for libri in libri_list:
        try:
            y, sr = librosa.load(librosa.example(libri), sr=16000)
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            segs = get_segments(y_trimmed, 16000, 10)
            print(f"  {libri}: found {len(segs)} segments")
            for seg in segs:
                y_pre, _, _ = preprocess_audio(seg, 16000)
                feat = extract_features(y_pre, 16000)["vector"]
                X.append(feat)
                y_labels.append("HUMAN_GENERATED")
        except Exception as e:
            print(f"  Error processing {libri}: {e}")

    # 2. User WhatsApp HUMAN (80s clip)
    print("\nExtracting user WhatsApp HUMAN speech features (80s clip)...")
    fp_human = r"C:\Users\Lenovo\Downloads\WhatsApp Audio 2026-06-23 at 2.38.04 PM.mpeg.wav"
    if os.path.exists(fp_human):
        y, sr = librosa.load(fp_human, sr=16000)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        segs = get_segments(y_trimmed, 16000, 10)
        print(f"  {os.path.basename(fp_human)}: found {len(segs)} segments")
        for seg in segs:
            y_pre, _, _ = preprocess_audio(seg, 16000)
            feat = extract_features(y_pre, 16000)["vector"]
            X.append(feat)
            y_labels.append("HUMAN_GENERATED")

    # 3. User WhatsApp AI (22s clip)
    print("\nExtracting user WhatsApp AI speech features (22s clip)...")
    fp_ai = r"C:\Users\Lenovo\Downloads\WhatsApp Audio 2026-06-23 at 2.38.04 PM (1).mp3"
    if os.path.exists(fp_ai):
        y, sr = librosa.load(fp_ai, sr=16000)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        segs = get_segments(y_trimmed, 16000, 10)
        print(f"  {os.path.basename(fp_ai)}: found {len(segs)} segments")
        for seg in segs:
            y_pre, _, _ = preprocess_audio(seg, 16000)
            feat = extract_features(y_pre, 16000)["vector"]
            X.append(feat)
            y_labels.append("AI_GENERATED")

    # 4. AI (gTTS)
    print("\nGenerating and extracting gTTS AI speech features...")
    gtts_texts = [
        "Hello, this is a voice security detection test.",
        "Artificial intelligence is changing the way we interact with technology.",
        "Deepfake audio can be used for malicious purposes if not detected.",
        "Our system analyzes mel frequency cepstral coefficients to classify speech.",
        "Thank you for using the automatic voice authenticating system."
    ]
    for idx, text in enumerate(gtts_texts):
        try:
            b64 = generate_tts_base64(text)
            if b64:
                fn = f"temp_gtts_{idx}.mp3"
                with open(fn, "wb") as f:
                    f.write(base64.b64decode(b64))
                y, sr = librosa.load(fn, sr=16000)
                y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                segs = get_segments(y_trimmed, 16000, 10)
                if not segs and len(y_trimmed) > 16000 * 2:
                     segs = [y_trimmed]
                for seg in segs:
                    y_pre, _, _ = preprocess_audio(seg, 16000)
                    feat = extract_features(y_pre, 16000)["vector"]
                    X.append(feat)
                    y_labels.append("AI_GENERATED")
                os.remove(fn)
        except Exception as e:
             print(f"  Error generating gTTS {idx}: {e}")

    X = np.array(X)
    y_labels = np.array(y_labels)

    print(f"\nDataset summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Human samples: {np.sum(y_labels == 'HUMAN_GENERATED')}")
    print(f"  AI samples   : {np.sum(y_labels == 'AI_GENERATED')}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define models
    svm = SVC(kernel="rbf", probability=True, C=10, gamma="scale")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    ensemble = VotingClassifier(
        estimators=[
            ("svm", svm),
            ("rf", rf),
            ("gb", gb),
        ],
        voting="soft",
        weights=[1, 1.5, 1.5],
    )

    loo = LeaveOneOut()
    scores = cross_val_score(ensemble, X_scaled, y_labels, cv=loo, n_jobs=-1)
    print(f"\nModel Performance (Leave-One-Out CV): {np.mean(scores):.2%}")

    ensemble.fit(X_scaled, y_labels)

    MODEL_PATH = MODEL_DIR / "ensemble_model.pkl"
    SCALER_PATH = MODEL_DIR / "scaler.pkl"

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ensemble, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSuccessfully trained and saved model files to {MODEL_DIR}!")

if __name__ == "__main__":
    main()
