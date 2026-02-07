import React, { useRef, useState } from "react";
import "./App.css";

export default function VoiceAuthUI() {

    const fileRef = useRef();

    const [file, setFile] = useState(null);
    const [fileName, setFileName] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const [history, setHistory] = useState([]);

    const handleFile = (e) => {
        const f = e.target.files[0];
        if (!f) return;

        setFile(f);
        setFileName(f.name);
    };

    const analyze = async () => {

        if (!file) {
            alert("Please select a voice file first");
            return;
        }

        setLoading(true);

        const base64 = await toBase64(file);

        const res = await fetch(`${process.env.REACT_APP_API_URL}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                audio_base64: base64.split(",")[1]
            })
        });

        const data = await res.json();

        setResult(data);

        setHistory((prev) => [
            {
                id: Date.now(),
                name: fileName,
                data
            },
            ...prev
        ]);

        setLoading(false);
    };

    return (
        <div className="voice-ui-root">

            {/* LEFT PANEL */}
            <div className="voice-left-panel">
                <h3>Recent Analysis</h3>

                {history.length === 0 && (
                    <div className="recent-item">No history</div>
                )}

                {history.map((item, i) => (
                    <div
                        key={item.id}
                        className="recent-item"
                        onClick={() => setResult(item.data)}
                    >
                        Voice Check #{history.length - i}
                        <div style={{ fontSize: 11, opacity: 0.6 }}>
                            {item.name}
                        </div>
                    </div>
                ))}
            </div>

            {/* CENTER */}
            <div className="voice-center">

                <div className="ai-orb">
                    <div className="ai-orb-line"></div>
                </div>

                <div className="ready-text">
                    {loading ? "ANALYSING..." : "READY FOR ANALYSIS"}
                </div>

                <div
                    className="drop-box"
                    onClick={() => fileRef.current.click()}
                    style={{
                        borderColor: file ? "#5af6ff" : ""
                    }}
                >
                    <div className="mic-icon">🎤</div>

                    {!file && <div>Drop voice file or click to upload</div>}

                    {file && (
                        <div style={{ fontSize: 13, color: "#5af6ff" }}>
                            Selected: {fileName}
                        </div>
                    )}

                    <small>MP3, WAV, M4A</small>

                    <input
                        ref={fileRef}
                        type="file"
                        hidden
                        accept="audio/*"
                        onChange={handleFile}
                    />
                </div>

                <button
                    className="analyze-btn"
                    onClick={analyze}
                    disabled={loading}
                >
                    ANALYZE VOICE
                </button>

            </div>

            {/* RIGHT PANEL */}
            <div className="voice-right-panel">

                {!result && <div className="placeholder">No result yet</div>}

                {result && (
                    <>
                        <h3
                            className={
                                result.classification === "AI_GENERATED"
                                    ? "ai-text"
                                    : "human-text"
                            }
                        >
                            {result.classification}
                        </h3>

                        <div className="circle-score">
                            {Math.round(result.confidence * 100)}%
                        </div>

                        <p>
                            Language:
                            {" "}
                            {result.language_detected || "Not detected"}
                        </p>

                        <p className="explain">{result.explanation}</p>

                        {result.audio_response_base64 && (
                            <audio
                                controls
                                src={`data:audio/mp3;base64,${result.audio_response_base64}`}
                            />
                        )}
                    </>
                )}

            </div>

        </div>
    );
}

function toBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}
