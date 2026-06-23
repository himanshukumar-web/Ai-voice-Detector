import React, { useRef, useState, useEffect, useCallback } from "react";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";
const HISTORY_KEY = "vad_history_v2";

/* ═══════════════════════════════════════════════════════════════════════
   Waveform Visualizer (Canvas-based)
   ═══════════════════════════════════════════════════════════════════════ */

function WaveformCanvas({ data, isAI }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const mid = H / 2;

    ctx.clearRect(0, 0, W, H);

    // Draw center line
    ctx.strokeStyle = "rgba(90, 246, 255, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(W, mid);
    ctx.stroke();

    const barW = Math.max(1.5, (W / data.length) - 1);
    const gap = W / data.length;

    const colorTop = isAI === true ? "rgba(248,113,113,0.9)" : isAI === false ? "rgba(52,211,153,0.9)" : "rgba(90,246,255,0.85)";
    const colorMid = isAI === true ? "rgba(251,146,60,0.6)" : isAI === false ? "rgba(34,211,238,0.6)" : "rgba(99,102,241,0.6)";

    data.forEach((val, i) => {
      const x = i * gap;
      const amplitude = Math.abs(val) * mid * 0.85;
      const gradient = ctx.createLinearGradient(x, mid - amplitude, x, mid + amplitude);
      gradient.addColorStop(0, colorTop);
      gradient.addColorStop(0.5, colorMid);
      gradient.addColorStop(1, colorTop);
      ctx.fillStyle = gradient;
      ctx.fillRect(x, mid - amplitude, barW, amplitude * 2);
    });
  }, [data, isAI]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />;
}

/* ═══════════════════════════════════════════════════════════════════════
   Confidence Meter (SVG circular gauge)
   ═══════════════════════════════════════════════════════════════════════ */

function ConfidenceMeter({ value, isAI }) {
  const radius = 58;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(1, Math.max(0, value));
  const progress = clamped * circumference;
  const offset = circumference - progress;

  const getColor = () => {
    if (isAI) {
      if (clamped > 0.8) return "#f87171";
      if (clamped > 0.6) return "#fb923c";
      return "#fbbf24";
    }
    if (clamped > 0.8) return "#34d399";
    if (clamped > 0.6) return "#22d3ee";
    return "#fbbf24";
  };

  const label = clamped >= 0.85
    ? (isAI ? "HIGH RISK" : "CONFIDENT")
    : clamped >= 0.65 ? "PROBABLE" : "UNCERTAIN";

  return (
    <div className="confidence-meter">
      <svg viewBox="0 0 140 140">
        <circle className="bg-ring" cx="70" cy="70" r={radius} />
        <circle
          className="fg-ring"
          cx="70"
          cy="70"
          r={radius}
          stroke={getColor()}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="meter-value">
        <span className="meter-number" style={{ color: getColor() }}>
          {Math.round(clamped * 100)}
        </span>
        <span className="meter-unit">%</span>
        <span className="meter-label" style={{ color: getColor() }}>{label}</span>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   Audio Player Mini — Custom HTML5 player
   ═══════════════════════════════════════════════════════════════════════ */

function AudioPlayerMini({ src, label }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  const toggle = () => {
    const a = audioRef.current;
    if (!a) return;
    if (playing) { a.pause(); setPlaying(false); }
    else { a.play().catch(() => {}); setPlaying(true); }
  };

  const onTimeUpdate = () => {
    const a = audioRef.current;
    if (a && a.duration) setProgress(a.currentTime / a.duration);
  };

  const onEnded = () => { setPlaying(false); setProgress(0); };
  const onLoadedMetadata = () => { if (audioRef.current) setDuration(audioRef.current.duration || 0); };

  // Reset when src changes
  useEffect(() => { setPlaying(false); setProgress(0); setDuration(0); }, [src]);

  const seek = (e) => {
    const a = audioRef.current;
    if (!a || !a.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    a.currentTime = ratio * a.duration;
    setProgress(ratio);
  };

  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const elapsed = progress * duration;

  return (
    <div className="audio-player-mini">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={onTimeUpdate}
        onEnded={onEnded}
        onLoadedMetadata={onLoadedMetadata}
      />
      <div className="apm-label">{label}</div>
      <div className="apm-controls">
        <button className="apm-play-btn" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
          {playing ? (
            <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>
        <div className="apm-seek" onClick={seek}>
          <div className="apm-seek-fill" style={{ width: `${progress * 100}%` }} />
        </div>
        <span className="apm-time">
          {duration > 0 ? `${fmt(elapsed)} / ${fmt(duration)}` : "--:--"}
        </span>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   Feature Bar Component
   ═══════════════════════════════════════════════════════════════════════ */

function FeatureItem({ label, value, displayValue, maxVal, aiHigh }) {
  const pct = Math.min(100, (Math.abs(value) / (maxVal || 1)) * 100);
  const barColor = aiHigh
    ? "linear-gradient(90deg, var(--orange), var(--red))"
    : "linear-gradient(90deg, var(--cyan), var(--purple))";
  return (
    <div className="feature-item">
      <div className="fi-label">{label}</div>
      <div className="fi-value">{displayValue ?? (typeof value === "number" ? value.toFixed(3) : value)}</div>
      <div className="fi-bar">
        <div className="fi-bar-fill" style={{ width: `${pct}%`, background: barColor }} />
      </div>
    </div>
  );
}

/* ── History persistence ── */
function loadHistory() {
  try { return JSON.parse(localStorage.getItem("vad_history_v2") || "[]"); }
  catch { return []; }
}
function saveHistory(hist) {
  try {
    const toSave = hist.map(({ audioUrl, ...rest }) => rest).slice(0, 30);
    localStorage.setItem("vad_history_v2", JSON.stringify(toSave));
  } catch { }
}

/* ── Language flags ── */
const LANG_FLAGS = { Hindi: "🇮🇳", Tamil: "🇮🇳", Telugu: "🇮🇳", Malayalam: "🇮🇳", English: "🇬🇧" };

/* ═══════════════════════════════════════════════════════════════════════
   Main UI Component
   ═══════════════════════════════════════════════════════════════════════ */

export default function VoiceAuthUI() {
  const fileRef = useRef(null);
  const audioUrlRef = useRef(null);

  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [currentAudioUrl, setCurrentAudioUrl] = useState(null);
  const [history, setHistory] = useState(() => loadHistory());
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const [featuresOpen, setFeaturesOpen] = useState(false);
  const [waveformData, setWaveformData] = useState(null);
  const [selectedHistoryId, setSelectedHistoryId] = useState(null);

  useEffect(() => { saveHistory(history); }, [history]);

  useEffect(() => {
    return () => { if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current); };
  }, []);

  // Auto-dismiss error
  useEffect(() => {
    if (error) { const t = setTimeout(() => setError(null), 4500); return () => clearTimeout(t); }
  }, [error]);

  // Load waveform from file via Web Audio API
  const loadWaveform = useCallback(async (audioFile) => {
    try {
      const arrayBuffer = await audioFile.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decoded = await audioCtx.decodeAudioData(arrayBuffer);
      const rawData = decoded.getChannelData(0);
      const points = 200;
      const step = Math.floor(rawData.length / points);
      const wf = [];
      for (let i = 0; i < points; i++) {
        wf.push(rawData[i * step]);
      }
      setWaveformData(wf);
      audioCtx.close();
    } catch {
      setWaveformData(null);
    }
  }, []);

  const createAudioUrl = (f) => {
    if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    const url = URL.createObjectURL(f);
    audioUrlRef.current = url;
    setCurrentAudioUrl(url);
    return url;
  };

  // File handling
  const handleFile = (f) => {
    if (!f) return;
    if (!f.type.startsWith("audio/")) {
      setError("Please select an audio file (MP3, WAV, M4A)");
      return;
    }
    if (f.size > 50 * 1024 * 1024) {
      setError("File too large. Maximum size is 50MB.");
      return;
    }
    setFile(f);
    setFileName(f.name);
    setResult(null);
    setSelectedHistoryId(null);
    createAudioUrl(f);
    loadWaveform(f);
  };

  const clearFile = () => {
    setFile(null);
    setFileName("");
    setWaveformData(null);
    setResult(null);
    setCurrentAudioUrl(null);
    if (audioUrlRef.current) { URL.revokeObjectURL(audioUrlRef.current); audioUrlRef.current = null; }
    if (fileRef.current) fileRef.current.value = "";
  };

  // ── Drag & Drop ──
  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);
  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    handleFile(f);
  };

  // Analyze
  const analyze = async () => {
    if (!file) { setError("Please select an audio file first"); return; }
    setLoading(true);
    setError(null);

    try {
      const base64 = await toBase64(file);
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_base64: base64.split(",")[1], filename: fileName }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
      if (data.waveform?.length > 0) setWaveformData(data.waveform);

      const entry = {
        id: data.id || `local_${Date.now()}`,
        name: fileName,
        data,
        audioUrl: currentAudioUrl,
        timestamp: new Date().toLocaleString("en-IN", {
          hour: "2-digit", minute: "2-digit",
          day: "2-digit", month: "short",
        }),
      };

      setHistory((prev) => [entry, ...prev.filter(h => h.id !== entry.id)].slice(0, 30));
      setSelectedHistoryId(entry.id);

    } catch (err) {
      setError(err.message || "Analysis failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(HISTORY_KEY);
  };

  const selectHistoryItem = (item) => {
    setResult(item.data);
    setSelectedHistoryId(item.id);
    if (item.data?.waveform) setWaveformData(item.data.waveform);
    setCurrentAudioUrl(item.audioUrl || null);
    setFileName(item.name);
    setFile(null);
  };

  const isAI = result?.classification === "AI_GENERATED";
  const confidence = result?.confidence ?? 0;

  // ═════════════════════════════════════════════════════════════════════
  //  RENDER
  // ═════════════════════════════════════════════════════════════════════

  return (
    <div className="voice-ui-root">

      {/* ── Loading Overlay ── */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <div className="loading-text">Analyzing Voice...</div>
        </div>
      )}

      {/* ── Error Toast ── */}
      <div className={`error-toast ${error ? "show" : ""}`}>
        ⚠ {error}
      </div>

      {/* ═══ LEFT PANEL — History ═══ */}
      <div className="voice-left-panel">
        <div className="panel-header">
          <div className="dot" />
          <h3>Recent History</h3>
        </div>

        {history.length === 0 && (
          <div className="history-empty">No analyses yet.<br />Upload a file to start.</div>
        )}

        <div className="history-list">
          {history.map((item) => {
            const itemIsAI = item.data?.classification === "AI_GENERATED";
            const isSelected = item.id === selectedHistoryId;
            return (
              <div
                key={item.id}
                className={`history-item ${itemIsAI ? "ai" : "human"} ${isSelected ? "selected" : ""}`}
                onClick={() => selectHistoryItem(item)}
              >
                <div className="h-title" title={item.name}>
                  🎵 {item.name.length > 22 ? item.name.slice(0, 22) + "\u2026" : item.name}
                </div>
                <div className="h-meta">
                  <span className={`h-badge ${itemIsAI ? "ai" : "human"}`}>
                    {itemIsAI ? "🤖 AI" : "👤 Human"}
                  </span>
                  <span className="h-conf">{Math.round((item.data?.confidence ?? 0) * 100)}%</span>
                </div>
                <div className="h-time">{item.timestamp}</div>
              </div>
            );
          })}
        </div>

        {history.length > 0 && (
          <button className="clear-btn" onClick={clearHistory}>
            🗑 Clear History
          </button>
        )}
      </div>

      {/* ═══ CENTER — Upload & Analyze ═══ */}
      <div className="voice-center">

        {/* Orb */}
        <div className="ai-orb-container">
          <div className="ai-orb-ring" />
          <div className={`ai-orb ${loading ? "analyzing" : ""} ${result ? (isAI ? "orb-ai" : "orb-human") : ""}`}>
            <div className="ai-orb-line" />
          </div>
        </div>

        <div className={`status-text ${loading ? "analyzing" : ""}`}>
          {loading ? "ANALYZING..." : result ? (isAI ? "\u26a0 AI DETECTED" : "\u2713 HUMAN DETECTED") : "READY FOR ANALYSIS"}
        </div>

        {/* Waveform */}
        <div className="waveform-container">
          <span className="waveform-label">Waveform</span>
          {waveformData ? (
            <WaveformCanvas data={waveformData} isAI={result ? isAI : null} />
          ) : (
            <div className="waveform-placeholder">Upload a file to see waveform</div>
          )}
        </div>

        {/* Audio Player for uploaded file */}
        {currentAudioUrl && (
          <AudioPlayerMini
            key={currentAudioUrl}
            src={currentAudioUrl}
            label={fileName ? `\uD83C\uDFB5 ${fileName}` : "Uploaded Audio"}
          />
        )}

        {/* Drop Zone */}
        <div
          className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          {!file ? (
            <>
              <div className="drop-icon">🎤</div>
              <div className="drop-text">
                Drop audio file or <strong>click to browse</strong>
              </div>
              <div className="drop-formats">MP3 · WAV · M4A · OGG · FLAC</div>
            </>
          ) : (
            <div className="selected-file">
              <span className="file-name">🎵 {fileName}</span>
              <button className="file-remove" onClick={(e) => { e.stopPropagation(); clearFile(); }}>✕</button>
            </div>
          )}
          <input
            ref={fileRef}
            type="file"
            hidden
            accept="audio/*"
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>

        <button
          className="analyze-btn"
          onClick={analyze}
          disabled={loading || !file}
        >
          {loading ? "ANALYZING..." : "ANALYZE VOICE"}
        </button>
      </div>

      {/* ═══ RIGHT PANEL — Results ═══ */}
      <div className="voice-right-panel">

        {!result && (
          <div className="result-placeholder">
            <div className="ph-icon">🔬</div>
            <div className="ph-text">
              Upload and analyze a voice file<br />to see detailed results
            </div>
          </div>
        )}

        {result && (
          <>
            {/* Verdict */}
            <div className={`verdict-badge ${isAI ? "ai" : "human"}`}>
              <div className="verdict-label">Classification</div>
              <div className={`verdict-text ${isAI ? "ai" : "human"}`}>
                {isAI ? "AI GENERATED" : "HUMAN VOICE"}
              </div>
            </div>

            {/* Confidence */}
            <div className="confidence-section">
              <div className="confidence-label">Confidence Score</div>
              <ConfidenceMeter value={confidence} isAI={isAI} />
              <div className="confidence-hint">
                {confidence < 0.65
                  ? "\u26a0 Low confidence \u2014 result may be uncertain"
                  : confidence < 0.80
                  ? "Moderate confidence in this result"
                  : "High confidence detection"}
              </div>
            </div>

            {/* Language & Duration */}
            <div className="info-row">
              <span className="info-label">Language</span>
              <span className="info-value">
                {LANG_FLAGS[result.language_detected] || "\uD83C\uDF10"} {result.language_detected || "Unknown"}
              </span>
            </div>

            {/* Audio Playback */}
            {currentAudioUrl && (
              <AudioPlayerMini key={currentAudioUrl + "r"} src={currentAudioUrl} label="\u25B6 Play Analyzed Audio" />
            )}

            {/* TTS Response */}
            {result.audio_response_base64 && (
              <AudioPlayerMini
                src={`data:audio/mp3;base64,${result.audio_response_base64}`}
                label="\uD83D\uDD0A Voice Response"
              />
            )}

            {/* Explanation */}
            <div className="explanation-box">
              <div className="ex-title">Detection Indicators</div>
              {result.explanation?.replace("Indicators: ", "").split(", ").map((ind, i) => (
                <div key={i} className="ex-indicator"><span className="ex-dot" />{ind}</div>
              ))}
            </div>

            {/* Feature Panel */}
            {result.features && (
              <div className="feature-panel">
                <div className="feature-panel-header" onClick={() => setFeaturesOpen(!featuresOpen)}>
                  <span className="fp-title">Audio Features</span>
                  <span className={`fp-arrow ${featuresOpen ? "open" : ""}`}>▼</span>
                </div>
                <div className={`feature-panel-body ${featuresOpen ? "open" : ""}`}>
                  <div className="feature-grid">
                    {/* AI-discriminative features first */}
                    <FeatureItem label="Pitch Jitter" value={result.features.pitch_jitter ?? 0} maxVal={0.15} />
                    <FeatureItem label="Harmonic Ratio" value={result.features.harmonic_ratio ?? 0} maxVal={1} aiHigh />
                    <FeatureItem label="RMS CV" value={result.features.rms_cv ?? 0} maxVal={2} />
                    <FeatureItem label="Pitch Entropy" value={result.features.pitch_entropy ?? 0} maxVal={5} />
                    <FeatureItem label="MFCC Change" value={result.features.mfcc_temporal_change ?? 0} maxVal={8} />
                    <FeatureItem label="Spectral Flux" value={result.features.spectral_flux ?? 0} maxVal={15} />
                    <FeatureItem label="Crest Factor" value={result.features.spectral_crest_factor ?? 0} maxVal={40} />
                    <FeatureItem label="Pitch Mean (Hz)" value={result.features.pitch_mean} maxVal={400} />
                    <FeatureItem label="Pitch Std" value={result.features.pitch_std} maxVal={80} />
                    <FeatureItem label="Pitch Range" value={result.features.pitch_range} maxVal={200} />
                    <FeatureItem label="Voiced Ratio" value={result.features.voiced_ratio} displayValue={`${(result.features.voiced_ratio * 100).toFixed(0)}%`} maxVal={1} />
                    <FeatureItem label="Silence Ratio" value={result.features.silence_ratio} displayValue={`${(result.features.silence_ratio * 100).toFixed(0)}%`} maxVal={1} />
                    <FeatureItem label="Spec. Flatness" value={result.features.spectral_flatness} maxVal={0.5} />
                    <FeatureItem label="ZCR" value={result.features.zcr} maxVal={0.15} />
                    <FeatureItem label="RMS Energy" value={result.features.rms_energy} maxVal={0.5} />
                    <FeatureItem label="Tempo (BPM)" value={result.features.tempo} maxVal={200} />
                    <FeatureItem label="MFCC Variability" value={result.features.mfcc_variability} maxVal={30} />
                  </div>
                </div>
              </div>
            )}

            {/* TTS Audio handled above already */}
          </>
        )}
      </div>
    </div>
  );
}

/* ── Utility ── */

function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
