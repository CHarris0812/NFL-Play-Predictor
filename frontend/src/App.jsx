import { useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

const DEFAULT_SITUATION = {
  down: 3,
  ydstogo: 2,
  yardline_100: 5,
  score_differential: 0,
  qtr: 4,
  minutes_left_in_quarter: 1,
  seconds_left_in_quarter: 50,
  posteam_timeouts_remaining: 3,
  defteam_timeouts_remaining: 3,
  is_home: true,
};

const FIELDS = [
  { key: "down", label: "Down" },
  { key: "ydstogo", label: "Yards to go" },
  { key: "yardline_100", label: "Yards from opponent end zone" },
  { key: "score_differential", label: "Score differential (offense - defense)" },
  { key: "qtr", label: "Quarter" },
  { key: "minutes_left_in_quarter", label: "Minutes left in quarter" },
  { key: "seconds_left_in_quarter", label: "Seconds left in quarter" },
  { key: "posteam_timeouts_remaining", label: "Offense timeouts left" },
  { key: "defteam_timeouts_remaining", label: "Defense timeouts left" },
];

// The model was trained on game_seconds_remaining/half_seconds_remaining
// (nflverse's own columns), but the clock a viewer actually sees on TV is
// quarter + a minutes:seconds countdown within that quarter - so the form
// collects the TV version and this converts it before calling the API.
function toApiPayload(situation) {
  const { qtr, minutes_left_in_quarter, seconds_left_in_quarter, ...rest } = situation;
  const secondsInQuarter = minutes_left_in_quarter * 60 + seconds_left_in_quarter;

  const regulationQuartersRemaining = qtr < 4 ? 4 - qtr : 0;
  const game_seconds_remaining = secondsInQuarter + regulationQuartersRemaining * 15 * 60;

  // Q1/Q3 are the first quarter of their half, so the half has another
  // full quarter left after this one; Q2/Q4/OT are the half's last quarter.
  const isFirstQuarterOfHalf = qtr === 1 || qtr === 3;
  const half_seconds_remaining = isFirstQuarterOfHalf
    ? secondsInQuarter + 15 * 60
    : secondsInQuarter;

  return { ...rest, qtr, game_seconds_remaining, half_seconds_remaining };
}

function ProbabilityList({ title, probabilities }) {
  if (!probabilities) return null;

  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="result-panel">
      <h3>{title}</h3>
      {sorted.map(([label, prob]) => (
        <div className="bar-row" key={label}>
          <span className="bar-label">{label}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${prob * 100}%` }} />
          </div>
          <span className="bar-value">{(prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [situation, setSituation] = useState(DEFAULT_SITUATION);
  const [prediction, setPrediction] = useState(null);
  const [predictError, setPredictError] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainLog, setTrainLog] = useState(null);
  const [trainError, setTrainError] = useState(null);

  function updateField(key, value) {
    setSituation((prev) => ({ ...prev, [key]: value }));
  }

  async function handlePredict() {
    setPredicting(true);
    setPredictError(null);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(toApiPayload(situation)),
      });
      if (!response.ok) {
        const body = await response.json();
        throw new Error(body.detail || "Prediction failed");
      }
      setPrediction(await response.json());
    } catch (err) {
      setPredictError(err.message);
      setPrediction(null);
    } finally {
      setPredicting(false);
    }
  }

  async function handleTrain() {
    setTraining(true);
    setTrainError(null);
    setTrainLog(null);
    try {
      const response = await fetch(`${API_BASE}/train`, { method: "POST" });
      if (!response.ok) {
        const body = await response.json();
        throw new Error(body.detail || "Training failed");
      }
      setTrainLog(await response.json());
    } catch (err) {
      setTrainError(err.message);
    } finally {
      setTraining(false);
    }
  }

  return (
    <div className="page">
      <h1>NFL Play Predictor</h1>

      <section className="card">
        <h2>Situation</h2>
        <div className="field-grid">
          {FIELDS.map(({ key, label }) => (
            <label key={key} className="field">
              {label}
              <input
                type="number"
                value={situation[key]}
                onChange={(e) => updateField(key, Number(e.target.value))}
              />
            </label>
          ))}
          <label className="field checkbox-field">
            <input
              type="checkbox"
              checked={situation.is_home}
              onChange={(e) => updateField("is_home", e.target.checked)}
            />
            Offense is home team
          </label>
        </div>

        <button onClick={handlePredict} disabled={predicting}>
          {predicting ? "Predicting..." : "Predict"}
        </button>

        {predictError && <p className="error">{predictError}</p>}

        {prediction && (
          <div className="results">
            <ProbabilityList title="Play type" probabilities={prediction.play_type} />
            <ProbabilityList title="Outcome" probabilities={prediction.outcome} />
          </div>
        )}
      </section>

      <section className="card">
        <h2>Model</h2>
        <p className="hint">
          Retrains both models on nflverse play-by-play (2016-2023). Takes roughly a minute.
        </p>
        <button onClick={handleTrain} disabled={training}>
          {training ? "Training..." : "Train models"}
        </button>

        {trainError && <p className="error">{trainError}</p>}

        {trainLog && (
          <details open>
            <summary>Training log</summary>
            <pre className="log">{trainLog.play_type_log}</pre>
            <pre className="log">{trainLog.outcome_log}</pre>
          </details>
        )}
      </section>
    </div>
  );
}
