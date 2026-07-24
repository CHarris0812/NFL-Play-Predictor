import { useState } from "react";
import { predictSituation, trainModels } from "./api";
import ProbabilityList from "./ProbabilityList";

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
  // Rough league-average placeholders - see the TENDENCY_FIELDS note below.
  posteam_run_rate: 0.42,
  posteam_run_rate_this_down: 0.42,
  defteam_epa_allowed_rush: 0.0,
  defteam_epa_allowed_pass: 0.0,
  posteam_epa_this_game_rush: 0.0,
  posteam_epa_this_game_pass: 0.0,
  defteam_epa_allowed_this_game_rush: 0.0,
  defteam_epa_allowed_this_game_pass: 0.0,
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

// These are rolling season-to-date stats (see features/tendency.py), not
// something a viewer can read off the screen - there's no team lookup
// yet, so for now they're just manually-entered numbers like everything
// else in the form. Auto-filling these from real team stats is a natural
// follow-up once there's a team selector.
const TENDENCY_FIELDS = [
  { key: "posteam_run_rate", label: "Offense run rate (season)", step: 0.01 },
  { key: "posteam_run_rate_this_down", label: "Offense run rate on this down", step: 0.01 },
  { key: "defteam_epa_allowed_rush", label: "Defense EPA allowed per rush (season)", step: 0.01 },
  { key: "defteam_epa_allowed_pass", label: "Defense EPA allowed per pass (season)", step: 0.01 },
  { key: "posteam_epa_this_game_rush", label: "Offense EPA per rush (this game)", step: 0.01 },
  { key: "posteam_epa_this_game_pass", label: "Offense EPA per pass (this game)", step: 0.01 },
  {
    key: "defteam_epa_allowed_this_game_rush",
    label: "Defense EPA allowed per rush (this game)",
    step: 0.01,
  },
  {
    key: "defteam_epa_allowed_this_game_pass",
    label: "Defense EPA allowed per pass (this game)",
    step: 0.01,
  },
];

// The model was trained on half_seconds_remaining (nflverse's own column),
// but the clock a viewer actually sees on TV is quarter + a minutes:seconds
// countdown within that quarter - so the form collects the TV version and
// this converts it before calling the API. (There's no game_seconds_remaining
// feature: given qtr and half_seconds_remaining it's fully determined, so it
// would add no information - see the model feature list for details.)
function toApiPayload(situation) {
  const { minutes_left_in_quarter, seconds_left_in_quarter, ...rest } = situation;
  const secondsInQuarter = minutes_left_in_quarter * 60 + seconds_left_in_quarter;

  // Q1/Q3 are the first quarter of their half, so the half has another
  // full quarter left after this one; Q2/Q4/OT are the half's last quarter.
  const isFirstQuarterOfHalf = rest.qtr === 1 || rest.qtr === 3;
  const half_seconds_remaining = isFirstQuarterOfHalf
    ? secondsInQuarter + 15 * 60
    : secondsInQuarter;

  return { ...rest, half_seconds_remaining };
}

export default function PredictTab() {
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
      setPrediction(await predictSituation(toApiPayload(situation)));
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
      setTrainLog(await trainModels());
    } catch (err) {
      setTrainError(err.message);
    } finally {
      setTraining(false);
    }
  }

  return (
    <>
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

        <h3>Team tendency (season and this game)</h3>
        <p className="hint">
          No team lookup yet - enter these manually. 0.42 run rate and 0.0 EPA are roughly
          league-average placeholders; the "this game" ones are 0 since no game is in progress.
        </p>
        <div className="field-grid">
          {TENDENCY_FIELDS.map(({ key, label, step }) => (
            <label key={key} className="field">
              {label}
              <input
                type="number"
                step={step}
                value={situation[key]}
                onChange={(e) => updateField(key, Number(e.target.value))}
              />
            </label>
          ))}
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
    </>
  );
}
