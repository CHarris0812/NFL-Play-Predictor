import { useState } from "react";
import { fetchReplay } from "./api";
import ProbabilityList from "./ProbabilityList";

// Super Bowl LVIII: 2023 season, week 22, San Francisco at Kansas City.
const DEFAULT_GAME = { year: 2023, week: 22, away: "SF", home: "KC" };

// The exact features the model saw for this play - shown in full when a
// play is clicked, alongside the full probability breakdown.
const FEATURE_LABELS = [
  { key: "down", label: "Down" },
  { key: "ydstogo", label: "Yards to go" },
  { key: "yardline_100", label: "Yards from opponent end zone" },
  { key: "score_differential", label: "Score differential" },
  { key: "qtr", label: "Quarter" },
  { key: "half_seconds_remaining", label: "Seconds left in half" },
  { key: "posteam_timeouts_remaining", label: "Offense timeouts left" },
  { key: "defteam_timeouts_remaining", label: "Defense timeouts left" },
  { key: "is_home", label: "Offense is home" },
  { key: "posteam_run_rate", label: "Offense run rate (season)" },
  { key: "posteam_run_rate_this_down", label: "Offense run rate on this down" },
  { key: "defteam_epa_allowed_rush", label: "Defense EPA allowed per rush (season)" },
  { key: "defteam_epa_allowed_pass", label: "Defense EPA allowed per pass (season)" },
  { key: "posteam_epa_this_game_rush", label: "Offense EPA per rush (this game)" },
  { key: "posteam_epa_this_game_pass", label: "Offense EPA per pass (this game)" },
  { key: "defteam_epa_allowed_this_game_rush", label: "Defense EPA allowed per rush (this game)" },
  { key: "defteam_epa_allowed_this_game_pass", label: "Defense EPA allowed per pass (this game)" },
];

function topChoice(probabilities) {
  return Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0];
}

// nflverse's game_id format: {season}_{week, zero-padded}_{away}_{home}.
function toGameId({ year, week, away, home }) {
  return `${year}_${String(week).padStart(2, "0")}_${away.toUpperCase()}_${home.toUpperCase()}`;
}

function formatFeatureValue(value) {
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number" && !Number.isInteger(value)) return value.toFixed(3);
  return value;
}

export default function ReplayTab() {
  const [game, setGame] = useState(DEFAULT_GAME);
  const [gameId, setGameId] = useState(null);
  const [plays, setPlays] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedIndex, setExpandedIndex] = useState(null);

  function updateGame(key, value) {
    setGame((prev) => ({ ...prev, [key]: value }));
  }

  async function handleLoadGame() {
    setLoading(true);
    setError(null);
    setPlays(null);
    setExpandedIndex(null);
    try {
      const data = await fetchReplay(toGameId(game));
      setGameId(data.game_id);
      setPlays(data.plays);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const rows = (plays ?? []).map((play) => {
    const [predictedPlayType] = topChoice(play.predicted_play_type);
    const [predictedOutcome] = topChoice(play.predicted_outcome);
    return {
      play,
      predictedPlayType,
      predictedOutcome,
      playTypeProb: play.predicted_play_type[predictedPlayType],
      outcomeProb: play.predicted_outcome[predictedOutcome],
      playTypeHit: predictedPlayType === play.play_type,
      outcomeHit: predictedOutcome === play.outcome,
    };
  });
  const playTypeCorrect = rows.filter((r) => r.playTypeHit).length;
  const outcomeCorrect = rows.filter((r) => r.outcomeHit).length;

  return (
    <section className="card">
      <h2>Replay a game</h2>

      <div className="field-grid">
        <label className="field">
          Year
          <input
            type="number"
            value={game.year}
            onChange={(e) => updateGame("year", Number(e.target.value))}
          />
        </label>
        <label className="field">
          Week
          <input
            type="number"
            value={game.week}
            onChange={(e) => updateGame("week", Number(e.target.value))}
          />
        </label>
        <label className="field">
          Away team
          <input
            type="text"
            value={game.away}
            onChange={(e) => updateGame("away", e.target.value)}
          />
        </label>
        <label className="field">
          Home team
          <input
            type="text"
            value={game.home}
            onChange={(e) => updateGame("home", e.target.value)}
          />
        </label>
      </div>

      <button onClick={handleLoadGame} disabled={loading}>
        {loading ? "Loading..." : "Load game"}
      </button>

      {error && <p className="error">{error}</p>}

      {plays && (
        <>
          <p className="hint">
            {gameId}: {plays.length} plays | play-type {playTypeCorrect}/{plays.length} | outcome{" "}
            {outcomeCorrect}/{plays.length}
          </p>
          <p className="hint">Click a play to see the full model prediction.</p>

          <div className="replay-log">
            {rows.map((r, i) => {
              const isExpanded = expandedIndex === i;
              return (
                <div className="replay-row-wrapper" key={i}>
                  <div
                    className="replay-row"
                    onClick={() => setExpandedIndex(isExpanded ? null : i)}
                  >
                    <span className="replay-situation">
                      {r.play.posteam} ball, Q{r.play.qtr} {r.play.down}&{r.play.ydstogo},{" "}
                      {r.play.yardline_100} yds from end zone
                    </span>
                    <span className={r.playTypeHit ? "replay-hit" : "replay-miss"}>
                      play: {r.predictedPlayType} ({(r.playTypeProb * 100).toFixed(0)}%) -&gt;{" "}
                      {r.play.play_type}
                    </span>
                    <span className={r.outcomeHit ? "replay-hit" : "replay-miss"}>
                      outcome: {r.predictedOutcome} ({(r.outcomeProb * 100).toFixed(0)}%) -&gt;{" "}
                      {r.play.outcome}
                    </span>
                  </div>

                  {isExpanded && (
                    <div className="replay-detail">
                      <p className="replay-desc">{r.play.desc}</p>
                      <div className="replay-detail-features">
                        {FEATURE_LABELS.map(({ key, label }) => (
                          <div className="replay-detail-feature" key={key}>
                            <span>{label}</span>
                            <span>{formatFeatureValue(r.play[key])}</span>
                          </div>
                        ))}
                      </div>
                      <div className="results">
                        <ProbabilityList
                          title="Play type"
                          probabilities={r.play.predicted_play_type}
                        />
                        <ProbabilityList
                          title="Outcome"
                          probabilities={r.play.predicted_outcome}
                        />
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
