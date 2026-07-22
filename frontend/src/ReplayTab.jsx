import { useState } from "react";
import { fetchReplay } from "./api";

// Super Bowl LVIII: 2023 season, week 22, San Francisco at Kansas City.
const DEFAULT_GAME = { year: 2023, week: 22, away: "SF", home: "KC" };

function topChoice(probabilities) {
  return Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0];
}

// nflverse's game_id format: {season}_{week, zero-padded}_{away}_{home}.
function toGameId({ year, week, away, home }) {
  return `${year}_${String(week).padStart(2, "0")}_${away.toUpperCase()}_${home.toUpperCase()}`;
}

export default function ReplayTab() {
  const [game, setGame] = useState(DEFAULT_GAME);
  const [gameId, setGameId] = useState(null);
  const [plays, setPlays] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function updateGame(key, value) {
    setGame((prev) => ({ ...prev, [key]: value }));
  }

  async function handleLoadGame() {
    setLoading(true);
    setError(null);
    setPlays(null);
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

          <div className="replay-log">
            {rows.map((r, i) => (
              <div className="replay-row" key={i}>
                <span className="replay-situation">
                  Q{r.play.qtr} {r.play.down}&{r.play.ydstogo}, {r.play.yardline_100} yds from
                  end zone
                </span>
                <span className={r.playTypeHit ? "replay-hit" : "replay-miss"}>
                  play: {r.predictedPlayType} ({(r.playTypeProb * 100).toFixed(0)}%) -&gt;{" "}
                  {r.play.play_type}
                </span>
                <span className={r.outcomeHit ? "replay-hit" : "replay-miss"}>
                  outcome: {r.predictedOutcome} -&gt; {r.play.outcome}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
