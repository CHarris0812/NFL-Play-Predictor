import { useState } from "react";
import { fetchReplay, predictSituation } from "./api";

const DEFAULT_GAME_ID = "2023_22_SF_KC"; // Super Bowl LVIII

function topChoice(probabilities) {
  return Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0];
}

export default function ReplayTab() {
  const [gameIdInput, setGameIdInput] = useState(DEFAULT_GAME_ID);
  const [gameId, setGameId] = useState(null);
  const [plays, setPlays] = useState(null);
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState([]);
  const [loading, setLoading] = useState(false);
  const [advancing, setAdvancing] = useState(false);
  const [error, setError] = useState(null);

  const atEnd = plays !== null && index >= plays.length;
  const playTypeCorrect = revealed.filter((r) => r.playTypeHit).length;
  const outcomeCorrect = revealed.filter((r) => r.outcomeHit).length;

  async function handleLoadGame() {
    setLoading(true);
    setError(null);
    setPlays(null);
    setRevealed([]);
    setIndex(0);
    try {
      const data = await fetchReplay(gameIdInput.trim());
      setGameId(data.game_id);
      setPlays(data.plays);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleNextPlay() {
    if (!plays || index >= plays.length) return;

    setAdvancing(true);
    setError(null);
    try {
      const play = plays[index];
      const prediction = await predictSituation(play);

      const [predictedPlayType] = topChoice(prediction.play_type);
      const [predictedOutcome] = topChoice(prediction.outcome);

      setRevealed((prev) => [
        ...prev,
        {
          play,
          prediction,
          predictedPlayType,
          predictedOutcome,
          playTypeHit: predictedPlayType === play.play_type,
          outcomeHit: predictedOutcome === play.outcome,
        },
      ]);
      setIndex((i) => i + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setAdvancing(false);
    }
  }

  return (
    <section className="card">
      <h2>Replay a game</h2>
      <p className="hint">
        Steps through a completed game's plays in order, predicting each one before revealing
        what actually happened - a stand-in for a live feed, using a game the models never
        trained on (2023 is held out for testing).
      </p>

      <div className="field-grid">
        <label className="field">
          Game ID
          <input
            type="text"
            value={gameIdInput}
            onChange={(e) => setGameIdInput(e.target.value)}
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
            {gameId}: play {Math.min(index + 1, plays.length)} of {plays.length}
            {revealed.length > 0 &&
              ` | play-type ${playTypeCorrect}/${revealed.length} | outcome ${outcomeCorrect}/${revealed.length}`}
          </p>

          <button onClick={handleNextPlay} disabled={advancing || atEnd}>
            {atEnd ? "Game complete" : advancing ? "Predicting..." : "Next play"}
          </button>

          <div className="replay-log">
            {revealed.map((r, i) => (
              <div className="replay-row" key={i}>
                <span className="replay-situation">
                  Q{r.play.qtr} {r.play.down}&{r.play.ydstogo}, {r.play.yardline_100} yds from
                  end zone
                </span>
                <span className={r.playTypeHit ? "replay-hit" : "replay-miss"}>
                  play: {r.predictedPlayType} ({(r.prediction.play_type[r.predictedPlayType] * 100).toFixed(0)}%) -&gt;{" "}
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
