"""Simulate a live game by replaying a completed one, play by play.

Predicts each play from only what was knowable at that point - the same
leak-free, prior-plays-only features used for training - then reveals
what actually happened. This is how the live prediction loop gets
exercised and tested without waiting for an actual live game: 2023 is
our held-out test season, never seen in training, so a replay of any
2023 game is an honest dry run of the real thing.
"""

import time

import polars as pl

from features.outcome import RAW_OUTCOME_COLUMNS, add_outcome_label
from features.situational import FEATURE_COLUMNS, ID_COLUMNS, LABEL_COLUMN, play_universe
from ingestion.nflverse import load_pbp
from models.persistence import load_xgb_model
from models.predict import predict

PLAY_TYPE_COLUMN = LABEL_COLUMN  # "play_type" - named for clarity alongside "outcome"
OUTCOME_COLUMN = "outcome"

# Not model features - just human-readable context about the real play,
# shown alongside the prediction. "desc" is nflverse's own broadcast-style
# text description (player names, yardage, etc.), so no need to build one.
DISPLAY_COLUMNS = ["posteam", "desc"]


def build_game_replay(pbp: pl.DataFrame, game_id: str) -> pl.DataFrame:
    """One game's plays, in order, with features and both actual labels attached."""
    extra = [PLAY_TYPE_COLUMN, *RAW_OUTCOME_COLUMNS, *DISPLAY_COLUMNS]
    df = add_outcome_label(play_universe(pbp, extra_columns=extra))
    return (
        df.filter(pl.col("game_id") == game_id)
        .select([*ID_COLUMNS, *FEATURE_COLUMNS, PLAY_TYPE_COLUMN, OUTCOME_COLUMN, *DISPLAY_COLUMNS])
        .sort("play_id")
    )


def _top_choice(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=probabilities.get)


def run_replay(game_id: str, season: int, delay_seconds: float = 0.0) -> None:
    """Print a play-by-play prediction-vs-actual replay of one game."""
    print(f"Loading {season} play-by-play from nflverse...")
    game = build_game_replay(load_pbp([season]), game_id)

    if game.shape[0] == 0:
        print(f"No plays found for game_id={game_id!r} in season {season}.")
        return

    play_type_model = load_xgb_model("play_type")
    outcome_model = load_xgb_model("outcome")

    play_type_correct = 0
    outcome_correct = 0
    n = game.shape[0]

    print(f"Replaying {game_id}: {n} plays")
    print()

    for i, row in enumerate(game.iter_rows(named=True), start=1):
        features = {col: row[col] for col in FEATURE_COLUMNS}

        play_type_probs = predict(play_type_model, features)
        outcome_probs = predict(outcome_model, features)
        predicted_play_type = _top_choice(play_type_probs)
        predicted_outcome = _top_choice(outcome_probs)

        play_type_hit = predicted_play_type == row[PLAY_TYPE_COLUMN]
        outcome_hit = predicted_outcome == row[OUTCOME_COLUMN]
        play_type_correct += play_type_hit
        outcome_correct += outcome_hit

        print(
            f"[{i:>3}/{n}] {row['posteam']} Q{row['qtr']:.0f} {row['down']:.0f}&{row['ydstogo']:.0f}, "
            f"{row['yardline_100']:.0f} yds from end zone | "
            f"play: {predicted_play_type} ({play_type_probs[predicted_play_type]:.0%}) "
            f"-> {row[PLAY_TYPE_COLUMN]} [{'y' if play_type_hit else 'n'}] | "
            f"outcome: {predicted_outcome} -> {row[OUTCOME_COLUMN]} "
            f"[{'y' if outcome_hit else 'n'}]"
        )
        print(f"        {row['desc']}")

        if delay_seconds:
            time.sleep(delay_seconds)

    print()
    print(f"Play-type accuracy: {play_type_correct}/{n} ({play_type_correct / n:.1%})")
    print(f"Outcome accuracy:   {outcome_correct}/{n} ({outcome_correct / n:.1%})")
