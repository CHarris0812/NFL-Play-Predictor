"""Run this file to check whether accuracy varies by how far into the
game we are.

The in-game tendency features (features.tendency) are plain averages
with no notion of sample size attached - a team's rush EPA after 1
carry looks structurally identical to the model as after 15 carries.
This checks whether that actually costs accuracy early in games, by
bucketing the test season's predictions by how many prior plays each
team has had in that specific game (not the model's own feature - a
diagnostic-only column computed the same leak-free way).

Requires the models to already be trained: run scripts/train_baseline.py
and scripts/train_outcome_baseline.py first if artifacts/ is empty.
"""

import polars as pl

from features.outcome import RAW_OUTCOME_COLUMNS, add_outcome_label
from features.situational import play_universe
from ingestion.nflverse import load_pbp
from models.persistence import load_xgb_model
from models.predict import predict_batch
from serving.replay import OUTCOME_COLUMN, PLAY_TYPE_COLUMN

TEST_SEASON = 2023

# (low, high) plays this team has already run this game, inclusive.
BUCKETS = [(0, 2), (3, 5), (6, 10), (11, 20), (21, 999)]


def _add_plays_this_game_so_far(pbp: pl.DataFrame) -> pl.DataFrame:
    """How many prior scrimmage plays this team has had this game -
    the sample size behind the in-game tendency features, before this
    play. Same leak-free logic as features.tendency, kept standalone
    here since it's diagnostic-only, not a model feature."""
    is_scrimmage = pl.col("play_type").is_in(["run", "pass"])
    time_order = ["season", "week", "game_id", "play_id"]
    count = (
        is_scrimmage.cast(pl.Int64)
        .cum_sum()
        .over(["game_id", "posteam"])
        .shift(1)
        .over(["game_id", "posteam"])
        .fill_null(0)
    )
    return pbp.sort(time_order).with_columns(count.alias("plays_this_game_so_far"))


def main() -> None:
    print(f"Loading {TEST_SEASON} play-by-play from nflverse...")
    pbp = _add_plays_this_game_so_far(load_pbp([TEST_SEASON]))

    extra = [PLAY_TYPE_COLUMN, *RAW_OUTCOME_COLUMNS, "plays_this_game_so_far"]
    df = add_outcome_label(play_universe(pbp, extra_columns=extra))
    rows = df.to_dicts()
    print(f"Evaluating {len(rows):,} plays\n")

    play_type_model = load_xgb_model("play_type")
    outcome_model = load_xgb_model("outcome")
    play_type_preds = predict_batch(play_type_model, rows)
    outcome_preds = predict_batch(outcome_model, rows)

    print(f"{'plays so far':<14}{'n':>8}{'play-type acc':>16}{'outcome acc':>14}")
    for lo, hi in BUCKETS:
        idx = [i for i, r in enumerate(rows) if lo <= r["plays_this_game_so_far"] <= hi]
        if not idx:
            continue
        pt_correct = sum(
            max(play_type_preds[i], key=play_type_preds[i].get) == rows[i][PLAY_TYPE_COLUMN]
            for i in idx
        )
        out_correct = sum(
            max(outcome_preds[i], key=outcome_preds[i].get) == rows[i][OUTCOME_COLUMN]
            for i in idx
        )
        n = len(idx)
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        print(f"{label:<14}{n:>8,}{pt_correct / n:>15.1%}{out_correct / n:>14.1%}")


if __name__ == "__main__":
    main()
