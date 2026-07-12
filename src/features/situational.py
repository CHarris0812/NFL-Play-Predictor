"""Situational features for the play-type baseline.

Builds a (features, label) dataset directly from nflverse play-by-play
columns - no external data sources needed. Used for both training and,
later, live inference, so this is the single source of truth for what
"the features" are.
"""

import polars as pl

# Real offensive-play decisions only. Excludes no_play (penalty/timeout,
# not an actual outcome - see the no_play investigation), kickoffs and
# extra points (not a play-calling decision), and qb_kneel/qb_spike
# (deterministic, not interesting to predict).
PLAY_TYPES = ["run", "pass", "punt", "field_goal"]

FEATURE_COLUMNS = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "qtr",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "is_home",
]

LABEL_COLUMN = "play_type"

# Carried through for splitting/grouping, not used as model inputs.
ID_COLUMNS = ["season", "week", "game_id"]


def build_dataset(pbp: pl.DataFrame) -> pl.DataFrame:
    """Filter raw play-by-play down to a clean features+label dataset.

    Keeps only plays on downs 1-4 with a play_type in PLAY_TYPES, adds
    an is_home indicator, and drops rows with nulls in any feature or
    the label.
    """
    return (
        pbp.filter(pl.col("down").is_in([1, 2, 3, 4]))
        .filter(pl.col(LABEL_COLUMN).is_in(PLAY_TYPES))
        .with_columns((pl.col("posteam_type") == "home").alias("is_home"))
        .select(ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN])
        .drop_nulls()
    )
