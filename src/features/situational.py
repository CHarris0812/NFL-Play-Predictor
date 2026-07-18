"""Situational features for the play-type baseline.

Builds a (features, label) dataset directly from nflverse play-by-play
columns - no external data sources needed. Used for both training and,
later, live inference, so this is the single source of truth for what
"the features" are.
"""

import polars as pl

from features.tendency import TENDENCY_COLUMNS, add_tendency_features

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
    # game_seconds_remaining deliberately omitted: given qtr and
    # half_seconds_remaining it's fully determined (game_seconds_remaining
    # = half_seconds_remaining, +1800 if qtr is 1 or 3), so it would add no
    # information - only noise for logistic regression and diluted SHAP
    # attribution later.
    "half_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "is_home",
    *TENDENCY_COLUMNS,
]

LABEL_COLUMN = "play_type"

# Carried through for splitting/grouping, not used as model inputs.
ID_COLUMNS = ["season", "week", "game_id"]


def play_universe(pbp: pl.DataFrame, extra_columns: list[str] | None = None) -> pl.DataFrame:
    """Filter raw play-by-play down to the shared feature-ready play set.

    Adds rolling tendency features (see features.tendency) before
    filtering, since those need the team's full play history to compute.
    Keeps only plays on downs 1-4 with a play_type in PLAY_TYPES, adds
    an is_home indicator, and drops rows with nulls in any selected
    column - including a team's first tendency-eligible play of the
    season, which has no prior history to compute a rate from yet.
    Shared by this module's build_dataset (labels: play type) and
    features.outcome.build_dataset (labels: what happened on the play)
    - same underlying plays and situational features either way.
    """
    columns = ID_COLUMNS + FEATURE_COLUMNS + (extra_columns or [])
    return (
        add_tendency_features(pbp)
        .filter(pl.col("down").is_in([1, 2, 3, 4]))
        .filter(pl.col("play_type").is_in(PLAY_TYPES))
        .with_columns((pl.col("posteam_type") == "home").alias("is_home"))
        .select(columns)
        .drop_nulls()
    )


def build_dataset(pbp: pl.DataFrame) -> pl.DataFrame:
    """Filter raw play-by-play down to a clean features+label dataset."""
    return play_universe(pbp, extra_columns=[LABEL_COLUMN])
