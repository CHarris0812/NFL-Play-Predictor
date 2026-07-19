"""Outcome-of-play features and label: what happens on the play.

Same play universe and situational features as features.situational,
but labels each play with its result instead of its type - touchdown,
turnover, first down, or no first down. Priority order for plays that
are technically more than one of these (e.g. a pick-six is both a
turnover and a touchdown): touchdown > turnover > first_down >
no_first_down.
"""

import polars as pl

from features.situational import FEATURE_COLUMNS, ID_COLUMNS, play_universe

OUTCOMES = ["touchdown", "turnover", "first_down", "no_first_down"]

LABEL_COLUMN = "outcome"

# Public: serving.replay also needs these to build a combined dataset
# carrying both this module's label and features.situational's.
RAW_OUTCOME_COLUMNS = ["touchdown", "interception", "fumble_lost", "first_down"]


def add_outcome_label(df: pl.DataFrame) -> pl.DataFrame:
    """Add the outcome column to a frame that already has RAW_OUTCOME_COLUMNS."""
    return df.with_columns(
        pl.when(pl.col("touchdown") == 1)
        .then(pl.lit("touchdown"))
        .when((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1))
        .then(pl.lit("turnover"))
        .when(pl.col("first_down") == 1)
        .then(pl.lit("first_down"))
        .otherwise(pl.lit("no_first_down"))
        .alias(LABEL_COLUMN)
    )


def build_dataset(pbp: pl.DataFrame) -> pl.DataFrame:
    """Filter raw play-by-play down to a clean features+outcome dataset."""
    df = play_universe(pbp, extra_columns=RAW_OUTCOME_COLUMNS)
    return add_outcome_label(df).select(ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN])
