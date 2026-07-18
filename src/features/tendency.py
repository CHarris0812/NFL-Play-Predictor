"""Rolling team tendency features.

Each stat is computed strictly from a team's own prior plays this
season - never the play being predicted, never anything after it. That
makes this safe to use for both training and live inference: at
prediction time you only ever know what already happened.
"""

import polars as pl

_TIME_ORDER = ["season", "week", "game_id", "play_id"]

TENDENCY_COLUMNS = [
    "posteam_run_rate",
    "posteam_run_rate_this_down",
    "defteam_epa_allowed_rush",
    "defteam_epa_allowed_pass",
]


def _prior_ratio(numerator: str, denominator: str, partition_by: list[str]) -> pl.Expr:
    """Expanding numerator/denominator, computed before the current row."""
    prior_num = pl.col(numerator).cum_sum().over(partition_by).shift(1).over(partition_by)
    prior_den = pl.col(denominator).cum_sum().over(partition_by).shift(1).over(partition_by)
    return pl.when(prior_den > 0).then(prior_num / prior_den).otherwise(None)


def add_tendency_features(pbp: pl.DataFrame) -> pl.DataFrame:
    """Add the columns in TENDENCY_COLUMNS to a raw play-by-play frame.

    Computed on the full frame (not the filtered play-type/down universe)
    so punts/kneels/etc. still get a valid tendency value attached, even
    though they don't themselves count toward any of the rates - a team's
    run rate is defined over their run/pass plays only.
    """
    is_run = pl.col("play_type") == "run"
    is_pass = pl.col("play_type") == "pass"
    is_scrimmage = is_run | is_pass

    df = pbp.sort(_TIME_ORDER).with_columns(
        is_run.cast(pl.Int64).alias("_is_run"),
        is_scrimmage.cast(pl.Int64).alias("_is_scrimmage"),
        pl.when(is_run).then(pl.col("epa")).otherwise(0.0).alias("_rush_epa"),
        pl.when(is_run).then(1).otherwise(0).alias("_is_rush_faced"),
        pl.when(is_pass).then(pl.col("epa")).otherwise(0.0).alias("_pass_epa"),
        pl.when(is_pass).then(1).otherwise(0).alias("_is_pass_faced"),
    )

    return df.with_columns(
        _prior_ratio("_is_run", "_is_scrimmage", ["season", "posteam"]).alias(
            "posteam_run_rate"
        ),
        _prior_ratio("_is_run", "_is_scrimmage", ["season", "posteam", "down"]).alias(
            "posteam_run_rate_this_down"
        ),
        _prior_ratio("_rush_epa", "_is_rush_faced", ["season", "defteam"]).alias(
            "defteam_epa_allowed_rush"
        ),
        _prior_ratio("_pass_epa", "_is_pass_faced", ["season", "defteam"]).alias(
            "defteam_epa_allowed_pass"
        ),
    ).drop(
        ["_is_run", "_is_scrimmage", "_rush_epa", "_is_rush_faced", "_pass_epa", "_is_pass_faced"]
    )
