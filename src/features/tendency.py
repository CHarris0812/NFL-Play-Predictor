"""Rolling team tendency features.

Two kinds of rolling stat, both strictly from prior plays - never the
play being predicted, never anything after it (safe for both training
and live inference, since at prediction time you only ever know what
already happened):

- Season-to-date: a team's overall tendency this season, resets each
  season. Good for "is this generally a run-heavy team."
- This-game: the same shape of stat, but windowed to just the plays
  already run in the CURRENT game. A season average dilutes in-game
  momentum - e.g. a normally strong rushing offense that's been stuffed
  all game so far. Computed separately per team (offense and defense
  each get their own game-scoped number), since one team's hot streak
  says nothing about the other team's.
"""

import polars as pl

_TIME_ORDER = ["season", "week", "game_id", "play_id"]

TENDENCY_COLUMNS = [
    "posteam_run_rate",
    "posteam_run_rate_this_down",
    "defteam_epa_allowed_rush",
    "defteam_epa_allowed_pass",
    "posteam_epa_this_game_rush",
    "posteam_epa_this_game_pass",
    "defteam_epa_allowed_this_game_rush",
    "defteam_epa_allowed_this_game_pass",
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
        is_pass.cast(pl.Int64).alias("_is_pass"),
        is_scrimmage.cast(pl.Int64).alias("_is_scrimmage"),
        pl.when(is_run).then(pl.col("epa")).otherwise(0.0).alias("_rush_epa"),
        pl.when(is_pass).then(pl.col("epa")).otherwise(0.0).alias("_pass_epa"),
    )

    return df.with_columns(
        # Season-to-date.
        _prior_ratio("_is_run", "_is_scrimmage", ["season", "posteam"]).alias(
            "posteam_run_rate"
        ),
        _prior_ratio("_is_run", "_is_scrimmage", ["season", "posteam", "down"]).alias(
            "posteam_run_rate_this_down"
        ),
        _prior_ratio("_rush_epa", "_is_run", ["season", "defteam"]).alias(
            "defteam_epa_allowed_rush"
        ),
        _prior_ratio("_pass_epa", "_is_pass", ["season", "defteam"]).alias(
            "defteam_epa_allowed_pass"
        ),
        # This game only - same ratios, windowed to game_id instead of
        # season, and computed for both sides of the ball.
        _prior_ratio("_rush_epa", "_is_run", ["game_id", "posteam"]).alias(
            "posteam_epa_this_game_rush"
        ),
        _prior_ratio("_pass_epa", "_is_pass", ["game_id", "posteam"]).alias(
            "posteam_epa_this_game_pass"
        ),
        _prior_ratio("_rush_epa", "_is_run", ["game_id", "defteam"]).alias(
            "defteam_epa_allowed_this_game_rush"
        ),
        _prior_ratio("_pass_epa", "_is_pass", ["game_id", "defteam"]).alias(
            "defteam_epa_allowed_this_game_pass"
        ),
    ).drop(["_is_run", "_is_pass", "_is_scrimmage", "_rush_epa", "_pass_epa"])
