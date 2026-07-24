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

The this-game numbers are shrunk toward that same team's season-to-date
number (see _shrunk_ratio) - confirmed empirically
(scripts/analyze_accuracy_by_game_progress.py) that without this, a
single early-game play could swing the raw in-game average wildly and
measurably hurt accuracy in the first few plays of a game.
"""

import polars as pl

_TIME_ORDER = ["season", "week", "game_id", "play_id"]

# How much weight the season-long prior gets when blending into an
# in-game estimate, in units of "in-game attempts" - see _shrunk_ratio.
# 10 means a team needs roughly 10 real attempts this game before its
# in-game number starts to outweigh its season-long one.
_SHRINKAGE_K = 10

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


def _prior_sum_count(
    numerator: str, denominator: str, partition_by: list[str]
) -> tuple[pl.Expr, pl.Expr]:
    """Expanding (sum, count) of numerator/denominator, before the current row."""
    prior_num = pl.col(numerator).cum_sum().over(partition_by).shift(1).over(partition_by)
    prior_den = pl.col(denominator).cum_sum().over(partition_by).shift(1).over(partition_by)
    return prior_num, prior_den


def _prior_ratio(numerator: str, denominator: str, partition_by: list[str]) -> pl.Expr:
    """Expanding numerator/denominator, computed before the current row."""
    prior_num, prior_den = _prior_sum_count(numerator, denominator, partition_by)
    return pl.when(prior_den > 0).then(prior_num / prior_den).otherwise(None)


def _shrunk_ratio(
    numerator: str, denominator: str, game_partition: list[str], season_prior: pl.Expr
) -> pl.Expr:
    """In-game numerator/denominator, shrunk toward season_prior by _SHRINKAGE_K.

    A team's raw in-game average is noisy with only a handful of plays
    behind it - one stuffed carry shouldn't convince the model this team
    can't run at all. Blending it toward that team's own season-long
    number (weighted by _SHRINKAGE_K "phantom" attempts) fixes that: with
    0 real attempts this game the estimate is just the season prior, and
    it gradually shifts to reflect what's actually happening today as
    real attempts accumulate. Unlike a plain ratio, this is never null -
    even on a team's first play of the game there's still a season prior
    to fall back to (or 0.0, a neutral EPA value, if even that's missing).
    """
    prior_num, prior_den = _prior_sum_count(numerator, denominator, game_partition)
    prior_num = prior_num.fill_null(0.0)
    prior_den = prior_den.fill_null(0)
    prior_value = season_prior.fill_null(0.0)
    return (prior_num + _SHRINKAGE_K * prior_value) / (prior_den + _SHRINKAGE_K)


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

    # Season-to-date ratios first - the in-game ones below shrink toward
    # these as their prior, so they need to already exist as columns.
    df = df.with_columns(
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
        # Same shape of stat for the offense's own season rushing/passing
        # efficiency, kept private - only used below as the shrinkage
        # prior, not exposed as its own feature (out of scope for the
        # in-game noise fix this is here for).
        _prior_ratio("_rush_epa", "_is_run", ["season", "posteam"]).alias(
            "_posteam_epa_season_rush"
        ),
        _prior_ratio("_pass_epa", "_is_pass", ["season", "posteam"]).alias(
            "_posteam_epa_season_pass"
        ),
    )

    return df.with_columns(
        _shrunk_ratio(
            "_rush_epa", "_is_run", ["game_id", "posteam"], pl.col("_posteam_epa_season_rush")
        ).alias("posteam_epa_this_game_rush"),
        _shrunk_ratio(
            "_pass_epa", "_is_pass", ["game_id", "posteam"], pl.col("_posteam_epa_season_pass")
        ).alias("posteam_epa_this_game_pass"),
        _shrunk_ratio(
            "_rush_epa", "_is_run", ["game_id", "defteam"], pl.col("defteam_epa_allowed_rush")
        ).alias("defteam_epa_allowed_this_game_rush"),
        _shrunk_ratio(
            "_pass_epa", "_is_pass", ["game_id", "defteam"], pl.col("defteam_epa_allowed_pass")
        ).alias("defteam_epa_allowed_this_game_pass"),
    ).drop(
        [
            "_is_run",
            "_is_pass",
            "_is_scrimmage",
            "_rush_epa",
            "_pass_epa",
            "_posteam_epa_season_rush",
            "_posteam_epa_season_pass",
        ]
    )
