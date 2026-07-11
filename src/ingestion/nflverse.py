"""Historical play-by-play access via nflverse (through nflreadpy)."""

import nflreadpy
import polars as pl


def load_pbp(seasons: list[int]) -> pl.DataFrame:
    """Load play-by-play data for the given seasons from nflverse.

    Wraps nflreadpy.load_pbp, which pulls from the nflverse-pbp GitHub
    releases and caches the result locally.
    """
    return nflreadpy.load_pbp(seasons)
