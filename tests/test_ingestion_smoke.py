"""Confirms we can actually pull and use nflverse play-by-play data.

Hits the network (nflverse GitHub releases) - excluded from the default
CI run via the "network" marker registered in pyproject.toml. Run with
`pytest -m network` to include it.
"""

import pytest

from ingestion.nflverse import load_pbp


@pytest.mark.network
def test_load_pbp_returns_usable_2023_season():
    df = load_pbp([2023])

    assert df.shape[0] > 40_000
    assert "play_type" in df.columns
    assert "down" in df.columns

    # 2023: 272 regular season games + 13 playoff games
    assert df["game_id"].n_unique() == 285

    play_types = set(df["play_type"].drop_nulls().unique().to_list())
    assert {"run", "pass"}.issubset(play_types)
