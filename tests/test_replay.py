import itertools

import polars as pl

from serving.replay import OUTCOME_COLUMN, PLAY_TYPE_COLUMN, build_game_replay

_play_id = itertools.count(1)


def _row(**overrides):
    row = {
        "season": 2023,
        "week": 1,
        "game_id": "2023_01_AAA_BBB",
        "play_id": next(_play_id),
        "posteam": "AAA",
        "defteam": "BBB",
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 75,
        "score_differential": 0,
        "qtr": 1,
        "half_seconds_remaining": 1700,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "posteam_type": "home",
        "play_type": "run",
        "desc": "some play description",
        "epa": 0.0,
        "touchdown": 0,
        "interception": 0,
        "fumble_lost": 0,
        "first_down": 0,
    }
    row.update(overrides)
    return row


def test_replay_returns_only_the_requested_game_in_play_order():
    # A prior week against a different game_id gives season-to-date
    # tendency history; a run+pass warmup within the target game itself
    # gives the in-game tendency features (features/tendency.py) their
    # own required history too, so the two asserted rows below aren't
    # dropped as "first play of this game" false starts. The warmup rows
    # themselves are dropped for exactly that reason - they're the ones
    # with no prior in-game history yet.
    season_warmup = [
        _row(week=1, game_id="2023_01_AAA_BBB", play_type="run"),
        _row(week=1, game_id="2023_01_AAA_BBB", play_type="pass"),
    ]
    game_warmup = [
        _row(week=2, game_id="2023_02_AAA_CCC", play_type="run"),
        _row(week=2, game_id="2023_02_AAA_CCC", play_type="pass"),
    ]
    target_game = [
        _row(week=2, game_id="2023_02_AAA_CCC", play_type="pass", first_down=1),
        _row(week=2, game_id="2023_02_AAA_CCC", play_type="run", touchdown=1),
    ]
    df = pl.DataFrame([*season_warmup, *game_warmup, *target_game])

    out = build_game_replay(df, "2023_02_AAA_CCC")

    assert out["game_id"].to_list() == ["2023_02_AAA_CCC"] * 2
    assert out.sort("play_id")["play_id"].to_list() == sorted(out["play_id"].to_list())
    assert out[PLAY_TYPE_COLUMN].to_list() == ["pass", "run"]
    assert out[OUTCOME_COLUMN].to_list() == ["first_down", "touchdown"]
    assert out["posteam"].to_list() == ["AAA"] * 2
    assert out["desc"].to_list() == ["some play description"] * 2


def test_replay_unknown_game_id_returns_empty():
    warmup = [_row(play_type="run"), _row(play_type="pass")]
    df = pl.DataFrame([*warmup, _row(play_type="run")])

    out = build_game_replay(df, "does_not_exist")

    assert out.shape[0] == 0
