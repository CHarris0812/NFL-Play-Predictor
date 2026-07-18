import polars as pl

from features.outcome import FEATURE_COLUMNS, LABEL_COLUMN, build_dataset


def _row(**overrides):
    row = {
        "season": 2023,
        "week": 1,
        "game_id": "2023_01_AAA_BBB",
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
        "touchdown": 0,
        "interception": 0,
        "fumble_lost": 0,
        "first_down": 0,
    }
    row.update(overrides)
    return row


def test_no_first_down_is_the_default():
    df = pl.DataFrame([_row()])

    out = build_dataset(df)

    assert out[LABEL_COLUMN].to_list() == ["no_first_down"]


def test_first_down_label():
    df = pl.DataFrame([_row(first_down=1)])

    out = build_dataset(df)

    assert out[LABEL_COLUMN].to_list() == ["first_down"]


def test_turnover_label_from_interception_or_fumble_lost():
    df = pl.DataFrame([_row(interception=1), _row(fumble_lost=1)])

    out = build_dataset(df)

    assert out[LABEL_COLUMN].to_list() == ["turnover", "turnover"]


def test_touchdown_takes_priority_over_turnover_and_first_down():
    df = pl.DataFrame([_row(touchdown=1, interception=1, first_down=1)])

    out = build_dataset(df)

    assert out[LABEL_COLUMN].to_list() == ["touchdown"]


def test_turnover_takes_priority_over_first_down():
    df = pl.DataFrame([_row(interception=1, first_down=1)])

    out = build_dataset(df)

    assert out[LABEL_COLUMN].to_list() == ["turnover"]


def test_reuses_shared_play_universe_filtering():
    df = pl.DataFrame([_row(play_type="run"), _row(play_type="no_play"), _row(play_type="kickoff", down=None)])

    out = build_dataset(df)

    assert out.shape[0] == 1


def test_output_has_expected_columns():
    df = pl.DataFrame([_row()])

    out = build_dataset(df)

    for col in [*FEATURE_COLUMNS, LABEL_COLUMN]:
        assert col in out.columns
