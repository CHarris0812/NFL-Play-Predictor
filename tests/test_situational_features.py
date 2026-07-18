import polars as pl

from features.situational import FEATURE_COLUMNS, LABEL_COLUMN, build_dataset


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
    }
    row.update(overrides)
    return row


def test_keeps_real_scrimmage_plays():
    df = pl.DataFrame([_row(play_type="run"), _row(play_type="pass")])

    out = build_dataset(df)

    assert out.shape[0] == 2
    assert set(out[LABEL_COLUMN].to_list()) == {"run", "pass"}


def test_drops_no_play_kickoff_and_kneel_spike():
    df = pl.DataFrame(
        [
            _row(play_type="run"),
            _row(play_type="no_play"),
            _row(play_type="kickoff", down=None),
            _row(play_type="extra_point", down=None),
            _row(play_type="qb_kneel"),
            _row(play_type="qb_spike"),
        ]
    )

    out = build_dataset(df)

    assert out.shape[0] == 1
    assert out[LABEL_COLUMN].to_list() == ["run"]


def test_drops_rows_with_null_features():
    df = pl.DataFrame([_row(play_type="run"), _row(play_type="pass", score_differential=None)])

    out = build_dataset(df)

    assert out.shape[0] == 1


def test_is_home_derived_from_posteam_type():
    df = pl.DataFrame([_row(posteam_type="home"), _row(posteam_type="away")])

    out = build_dataset(df)

    assert sorted(out["is_home"].to_list()) == [False, True]


def test_output_has_expected_columns():
    df = pl.DataFrame([_row()])

    out = build_dataset(df)

    for col in [*FEATURE_COLUMNS, LABEL_COLUMN]:
        assert col in out.columns
