import polars as pl

from features.tendency import add_tendency_features


def test_rolling_features_are_leak_free_and_correct():
    # Team A on offense throughout, vs team B on defense, in play order.
    df = pl.DataFrame(
        {
            "season": [2023] * 4,
            "week": [1, 1, 1, 1],
            "game_id": ["g1", "g1", "g1", "g1"],
            "play_id": [1, 2, 3, 4],
            "posteam": ["A", "A", "A", "A"],
            "defteam": ["B", "B", "B", "B"],
            "down": [1, 2, 1, 1],
            "play_type": ["run", "pass", "run", "punt"],
            "epa": [0.5, -0.2, 1.0, 0.1],
        }
    )

    out = add_tendency_features(df)
    rows = out.sort(["week", "play_id"]).to_dicts()

    # Play 1: no prior plays at all for team A - everything null.
    assert rows[0]["posteam_run_rate"] is None
    assert rows[0]["posteam_run_rate_this_down"] is None
    assert rows[0]["defteam_epa_allowed_rush"] is None

    # Play 2 (down=2): 1 prior play (the run) -> run rate 1/1.
    assert rows[1]["posteam_run_rate"] == 1.0
    # No prior play on down=2 specifically.
    assert rows[1]["posteam_run_rate_this_down"] is None
    # 1 prior rush faced by B, epa 0.5 -> average 0.5.
    assert rows[1]["defteam_epa_allowed_rush"] == 0.5
    # No prior pass faced by B yet.
    assert rows[1]["defteam_epa_allowed_pass"] is None

    # Play 3 (down=1): 2 prior plays (run, pass) -> run rate 1/2.
    assert rows[2]["posteam_run_rate"] == 0.5
    # 1 prior play on down=1 (the first run) -> this-down rate 1/1.
    assert rows[2]["posteam_run_rate_this_down"] == 1.0

    # Play 4 (punt): doesn't count toward any rate itself, but still
    # carries the tendency values computed from the 3 plays before it.
    assert rows[3]["posteam_run_rate"] == 2 / 3
    # Prior down=1 plays: play 1 and play 3, both runs -> 2/2.
    assert rows[3]["posteam_run_rate_this_down"] == 1.0
    # Prior rush plays faced by B: play 1 (0.5) and play 3 (1.0) -> avg 0.75.
    assert rows[3]["defteam_epa_allowed_rush"] == 0.75
    # Prior pass plays faced by B: play 2 (-0.2) -> avg -0.2.
    assert rows[3]["defteam_epa_allowed_pass"] == -0.2


def test_teams_and_seasons_do_not_leak_into_each_other():
    # g1: A's 1st play of 2023 (run). g2: B's 1st play of 2023 (pass) -
    # must not see A's run. g3: A's 1st play of *2024* (pass) - must not
    # see A's 2023 history. g4: A's 2nd play of 2023 (run) - should see
    # only g1's run.
    df = pl.DataFrame(
        {
            "season": [2023, 2023, 2024, 2023],
            "week": [1, 1, 1, 1],
            "game_id": ["g1", "g2", "g3", "g4"],
            "play_id": [1, 1, 1, 1],
            "posteam": ["A", "B", "A", "A"],
            "defteam": ["X", "Y", "X", "Y"],
            "down": [1, 1, 1, 1],
            "play_type": ["run", "pass", "pass", "run"],
            "epa": [0.5, -0.2, 1.0, 0.1],
        }
    )

    out = add_tendency_features(df).sort(["season", "game_id"])
    by_game = {row["game_id"]: row for row in out.to_dicts()}

    assert by_game["g1"]["posteam_run_rate"] is None
    assert by_game["g2"]["posteam_run_rate"] is None
    assert by_game["g3"]["posteam_run_rate"] is None
    assert by_game["g4"]["posteam_run_rate"] == 1.0
