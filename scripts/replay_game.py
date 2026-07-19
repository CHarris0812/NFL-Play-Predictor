"""Run this file to simulate a live game using a completed one.

Predicts each play before revealing what actually happened, using the
same models saved by scripts/train_baseline.py / train_outcome_baseline.py
- run those first if artifacts/ is empty.

Usage:
  python scripts/replay_game.py                       # Super Bowl LVIII
  python scripts/replay_game.py 2023_01_DET_KC         # any 2023 game
  python scripts/replay_game.py 2023_01_DET_KC 2       # with a 2s delay/play
"""

import sys

from serving.replay import run_replay

DEFAULT_GAME_ID = "2023_22_SF_KC"  # Super Bowl LVIII


def main() -> None:
    game_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GAME_ID
    delay_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    season = int(game_id.split("_")[0])
    run_replay(game_id, season, delay_seconds=delay_seconds)


if __name__ == "__main__":
    main()
