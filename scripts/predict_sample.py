"""Run this file to predict on a sample situation using the saved models.

No training happens here - this only loads what scripts/train_baseline.py
and scripts/train_outcome_baseline.py already saved to artifacts/. Run
those two first if artifacts/ is empty.
"""

from models.persistence import load_xgb_model
from models.predict import predict

# 3rd and 2, opponent's 5 yard line, tied game, 4th quarter, under 2 minutes,
# both teams have all their timeouts, home team has the ball.
SITUATION = {
    "down": 3,
    "ydstogo": 2,
    "yardline_100": 5,
    "score_differential": 0,
    "qtr": 4,
    "half_seconds_remaining": 110,
    "posteam_timeouts_remaining": 3,
    "defteam_timeouts_remaining": 3,
    "is_home": 1,
}


def _print_prediction(title: str, name: str) -> None:
    saved = load_xgb_model(name)
    probabilities = predict(saved, SITUATION)

    print(title)
    for label, prob in sorted(probabilities.items(), key=lambda item: -item[1]):
        print(f"  {label:<15} {prob:.1%}")
    print()


def main() -> None:
    print(f"Situation: {SITUATION}")
    print()
    _print_prediction("Play type:", "play_type")
    _print_prediction("Outcome:", "outcome")


if __name__ == "__main__":
    main()
