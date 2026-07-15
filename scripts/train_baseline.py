"""Run this file to train and evaluate the play-type baseline models.

Pulls historical play-by-play, builds the situational feature set, and
reports accuracy/log-loss/confusion matrix for a season-based holdout.
Requires the package to be installed first: pip install -e ".[dev]"
"""

from features.situational import FEATURE_COLUMNS, PLAY_TYPES, build_dataset
from ingestion.nflverse import load_pbp
from models.persistence import save_xgb_model
from models.play_type_baseline import train_and_evaluate

TRAIN_SEASONS = list(range(2016, 2023))  # 2016-2022
TEST_SEASON = 2023


def main() -> None:
    seasons = [*TRAIN_SEASONS, TEST_SEASON]
    print(f"Loading {seasons[0]}-{seasons[-1]} play-by-play from nflverse...")
    pbp = load_pbp(seasons)

    dataset = build_dataset(pbp)
    print(f"Usable plays after filtering: {dataset.shape[0]:,}")

    print()
    print("Label distribution:")
    counts = dataset["play_type"].value_counts().sort("count", descending=True)
    for play_type, count in counts.iter_rows():
        print(f"  {play_type:<12} {count:>7,}")

    print()
    print(f"Training on {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}, testing on {TEST_SEASON}...")
    results = train_and_evaluate(dataset, test_season=TEST_SEASON)

    for result in results:
        print()
        print(f"=== {result.name} ===")
        print(f"Accuracy: {result.accuracy:.3f}")
        print(f"Log loss: {result.log_loss:.3f}")
        print()
        print(result.report)
        print(f"Confusion matrix (rows=actual, cols=predicted), order={PLAY_TYPES}:")
        print(result.confusion)

    xgb_result = next(r for r in results if r.name == "XGBoost")
    save_xgb_model(xgb_result.model, "play_type", FEATURE_COLUMNS, sorted(PLAY_TYPES))
    print()
    print("Saved XGBoost model to artifacts/play_type.xgb.json")


if __name__ == "__main__":
    main()
