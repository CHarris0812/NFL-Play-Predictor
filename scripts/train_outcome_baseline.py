"""Run this file to train and evaluate the outcome baseline models.

Pulls historical play-by-play, builds the outcome feature set, and
reports accuracy/log-loss/confusion matrix for a season-based holdout.
Requires the package to be installed first: pip install -e ".[dev]"
"""

from features.outcome import OUTCOMES, build_dataset
from ingestion.nflverse import load_pbp
from models.outcome_baseline import train_and_evaluate

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
    counts = dataset["outcome"].value_counts().sort("count", descending=True)
    for outcome, count in counts.iter_rows():
        print(f"  {outcome:<15} {count:>7,}")

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
        print(f"Confusion matrix (rows=actual, cols=predicted), order={OUTCOMES}:")
        print(result.confusion)


if __name__ == "__main__":
    main()
