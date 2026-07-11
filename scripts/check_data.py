"""Run this file to confirm historical data access is working.

Pulls one season of play-by-play from nflverse and prints a summary.
Requires the package to be installed first: pip install -e ".[dev]"
"""

from ingestion.nflverse import load_pbp

SEASON = 2023


def main() -> None:
    print(f"Loading {SEASON} play-by-play from nflverse...")
    df = load_pbp([SEASON])

    print()
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Games: {df['game_id'].n_unique()}")
    weeks = sorted(df["week"].drop_nulls().unique().to_list())
    print(f"Weeks: {weeks[0]}-{weeks[-1]}")

    print()
    print("Play type counts:")
    counts = df["play_type"].value_counts().sort("count", descending=True)
    for play_type, count in counts.iter_rows():
        label = play_type or "(none)"
        print(f"  {label:<15} {count:>6,}")

    print()
    print("Sample rows:")
    cols = ["week", "posteam", "play_type", "down", "ydstogo", "yards_gained"]
    for row in df.select(cols).head(10).iter_rows(named=True):
        play_type = row["play_type"] or "(none)"
        posteam = row["posteam"] or "---"
        print(
            f"  wk{row['week']:<3} {posteam:<4} {play_type:<10} "
            f"down={row['down']} ydstogo={row['ydstogo']} gain={row['yards_gained']}"
        )

if __name__ == "__main__":
    main()
