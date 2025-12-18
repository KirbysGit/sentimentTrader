from pathlib import Path

import pandas as pd


def main():
    # locate full_equities.csv relative to this script
    tickers_dir = Path(__file__).resolve().parent.parent
    csv_path = tickers_dir / "full_equities.csv"

    if not csv_path.exists():
        raise SystemExit(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)
    if "exchange" not in df.columns:
        raise SystemExit("No 'exchange' column found in full_equities.csv")

    exchanges = (
        df["exchange"]
        .astype(str)
        .str.strip()
        .replace({"nan": ""})
    )

    counts = exchanges.value_counts(dropna=False).sort_index()

    print(f"Unique exchanges in {csv_path.name}: {len(counts)}\n")
    for exch, cnt in counts.items():
        label = exch or "<empty>"
        print(f"{label:20s} {cnt}")


if __name__ == "__main__":
    main()

