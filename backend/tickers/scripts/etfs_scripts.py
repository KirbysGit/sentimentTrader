# etfs_script.py
from pathlib import Path
import financedatabase as fd

def build_etf_universe(
    exchanges=None,          # e.g., ["NMS","NGM","NCM","NAS","NYS","ASE"]
    only_primary=True,
    out_path="./etf_universe.csv",
    keep_columns=("symbol", "name", "exchange", "category_group", "category"),
):
    etf = fd.ETFs()
    df = etf.select(
        exchange=exchanges,
        only_primary_listing=only_primary,
    )

    # drop rows without symbols, reset index
    df = df[df.index.notna()]
    df.index.name = "symbol"
    df = df.reset_index()

    df = df.loc[:, [c for c in keep_columns if c in df.columns]]

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path_obj, index=False)
    print(f"wrote {len(df)} ETFs → {out_path_obj}")

if __name__ == "__main__":
    # US-centric plus a few major international codes that exist in financedatabase
    major_exchanges = [
        "NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "NEO",  # US/NA
        "LSE", "FRA", "GER", "MUN", "JPX", "ASX", "TOR", "TAI",
    ]
    build_etf_universe(
        exchanges=major_exchanges,
        only_primary=True,
        out_path="./data/etf_universe.csv",
    )