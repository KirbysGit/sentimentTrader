import financedatabase as fd
from pathlib import Path

def build_equity_universe(
    exchanges=None,
    country=None,
    only_primary=True,
    out_path="./data/ticker_universe.csv",
    keep_columns=("symbol", "name", "exchange"),
):
    eq = fd.Equities()
    df = eq.select(
        exchange=exchanges,
        country=country,
        only_primary_listing=only_primary,
    )

    # drop rows without symbols
    df = df[df.index.notna()]
    df.index.name = "symbol"
    df = df.reset_index()

    # keep a few columns if you want
    df = df.loc[:, [c for c in keep_columns if c in df.columns]]

    symbols = df["symbol"].dropna().astype(str).str.upper().unique()
    # ensure output directory exists
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    # save metadata (including symbol) as CSV
    df.to_csv(out_path_obj, index=False)

if __name__ == "__main__":
    # Add US plus major international exchanges
    major_exchanges = [
        "NMS", "NGM", "NCM", "NAS", "NYS", "ASE",      # US (NASDAQ/NYSE/AMEX variants)
        "GER", "FRA", "BER", "MUN",                    # Germany (for Siemens, etc.)
        "LSE",                                         # UK
        "HKG",                                         # Hong Kong
        "JPX",                                         # Japan
        "ASX",                                         # Australia
        "CNQ",                                  # Canada
    ]

    build_equity_universe(
        exchanges=major_exchanges,
        country=None,  # allow international
        only_primary=True,
        out_path="data/ticker_universe.csv",
        keep_columns=("symbol", "name", "exchange", "sector", "industry"),
    )