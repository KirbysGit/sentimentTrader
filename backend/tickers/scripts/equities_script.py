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

    print(f"Reduced from ~160K eqs to {len(df)} after our sort.")
    # save metadata (including symbol) as CSV
    df.to_csv(out_path_obj, index=False)

if __name__ == "__main__":
    major_exchanges = [
        # US
        "NMS", "NGM", "NCM", "NAS", "NYQ", "NYS", "ASE", "PCX",  # skip "PNK" unless you want OTC noise

        # Germany (Siemens, etc.)
        "FRA", "GER", "MUN", "DUS", "STU", "BER", "HAM", "HAN",

        # Taiwan (TSMC)
        "TAI",

        # Japan
        "JPX",

        # Hong Kong
        "HKG",

        # UK
        "LSE", "IOB",

        # Canada
        "TOR", "VAN", "CNQ", "CSE",

        # France / Benelux
        "PAR", "ENX", "AMS"
    ]

    build_equity_universe(
        exchanges=major_exchanges,
        country=None,  # allow international
        only_primary=True,
        out_path="data/ticker_universe.csv",
        keep_columns=("symbol", "name", "exchange", "sector", "industry"),
    )