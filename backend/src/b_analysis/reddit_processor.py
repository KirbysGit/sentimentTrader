# 
# imports.
import re
import pandas as pd
from pathlib import Path
from colorama import Fore, Style

# local imports.
from src.utils.config import COMMON_WORDS
from src.utils.path_config import tickers_dir

class RedditProcessor:

    # --- self-initialize.

    def __init__(self, input_file: Path):

        # -- 1. get input file from phase 1.
        self.input_file = input_file

        # -- 2. set up data from etfs and equities we want to reference.
        self.etf_universe = pd.read_csv(tickers_dir / "etfs.csv")
        self.equity_universe = pd.read_csv(tickers_dir / "equities.csv")

    # --- helper methods.

    @staticmethod
    def has_clean_boundary(text, start, end):
        # ensure match is not in middle of a word.
        def is_valid_prev(char):
            if not char:
                return True
            return not char.isalnum() and char != "_"
        def is_valid_next(char):
            if not char:
                return True
            return not char.isalnum() and char != "_"

        prevChar = text[start - 1] if start > 0 else ""
        nextChar = text[end] if end < len(text) else ""

        return is_valid_prev(prevChar) and is_valid_next(nextChar)
    

    # --- main processing methods ---

    def extract_tickers(self, text, title): 

        RAW_REGEX = r"(?<![A-Za-z0-9\$])[A-Z]{1,5}(?![A-Za-z0-9])"              # set up raw ticker regex. (TSLA)
        DOLLAR_REGEX = r"(?<![A-Za-z0-9\$])\$[A-Za-z]{1,5}(?![A-Za-z0-9])"      # set up dollar ticker regex. ($TSLA)
        
        tickers = set()

        for match in re.finditer(DOLLAR_REGEX, text):                           # -- 1. dollar-style tickers. ($AAPL)
            if not self.has_clean_boundary(text, match.start(), match.end()):   # if not a clean boundary.
                continue                                                        # skip it.
            tickers.add(match.group().replace("$", "").upper())                 # if so, add ticker but get rid of $.

        for match in re.finditer(RAW_REGEX, text):                              # -- 2. raw uppercase tickers. (AAPL, NVDA)
            if not self.has_clean_boundary(text, match.start(), match.end()):   # if not a clean boundary.
                continue                                                        # skip it.
            tickers.add(match.group().upper())                                  # if so, add ticker.

        universe = set()

        for ticker in tickers:                                                  # -- 3. equity universe check.
            cleaned = ticker.strip().upper()                                    # clean ticker val.
            if cleaned in self.equity_universe["symbol"].values:                # check if it exists in equity.
                universe.add(cleaned)                                           # if so, add to cleaned.

        for ticker in tickers:                                                  # -- 4. etf universe check.
            cleaned = ticker.strip().upper()                                    # clean ticker val.
            if cleaned in self.etf_universe["symbol"].values:                   # check if it exists in etf.
                universe.add(cleaned)                                           # if so, add to cleaned.

        return universe

    def process_row(self, row: pd.Series):
        # -- 1. grab the title and text for extraction.
        title_col = row.get("title", "")
        text_col = row.get("text", "")
        title = "" if not isinstance(title_col, str) else title_col
        text = "" if not isinstance(text_col, str) else text_col
        
        # -- 2. extract tickers.
        tickers = self.extract_tickers(text, title)
        print(tickers)
        # -- 3. context checking.


    def process(self) -> pd.DataFrame:
s
            # -- 1. get the raw file from today.
            raw_file = Path(self.input_file)

            # -- 2. read the raw file.
            df = pd.read_csv(raw_file)
            self.raw_count = len(df)
            
            # -- 3. iterate through the posts.
            for _, row in df.iterrows():
                self.process_row(row)
                
