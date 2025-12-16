# 


# imports.
import re
import pandas as pd
from pathlib import Path
from colorama import Fore, Style

# local imports.
from src.utils.ticker_universe import TICKER_UNIVERSE
from src.utils.config import COMMON_WORDS

class RedditProcessor:

    def __init__(self, input_file: Path):
        self.input_file = input_file
        self.ticker_universe = TICKER_UNIVERSE
        self.common_words = 

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
    
    def extract_tickers(self, text, title): 

        RAW_REGEX = r"(?<![A-Za-z0-9\$])[A-Z]{1,5}(?![A-Za-z0-9])"
        DOLLAR_REGEX = r"(?<![A-Za-z0-9\$])\$[A-Za-z]{1,5}(?![A-Za-z0-9])"
        
        tickers = set()

        # -- 1. dollar-style tickers. ($AAPL)
        for match in re.finditer(DOLLAR_REGEX, text):
            if not self.has_clean_boundary(text, match.start(), match.end()):
                continue
            tickers.add(match.group().replace("$", "").upper())

        # -- 2. raw uppercase tickers. (AAPL, NVDA)
        for match in re.finditer(RAW_REGEX, text):
            if not self.has_clean_boundary(text, match.start(), match.end()):
                continue
            tickers.add(match.group().upper())

        universe_only = set()

        # -- 3. ticker universe check.
        for ticker in tickers:
            cleaned = ticker.strip().upper()
            if cleaned in self.ticker_universe:
                universe_only.add(cleaned)

        return universe_only


    
    def process_row(self, row: pd.Series):
        
        # -- 1. grab the title and text for extraction.
        title_col = row.get("title", "")
        text_col = row.get("text", "")
        title = "" if not isinstance(title_col, str) else title_col
        text = "" if not isinstance(text_col, str) else text_col
        
        # -- 2. extract tickers.
        tickers = self.extract_tickers(text, title)

        # -- 3. context checking.




    def process(self) -> pd.DataFrame:

            # -- 1. get the raw file from today.

            raw_file = Path(self.input_file)

            # -- 2. read the raw file.

            df = pd.read_csv(raw_file)
            self.raw_count = len(df)
            print(f"{Fore.GREEN} grabbed {self.raw_count} raw posts from today's parsing.{Style.RESET_ALL}")
            
            # -- 3. iterate through the posts.
            for _, row in df.iterrows():
                self.process_row(row)
                
