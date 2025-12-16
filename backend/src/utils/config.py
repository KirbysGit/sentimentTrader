# main pipeline configuration file

# ============================================================================
# stage 1: reddit collection config
# ============================================================================

# high-signal subreddits for stock prediction (organized by priority)
HIGH_SIGNAL_MUST_HAVE = [
    'wallstreetbets',  # extreme sentiment, high volume
]

#'stocks',          # general stock discussion
#'investing',       # investment-focused discussions
#'StockMarket'      # market-wide discussions

HIGH_SIGNAL_ADDITIONAL = [
    'news',            # general news (can affect markets)
    'worldnews',       # global news (market impacts)
    'finance',         # broader finance discussions
    'technology',      # tech sector sentiment
    'cryptocurrency',  # overlapping sentiment trends
    'pennystocks'      # extreme sentiment spikes
]

# active production subreddits (currently using must-have only)
SUBREDDITS = HIGH_SIGNAL_MUST_HAVE.copy()
SORT_METHODS = ['new', 'top', 'hot']  # collects from all 3 sort methods

# test mode config (faster for testing - single subreddit, single sort)
TEST_SUBREDDITS = ['wallstreetbets']
TEST_SORT_METHODS = ['hot']

# collection math:
# - production: 4 subreddits × 3 sorts × 100 posts = up to 1,200 posts
# - test: 1 subreddit × 1 sort × 10 posts = 10 posts
# - with additional subs: 10 subreddits × 3 sorts × 100 posts = up to 3,000 posts


# ============================================================================
# stage 2: ticker analysis config
# ============================================================================

# basic ticker config
TICKERS = ['NVDA', 'NVIDIA', 'AMD', 'INTC', 'TSMC']
MAX_TEXT_LENGTH = 500
SUMMARY_LENGTH = 100

# noise filters for ticker extraction
MACRO_TERMS = {
    'GDP', 'CPI', 'PPI', 'FOMC', 'FED', 'YCC', 'VAT', 'RATE', 'TAX',
    'INFLATION', 'YIELD', 'TREASURY', 'JOBS', 'ENERGY', 'OPEC', 'ECB',
    'WWII', 'AI', 'TL', 'DR'
}

WSB_SLANG = {
    'DD', 'YOLO', 'IV', 'TA', 'FA', 'ATH', 'OTM', 'ITM', 'RH', 'FOMO',
    'FD', 'STONK', 'GANG', 'MOASS', 'HF', 'BAG', 'PUMP', 'DUMP'
}

CONTEXT_REQUIRED_TICKERS = {
    'AI', 'GDP', 'VAT', 'YCC', 'TL', 'DR', 'GOLD'
}

# comprehensive blacklist covering slang, macro terms, finance acronyms, TA words
WSB_FINANCE_BLACKLIST = {
    "A", "ADHD", "ADV", "AGAIN", "AGI", "AHH", "AI", "ALL", "AND", "ANY", "AOV", "API",
    "ARE", "ASI", "ATH", "ATM", "AWS", "BBC", "BNPL", "BREAK", "BREAKOUT", "BS", "BULL",
    "BUT", "CALL", "CALLS", "CAN", "CAPEX", "CARE", "CAT", "CBO", "CEO", "CFO", "CHAD", "CHEAP",
    "CNBC", "COO", "COULD", "COVID", "CPI", "CPU", "CUDA", "DCA", "DD", "DCF", "DEAD", "DELTA",
    "DO", "DOWN", "DR", "DUMP", "DYOR", "EDA", "EBIT", "EBITDA", "EMH", "EPS", "ESOP", "ESPN",
    "ET", "ETF", "ETFS", "EV", "EU", "FAQ", "FAILS", "FBI", "FCF", "FDA", "FIB", "FOMO",
    "FOMU", "FSD", "FUD", "GAIN", "GAINZ", "GAMMA", "GBP", "GEX", "GFC", "GG", "GMV", "GOD",
    "GO", "GREAT", "GREEN", "GPU", "GPT", "HBO", "HCOL", "HIGH", "HODL", "HOT", "HOW", "HSA",
    "HUGE", "HYSA", "III", "IMF", "IN", "IRA", "IRS", "ITM", "IV", "IVR", "JOB", "JPY",
    "LARGE", "LEAP", "LEAPS", "LEFT", "LINE", "LLC", "LMAO", "LOL", "LOSS", "LOW", "LTCG", "LTV",
    "MACD", "MAG", "MAANG", "MAY", "MIT", "ML", "MOASS", "MORE", "MUCH", "MUST", "NAV", "NASA",
    "NATO", "NEXT", "NIM", "NOT", "NOW", "NTM", "NSSL", "OF", "OI", "ON", "ONE", "OS",
    "OTC", "OTCQB", "OTCPK", "OTM", "OUT", "P", "PAT", "PBT", "PCE", "PE", "PIVOT", "PLR",
    "PM", "PRESS", "PS", "PUMP", "PUT", "PUTS", "QOQ", "QE", "QT", "RED", "RESISTANCE", "RETURN",
    "RFK", "RIGHT", "ROA", "ROE", "ROI", "ROTH", "RSI", "S", "SAFE", "SCALP", "SEC", "SHOULD",
    "SHORT", "SINCE", "SMA", "STONKS", "STOCK", "STRIKE", "SUPPORT", "T", "TENDIES", "THAT", "THE", "THEM",
    "THEY", "THETA", "THICC", "THREE", "THIS", "TIME", "TILTS", "TL", "TL;DR", "TLDR", "TOKYO", "TREND",
    "TURN", "TWO", "UAE", "UK", "UP", "US", "USA", "USD", "VOTER", "VWAP", "WAS", "WEEKS",
    "WENDY", "WHAT", "WHY", "WILL", "WTF", "WSB", "WSJ", "YCC", "YOLO", "YOY", "ZERO", "ZIRP",
    "XXXXX",
}

# unified blocklist used across stages
BLOCKLIST = WSB_FINANCE_BLACKLIST

# extra stopwords removed after stage 2 to avoid false positives
FINAL_STAGE_STOPWORDS = {
    "YES", "RE", "PART", "MODE", "MINE", "JUICY", "STILL", "SETUP", "RJ", "MATH",
    "OPEN", "ZERO", "GOD", "BIBLE", "DID", "HYSA", "STOCK",
}

# subreddit to ticker mapping (all lowercase keys)
SUBREDDIT_TICKERS = {
    'nvidia': 'NVDA',
    'amd': 'AMD',
    'intel': 'INTC',
    'tsmc': 'TSMC',
    'wallstreetbets': None,
    'stocks': None,
    'investing': None,
    'stockmarket': None
}

# financial context words (merged strong + weak)
FINANCE_CONTEXT_WORDS = {
    'stock', 'shares', 'ticker', 'earnings', 'revenue', 'dividend', 'market cap',
    'trading', 'investor', 'bullish', 'bearish', '$', 'calls', 'puts', 'options',
    'portfolio', 'shareholders', 'eps', 'pe ratio', 'market share', 'guidance',
    'analyst', 'upgrade', 'downgrade', 'price target', 'short interest', 'float',
    'institutional', 'hedge fund', 'etf', 'ipo', 'spac', 'merger', 'acquisition',
    'sold', 'selling', 'dumped', 'dumping', 'trim', 'trimmed', 'bagged', 'bagging',
    'positioned', 'positions', 'averaged', 'scaling', 'volatility', 'assets',
    'forecast', 's&p', 'sp500',
    'buy', 'sell', 'price', 'trade', 'invest', 'market', 'position',
    'profit', 'loss', 'analysis', 'company', 'corporation', 'inc', 'ltd', 'tech',
    'up', 'down', 'gain', 'drop', 'rise', 'fall', 'quarter', 'growth', 'decline',
    'performance', 'trend', 'sector', 'industry', 'competition', 'partnership',
    'deal', 'contract', 'launch', 'product', 'service', 'expansion', 'strategy',
    'turbulence', 'turbulences',
}

# sentiment lexicon used by SentimentScorer (extend freely)
POSITIVE_SENTIMENT_WORDS = {
    "up", "bull", "bullish", "gain", "green", "beat", "pump", "moon", "mooning",
    "strong", "win", "positive", "profit", "surge", "soar", "rocket", "pump",
    "rip", "squeeze", "run", "ath", "momentum", "breakout", "climb", "pumpage",
    "jump", "skyrocket", "crush", "smash", "crank", "double", "tripled", "explode"
}

NEGATIVE_SENTIMENT_WORDS = {
    "down", "bear", "bearish", "loss", "dump", "crash", "bad", "miss", "weak",
    "negative", "red", "selloff", "plunge", "tank", "bleed", "collapse", "bag",
    "rug", "rugged", "sink", "dumped", "dropped", "halved", "wrecked", "implode",
    "panic", "sell", "sold", "fear", "beartrap"
}

# common words that might be mistaken for tickers
COMMON_WORDS = {
    "ABOUT", "AFTER", "ALL", "ALSO", "AND", "ARE", "BACK", "BEEN",
    "BEING", "BEST", "BUT", "CAN", "CASE", "COME", "COULD", "DAYS",
    "EVEN", "FACT", "FEAST", "FIND", "FIRST", "FOR", "FROM", "GIVE",
    "GOOD", "GREAT", "HAD", "HAND", "HAS", "HAVE", "HEAD", "HELP",
    "HERE", "HER", "HIS", "HOW", "IDEA", "INTO", "ITS", "JUST",
    "KIND", "KNOW", "LAST", "LIFE", "LIKE", "LINE", "LIVE", "LONG",
    "LOOK", "MAKE", "MAN", "MANY", "MAY", "MIGHT", "MORE", "MOST",
    "MOVE", "MUCH", "MUST", "NAME", "NEED", "NEVER", "NEXT", "NOT",
    "NOW", "OF", "OUR", "OUT", "OVER", "PART", "POST", "SAID",
    "SAY", "SEVEN", "SHALL", "SHOULD", "SOME", "SUCH", "SURE", "TAKE",
    "TALK", "TELL", "THAN", "THAT", "THE", "THEIR", "THEM", "THEN",
    "THERE", "THESE", "THEY", "THINK", "THIS", "THOSE", "TIME", "TIS",
    "TURN", "TYPE", "WANT", "WAS", "WAY", "WEEK", "WELL", "WERE",
    "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHY", "WILL", "WITH",
    "WORK", "WOULD", "WORTH", "YEAR", "YOU",
}

# etf categories
ETF_CATEGORIES = {
    'MARKET_INDEX': {
        'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI',
        'SPXL', 'TQQQ',
    },
    'SECTOR': {
        'XLF', 'XLE', 'XLV', 'XLK', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE', 'XLC'
    },
    'COMMODITY': {
        'GLD', 'SLV', 'USO', 'UNG', 'PHYS', 'URA'
    },
    'BOND': {
        'TLT', 'IEF', 'HYG', 'LQD', 'AGG', 'BND', 'VHYG', 'TIPS',
        'FLOT', 'SGOV', 'VGSH',
    },
    'INTERNATIONAL': {
        'EFA', 'EEM', 'VEA', 'VWO', 'VGK', 'VEQT', 'VXUS'
    },
    'CRYPTO': {
        'IBIT', 'BITB', 'BTCI'
    }
}

TIMESTAMP_TICKERS = {
    "ET", "EST", "EDT", "CT", "CST", "CDT", "PT", "PST", "PDT", "MT", "MST", "MDT", "UTC", "GMT"
}

# flatten etf list for quick lookup
VALID_ETFS = {etf for category in ETF_CATEGORIES.values() for etf in category}

# unified allow list (etfs + special non-universe symbols)
ALWAYS_ALLOW = VALID_ETFS | {"BTC", "ETH", "SPX", "GOLD", "VXUS"}

# well-known stock tickers
WELL_KNOWN_TICKERS = {
    "AAPL", "ACHR", "ADP", "AMC", "AMD", "AMZN", "ANET", "ANF",
    "ASTS", "AVGO", "BABA", "BAC", "BA", "BB", "BE", "BILI",
    "BUD", "CMCSA", "COIN", "CRWD", "CSCO", "CVS", "DASH",
    "DIS", "DKNG", "DLTR", "EDU", "EDIT", "ESNT", "FDVV",
    "F", "FSKAX", "FSPGX", "GE", "GGLL", "GM", "GME", "GOOG",
    "GOOGL", "HOOD", "IBM", "INTC", "IREN", "JNJ", "JPM", "KO",
    "KSS", "LLY", "LMT", "LSEG", "MAGA", "MCD", "META", "MSTR",
    "MRK", "MSCI", "MSFT", "NEE", "NFLX", "NOC", "NQ", "NVDA",
    "OKLO", "ORCL", "PATH", "PEP", "PDD", "PFE", "PHYS", "PLD",
    "PLTR", "RDDT", "RGTI", "RGTZ", "RKLB", "RTX", "SCHD", "SEZL",
    "SMR", "SOND", "T", "TAL", "TSLA", "UNH", "URA", "UUUU",
    "VEQT", "VIGAX", "VHYG", "VNYTX", "VOO", "VYMI", "VZ",
    "WB", "WBD", "WFC", "XOVR",
}

# negative context patterns that invalidate ticker matches
NEGATIVE_CONTEXT_PATTERNS = {
    'COIN': ['meme coin', 'shit coin', 'shitcoin', 'alt coin', 'altcoin', 'stable coin', 'stablecoin', 'dog coin', 'dogcoin', 'moon coin', 'mooncoin', 'pump coin', 'dump coin', 'new coin', 'this coin', 'the coin', 'that coin', 'any coin', 'my coin', 'your coin', 'their coin', 'crypto coin', 'cryptocurrency', 'token'],
    'GOLD': ['gold standard', 'gold medal', 'gold mine', 'gold rush', 'gold price'],
    'GOOD': ['good morning', 'good night', 'good day', 'good luck', 'good job', 'good news', 'good boy'],
    'CASH': ['cash app', 'cash out', 'cash flow', 'cash back', 'cash money'],
    'MOON': ['to the moon', 'moon shot', 'moon boy', 'moon mission'],
    'PUMP': ['pump and dump', 'pump scheme', 'pump group'],
    'HOLD': ['hold on', 'hold up', 'hold tight', 'hold steady'],
    'GAS': ['gas price', 'gas fee', 'gas station', 'gas tank'],
    'DASH': [' - ', '--', '—', ' – '],
    'BOT': ['robot', 'bot army', 'chatbot'],
    'ACA': ['affordable care act', 'aca credits', 'aca subsidies'],
}

# ambiguous financial tickers that need extra validation
AMBIGUOUS_FINANCIAL_TICKERS = {
    'COIN': {
        'required_context': ['coinbase', 'nasdaq:coin', 'nyse:coin'],
        'company_terms': ['coinbase', 'armstrong', 'crypto exchange', 'cryptocurrency exchange'],
        'min_confidence': 0.9
    },
    'GOLD': {
        'required_context': ['barrick', 'barrick gold', 'gld etf', 'gold etf', 'gold shares'],
        'company_terms': ['barrick', 'spdr', 'state street', 'gold trust'],
        'min_confidence': 0.8
    },
    'CASH': {
        'required_context': ['money market', 'cash management'],
        'company_terms': ['money market fund', 'cash equivalent'],
        'min_confidence': 0.9
    }
}

