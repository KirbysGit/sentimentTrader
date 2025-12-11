# main pipeline configuration file

# ============================================================================
# stage 1: reddit collection config
# ============================================================================

# high-signal subreddits for stock prediction (organized by priority)
HIGH_SIGNAL_MUST_HAVE = [
    'wallstreetbets',  # extreme sentiment, high volume
    'stocks',          # general stock discussion
    'investing',       # investment-focused discussions
    'StockMarket'      # market-wide discussions
]

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
TEST_SORT_METHODS = ['new']

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
    "A", "AGAIN", "AHH", "AI", "ALL", "AND", "ANY", "ARE",
    "ATH", "ATM", "BREAK", "BREAKOUT", "BULL", "BUT",
    "CALL", "CALLS", "CAN", "CAGR", "CAT", "CHEAP", "CPI", "CPU",
    "COULD", "CUDA", "DD", "DEAD", "DELTA", "DIAMOND", "DUMP", "DCF",
    "DO", "DOWN", "EDA", "EBITDA", "EMA", "EPS", "ETFS",
    "ETF", "ET", "EV", "EU", "FIB", "FED", "FCF", "FAILS", "FOMO",
    "FOMU", "FUD", "GAIN", "GAINZ", "GAMMA", "GDP",
    "GG", "GO", "GREEN", "GPU", "HODL", "HIGH",
    "HOW", "HUGE", "IMF", "IN", "IRS", "ITM",
    "IV", "IVR", "JOB", "LEFT", "LEAP", "LEAPS",
    "LOSS", "LMAO", "LOL", "LOW", "MACD", "MAY",
    "ML", "MOASS", "MUST", "NAV", "NEXT", "NIM", "NOT",
    "NOW", "ON", "ONE", "OTM", "OUT", "OI", "P",
    "PAT", "PBT", "PE", "PCE", "PIVOT", "PRESS", "PUMP",
    "PUT", "PUTS", "QOQ", "QE", "QT", "RED",
    "RESISTANCE", "RETURN", "RIGHT", "ROA",
    "ROE", "ROI", "RSI", "SAFE", "SCALP",
    "SEC", "SHOULD", "SHORT", "SMA", "STONKS",
    "STRIKE", "SUPPORT", "S", "T", "TENDIES",
    "THE", "THEY", "THEM", "THAT", "THIS",
    "THETA", "THICC", "TLDR", "TPU", "TREND", "TWO",
    "UP", "UK", "VWAP", "WAS", "WENDY", "WILL", "WHAT", "WHY",
    "WTF", "YOLO", "YOY", "YCC",
    # pipeline-observed false tickers
    "BNPL", "CEO", "FDA", "WSB", "USD", "NTM",
    "OTCPK", "OTCQB", "OTC",
    # additional review false positives
    "US", "USA", "UAE", "EUR", "GBP", "JPY",
    "MIT", "WSJ", "CNBC", "NYT", "BBC", "NCIA",
    "COVID", "HSA", "IRA", "LTCG",
    "GPT", "API", "AWS", "FSD", "GEX", "ZIRP", "DCA", "EMH", "AGI", "ASI", "MAG",
    "HCOL", "ADV", "LLC", "PM", "OF", "SINCE",
    "HYSA",
    "EBIT",
    "AOV", "GMV", "LTV", "PLR", "CAPEX",
    "ADHD", "CHAD", "GOD", "ZERO", "TILTS",
    "STOCK", "HOT",
    "THREE", "WEEKS", "MINUS", "CARE", "NATO", "ANTH",
    "MAANG", "ROTH", "ESPN", "HBO", "FAQ", "FBI", "GFC", "ATP",
    "ESOP", "BS", "III", "PS", "LARGE",
    "VOTER", "MUCH", "MORE", "TOKYO", "TL", "TL;DR", "DR",
    "RFK", "NASA", "NSSL", "OS",
    "TIME", "LINE", "TURN", "GREAT", "TIS", "XXXXX",
    # newly observed false positives
    "DYOR",
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
    'microgrid', 'microgrids', 'grid', 'grid stability',
    'distributed energy', 'energy storage', 'industrial power',
    'batteries', 'battery', 'power applications',
    'buy', 'sell', 'price', 'trade', 'invest', 'market', 'position',
    'profit', 'loss', 'analysis', 'company', 'corporation', 'inc', 'ltd', 'tech',
    'up', 'down', 'gain', 'drop', 'rise', 'fall', 'quarter', 'growth', 'decline',
    'performance', 'trend', 'sector', 'industry', 'competition', 'partnership',
    'deal', 'contract', 'launch', 'product', 'service', 'expansion', 'strategy',
    'turbulence', 'turbulences',
    'storage', 'renewables', 'missile', 'missiles'
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
    'THE', 'AND', 'FOR', 'ARE', 'OF', 'WAS', 'YOU', 'HAS', 'HAD', 'HIS', 'HER', 'ITS', 'OUR', 'THEIR',
    'FROM', 'THIS', 'THAT', 'WITH', 'WHICH', 'WHEN', 'WHERE', 'WHAT', 'WHY', 'HOW', 'WHO',
    'CAN', 'MAN', 'POST', 'LIVE', 'HAS', 'HAD', 'WAS', 'WERE', 'BEEN', 'BEING', 'HAVE', 'HAS',
    'WILL', 'WOULD', 'SHALL', 'SHOULD', 'MAY', 'MIGHT', 'MUST', 'COULD', 'SHOULD', 'WOULD',
    'NOT', 'BUT', 'LIKE', 'MORE', 'JUST', 'NOW', 'OUT', 'ALL', 'THEY', 'SAID', 'TIME', 'ABOUT',
    'SOME', 'INTO', 'ALSO', 'THAN', 'THEN', 'WHEN', 'WHERE', 'WHY', 'HOW', 'WHAT', 'WHICH',
    'THERE', 'HERE', 'THOSE', 'THESE', 'THEIR', 'THEM', 'THIS', 'THAT', 'THOSE', 'THESE',
    'NOT', 'BUT', 'LIKE', 'MORE', 'JUST', 'SOME', 'TIME', 'GOOD', 'SAY', 'WAY', 'MOVE',
    'BACK', 'LOOK', 'THINK', 'KNOW', 'MAKE', 'TAKE', 'COME', 'WELL', 'EVEN', 'WANT',
    'NEED', 'MUCH', 'MANY', 'SUCH', 'MOST', 'PART', 'OVER', 'YEAR', 'HELP', 'WORK',
    'LIFE', 'TELL', 'CASE', 'DAYS', 'FIND', 'NEXT', 'LAST', 'WEEK', 'GIVE', 'NAME',
    'BEST', 'IDEA', 'TALK', 'SURE', 'KIND', 'HEAD', 'HAND', 'FACT', 'TYPE', 'LINE',
    'WAIT', 'AFTER', 'LONG', 'FIRST', 'NEVER', 'WORTH', 'SEVEN', 'FEAST',
    "TIME", "LINE", "TURN", "GREAT", "TIS",
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

# flatten etf list for quick lookup
VALID_ETFS = {etf for category in ETF_CATEGORIES.values() for etf in category}

# unified allow list (etfs + special non-universe symbols)
ALWAYS_ALLOW = VALID_ETFS | {"BTC", "ETH", "SPX", "GOLD", "VXUS"}

# well-known stock tickers
WELL_KNOWN_TICKERS = {
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'WFC',
    'INTC', 'AMD', 'CSCO', 'ORCL', 'IBM', 'PLTR', 'COIN', 'GME', 'AMC', 'BB',
    'F', 'GM', 'GE', 'BA', 'RTX', 'LMT', 'NOC',
    'PFE', 'JNJ', 'MRK', 'CVS', 'UNH',
    'KO', 'PEP', 'MCD', 'WMT', 'TGT',
    'DIS', 'NFLX', 'CMCSA', 'T', 'VZ',
    'RGTI', 'RGTZ', 'SEZL',
    'MSTR', 'RDDT', 'GOOG', 'ANET', 'HOOD', 'DKNG', 'MSCI', 'ADP',
    'VEQT', 'VHYG', 'PHYS', 'URA',
    'SCHD', 'VIGAX', 'FSKAX', 'FSPGX', 'VNYTX', 'FDVV', 'VYMI',
    'MAGA', 'GGLL', 'XOVR',
    'UUUU', 'SMR', 'OKLO', 'RKLB', 'ASTS', 'ACHR',
    'SOND', 'ANF', 'KSS', 'LSEG',
    'ALAB', 'IREN',
    'ESNT', 'PATH', 'CRWD',
    'DLTR', 'AVGO', 'NEE', 'PLD', 'LLY', 'BUD', 'NQ', 'DASH', 'BE',
    # newly promoted from review queue
    'WBD', 'BABA', 'BILI', 'WB', 'PDD', 'TAL', 'EDU', 'EDIT',
}

# negative context patterns that invalidate ticker matches
NEGATIVE_CONTEXT_PATTERNS = {
    'COIN': [
        'meme coin', 'shit coin', 'shitcoin', 'alt coin', 'altcoin', 'stable coin', 'stablecoin',
        'dog coin', 'dogcoin', 'moon coin', 'mooncoin', 'pump coin', 'dump coin', 'new coin',
        'this coin', 'the coin', 'that coin', 'any coin', 'my coin', 'your coin', 'their coin',
        'crypto coin', 'cryptocurrency', 'token'
    ],
    'GOLD': ['gold standard', 'gold medal', 'gold mine', 'gold rush', 'gold price'],
    'GOOD': ['good morning', 'good night', 'good day', 'good luck', 'good job'],
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

