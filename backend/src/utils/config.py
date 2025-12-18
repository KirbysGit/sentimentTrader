# main pipeline configuration file.

# ============================================================================
# stage 1: reddit collection config
# ============================================================================

# later :
# news, worldnews, finance, technology, cryptocurrency, pennystocks

SUBREDDITS = ['wallstreetbets', 'news', 'investing']
SORT_METHODS = ['new', 'top', 'hot']
LOOKBACK = 30
NUM_POSTS = 10

# ============================================================================
# stage 2: ticker analysis config
# ============================================================================

suffixes = (" inc", " corp", " corporation", " ltd", " plc", " sa", " nv", " ag", " co", " company", " limited")

ticker_stop_terms = {
    "cfo", "ceo", "coo", "eps", "fcf", "ai", "ipo", "pt", "news", "adr", "etf", "etfs", "spac",
    "roi", "ip", "lego", "usd", "nswa", "wsbf", "for", "by", "as", "yolo", "dd", "eu", "ap", "mit", 
    "nav", "line", "un", "iii", "irs", "thc", "best", "vieww", "uk", "eu", "gbp", "roic"
}

popular_tickers = {
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "INTC", "QCOM", "MU", "TSM", "AVGO", "ASML", "ARM",
    "CRM", "ORCL", "NOW", "ADBE", "SNOW", "DDOG", "MDB", "ZS",
    "V", "MA", "PYPL", "SQ", "AFRM", "SOFI", "COIN",
    "NFLX", "DIS", "HD", "COST", "WMT", "TGT", "LULU", "NKE",
    "SBUX", "MCD", "XOM", "CVX", "OXY", "SLB", "COP",
    "JPM", "GS", "MS", "BAC", "C", "WFC", "BLK",
    "LLY", "JNJ", "UNH", "PFE", "MRK", "ABBV", "NVO",
    "WBD", "PARA", "SONY", "NTDOY", "F", "GM", "CAT", "DE",
    "BA", "LVMH", "RACE", "SPY", "QQQ", "DIA", "MARA", "RIOT",
    "HUT", "CLSK",
}


# financial context words (merged strong + weak)
common_finance_words = {
    'acquisition', 'acquire', 'analysis', 'analyst', 'antitrust', 'assets',
    'averaged', 'bagged', 'bagging', 'bearish', 'bid', 'bond', 'boost',
    'bullish', 'buy', 'buyback', 'calls', 'capex', 'circuit breaker',
    'company', 'competition', 'consensus', 'contract', 'convert', 'convertible',
    'corporation', 'coupon', 'cut', 'dark pool', 'deal', 'debt', 'decline',
    'direct listing', 'dividend', 'down', 'drop', 'dumped', 'dumping',
    'downgrade', 'downgraded', 'ebit', 'ebitda', 'earnings', 'eps', 'estimate',
    'etf', 'expansion', 'fall', 'fcf', 'float', 'flow', 'follow-on', 'forecast',
    'free cash flow', 'gain', 'gamma', 'guidance', 'growth', 'gross margin',
    'halt', 'headcount', 'hedge fund', 'hike', 'industry', 'initiated',
    'institutional', 'invest', 'investor', 'ipo', 'iv', 'launch', 'layoff',
    'lbo', 'lockup', 'loss', 'ltd', 'margin', 'margins', 'market', 'market cap',
    'market share', 'merger', 'miss', 'offer', 'operating margin', 'opex',
    'options', 'outlook', 'partnership', 'payout', 'pe', 'pe ratio',
    'performance', 'pipe', 'portfolio', 'position', 'positioned', 'positions',
    'preannounce', 'prelim', 'preliminary', 'price', 'price target', 'product',
    'profit', 'profit warning', 'puts', 'quarter', 'raise', 'reiterated',
    'restructuring', 'revenue', 'rise', 's&p', 'scaling', 'secondary', 'sector',
    'sees', 'sell', 'selling', 'service', 'shares', 'shareholders',
    'short interest', 'slash', 'sold', 'sp500', 'spac', 'stock', 'strategy',
    'strike', 'takeover', 'tech', 'theta', 'ticker', 'trade', 'trading',
    'trend', 'trim', 'trimmed', 'turbulence', 'turbulences', 'up', 'upgrade',
    'upgraded', 'vega', 'volatility', 'volume', 'yield',
}


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

