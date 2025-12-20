# main pipeline configuration file.

# ============================================================================
# stage 1: reddit collection config
# ============================================================================

# later :
# news, worldnews, finance, technology, cryptocurrency, pennystocks
# personalfinance, algotrading, realestateinvesting, accounting

# === consider later (requires additional handling / gating) ===
# "news"                 # requires strong ticker anchoring
# "worldnews"            # macro only, very noisy
# "finance"              # mixed quality, brand/consumer noise
# "technology"           # company names without stock intent
# "cryptocurrency"       # separate pipeline entirely
# "pennystocks"          # pump risk, low signal stability

# === explicitly excluded (consumer / non-equity focus) ===
# "personalfinance"      # debt, banking, credit issues
# "algotrading"          # strategy talk, not sentiment
# "realestateinvesting"  # asset class mismatch
# "accounting"           # professional / technical, not market sentiment

SUBREDDITS = ['wallstreetbets', 'investing', 'stocks', 'securityanalysis', 'valueinvesting', 'etfs', 'financialnews']
SORT_METHODS = ['new', 'top', 'hot']
LOOKBACK = 365
NUM_POSTS = 10

# ============================================================================
# stage 2: ticker analysis config
# ============================================================================

suffixes = (" inc", " corp", " corporation", " ltd", " plc", " sa", " nv", " ag", " co", " company", " limited")

months = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}

time_tokens = { "AM", "PM", "EST", "EDT", "CST", "CDT", "MST", "MDT", "PST", "PDT", "GMT", "UTC" }

ambiguous = {"ON", "OR", "IT", "TGT", "ALL", "ONE", "UP", "NOW", "ANY", "TISI", "EDIT", "TILE", "NICE", "EVER", "SHOP", "HBM", "DBX"}

us_states = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

ticker_stop_terms = {
    "cfo", "ceo", "coo", "eps", "fcf", "ai", "ipo", "pt", "news", "adr", "etf", "etfs", "spac",
    "roi", "ip", "lego", "usd", "nswa", "wsbf", "for", "by", "as", "yolo", "dd", "eu", "ap", "mit", 
    "nav", "line", "un", "iii", "irs", "thc", "best", "vieww", "uk", "eu", "gbp", "roic", "leap",
    "pc", "id", "cs", "you", "ice", "hhs", "oz", "nov", "arr", "nyc", "tlh", "usa", "aaa", "aa", "zt",
    "xyz",

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
