"""Centralized ticker context keywords and disambiguation helpers."""

TICKER_CONTEXT = {
    # mega-cap tech
    "NVDA": [
        "nvidia", "geforce", "rtx", "cuda", "gpu", "jensen",
        "chips", "semiconductor", "data center", "taiwan"
    ],
    "TSLA": [
        "tesla", "elon", "musk", "gigafactory", "autopilot",
        "cybertruck", "model 3", "model y", "full self driving"
    ],
    "META": [
        "meta", "facebook", "instagram", "whatsapp", "oculus",
        "vr", "metaverse", "threads", "zuck", "zuckerberg"
    ],
    "AAPL": [
        "apple", "iphone", "ipad", "mac", "macbook", "ipad pro"
    ],
    "MSFT": [
        "microsoft", "azure", "windows", "xbox", "copilot", "nadella"
    ],
    "GME": [
        "gamestop", "ryan cohen", "stonks", "moass", "retail trader"
    ],
    "IOVA": [
        "iova", "biotech", "til therapy", "lifileucel", "melanoma",
        "cell therapy", "oncology"
    ],
    "GOLD": [
        "barrick", "barrick gold", "mining", "ounces", "commodity",
        "goldcorp", "newmont"
    ],
    "IBIT": [
        "blackrock", "bitcoin etf", "spot bitcoin", "crypto fund"
    ],
    "SPY": ["s&p 500", "index fund", "sp500"],
    "QQQ": ["nasdaq 100", "tech index", "nasdaq"],
    "RGTI": ["rigetti", "quantum", "kulkarni", "rigetti computing"],
    "RGTZ": ["rigetti", "inverse", "quantum"],
    "SEZL": ["sezzle", "bnpl", "installments", "credit", "buy now pay later"],
    "VEQT": ["vanguard all equity", "all equity etf", "veqt.to"],
    "VHYG": ["vanguard usd", "corporate bond", "high yield bond", "vhyg.l"],
    "LSEG": ["london stock exchange", "exchange group", "ftse owner"],
    "LVMH": ["louis vuitton", "moet hennessy", "bernard arnault", "luxury conglomerate"],
    "LLY": ["eli lilly", "mounjaro", "zepbound", "pharma", "weight loss drug"],
    "GOOG": ["google", "alphabet", "search", "youtube", "gemini"],
    "ANET": ["arista", "networking", "switches", "datacenter"],
    "MSTR": ["microstrategy", "michael saylor", "bitcoin", "btc treasury"],
    "HOOD": ["robinhood", "brokerage", "trading app", "vlad tenev"],
    "DKNG": ["draftkings", "sports betting", "gambling"],
    "MSCI": ["index provider", "benchmark", "indices"],
    "ADP": ["automatic data processing", "payroll", "hr software"],
    "UUUU": ["energy fuels", "uranium", "rare earths"],
    "SMR": ["small modular reactor", "nuclear"],
    "OKLO": ["nuclear", "microreactor", "oklo"],
    "RKLB": ["rocket lab", "launch", "space"],
    "ASTS": ["ast spacemobile", "satellite"],
    "ACHR": ["archer aviation", "evtol", "air taxi"],
    "SOND": ["sonder", "hospitality", "lodging"],
    "ANF": ["abercrombie", "fitch", "retail"],
    "KSS": ["kohls", "department store"],
    "PHYS": ["sprott", "gold trust"],
    "URA": ["uranium etf", "miners"],
    "ALAB": ["astera labs", "cxl", "ai networking", "pcie"],
    "IREN": ["iris energy", "bitcoin miner", "data center", "ai infrastructure"],
    "CRWD": ["crowdstrike", "endpoint", "falcon", "cybersecurity", "edr", "xdr", "threat hunting"],
    "ESNT": ["essent", "mortgage insurance", "private mortgage", "bermuda", "home loans"],
    "PATH": ["uipath", "automation", "rpa", "robotic process automation", "maestro"],
    "FLOT": ["floating rate", "bond etf", "treasury", "investment grade", "short term bond"],
    "SGOV": ["t bill", "treasury etf", "short term treasury", "government bond", "cash alternative"],
    "BMNR": ["ethereum", "validator network", "staking", "tokenization"],
}

# Keywords that strongly indicate the uppercase token is NOT a ticker.
CONTEXT_BLACKLIST = {
    "CEO": ["executive", "leadership", "position", "company", "hiring"],
    "FDA": ["approval", "regulator", "drug", "clinical", "therapy"],
    "BNPL": ["buy now pay later", "fintech", "payments", "service"],
    "USD": ["dollar", "forex", "currency", "fx", "macro"],
    "WSB": ["subreddit", "reddit", "community", "mods"],
    "SPAC": ["blank check", "shell company", "acquisition vehicle", "special purpose"],
    "ARPU": ["average revenue per user", "metric", "kpi"],
    "NYSE": ["exchange", "trading floor", "listing", "market open"],
    "FTSE": ["index", "ftse 100", "ftse 250"],
    "CNBC": ["tv", "anchor", "interview", "network"],
    "ADP": [
        "jobs report", "employment report", "labor market", "private employers",
        "payroll data", "adp numbers", "jobs data"
    ],
}

