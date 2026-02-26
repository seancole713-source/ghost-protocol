#!/usr/bin/env python3
"""
📋 SYMBOL REGISTRY — One Source of Truth
==========================================

Upgrade #111-116 from the 200 Upgrades Blueprint.

PROBLEM:  4+ duplicate CoinGecko ID maps scattered across:
  - ghost_scout.py  _get_crypto_price()          (~45 entries)
  - ghost_scout.py  _technical_prediction()       (~22 entries)
  - crypto/crypto_providers.py _COINGECKO_IDS     (~50 entries)
  - (plus brain's _KNOWN_CRYPTO set)

SOLUTION: ONE canonical registry for all symbol metadata.
  Every module imports from here. Zero drift. Zero duplicates.

Usage:
    from core.symbol_registry import (
        SYMBOL_REGISTRY,
        get_coingecko_id,
        is_crypto,
        is_stock,
        ALL_STOCKS,
        ALL_CRYPTO,
        KNOWN_CRYPTO,
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, FrozenSet

LOGGER = logging.getLogger("symbol_registry")


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SymbolInfo:
    """Canonical metadata for a tradeable symbol."""
    symbol: str
    name: str
    asset_class: str             # "stock" | "crypto"
    coingecko_id: Optional[str] = None   # Only for crypto
    polygon_ticker: Optional[str] = None # Only for stocks (usually == symbol)
    sector: Optional[str] = None
    market_cap_tier: Optional[str] = None  # "mega" | "large" | "mid" | "small" | "micro"


# ═══════════════════════════════════════════════════════════════════
# THE REGISTRY — Merged from all 4 duplicate sources
# ═══════════════════════════════════════════════════════════════════
# This is the ONLY place CoinGecko IDs should be defined.
# Sorted alphabetically by symbol within each section.

_CRYPTO_SYMBOLS: Dict[str, SymbolInfo] = {
    # ─── Majors ───
    "BTC":   SymbolInfo("BTC",   "Bitcoin",          "crypto", coingecko_id="bitcoin"),
    "ETH":   SymbolInfo("ETH",   "Ethereum",         "crypto", coingecko_id="ethereum"),
    "SOL":   SymbolInfo("SOL",   "Solana",           "crypto", coingecko_id="solana"),
    "XRP":   SymbolInfo("XRP",   "Ripple",           "crypto", coingecko_id="ripple"),
    "ADA":   SymbolInfo("ADA",   "Cardano",          "crypto", coingecko_id="cardano"),
    "DOGE":  SymbolInfo("DOGE",  "Dogecoin",         "crypto", coingecko_id="dogecoin"),
    "AVAX":  SymbolInfo("AVAX",  "Avalanche",        "crypto", coingecko_id="avalanche-2"),
    "DOT":   SymbolInfo("DOT",   "Polkadot",         "crypto", coingecko_id="polkadot"),
    "MATIC": SymbolInfo("MATIC", "Polygon",          "crypto", coingecko_id="matic-network"),
    "LINK":  SymbolInfo("LINK",  "Chainlink",        "crypto", coingecko_id="chainlink"),
    "UNI":   SymbolInfo("UNI",   "Uniswap",          "crypto", coingecko_id="uniswap"),
    "LTC":   SymbolInfo("LTC",   "Litecoin",         "crypto", coingecko_id="litecoin"),
    "BCH":   SymbolInfo("BCH",   "Bitcoin Cash",     "crypto", coingecko_id="bitcoin-cash"),
    "ATOM":  SymbolInfo("ATOM",  "Cosmos",           "crypto", coingecko_id="cosmos"),
    "XLM":   SymbolInfo("XLM",   "Stellar",          "crypto", coingecko_id="stellar"),

    # ─── Layer 1 ───
    "NEAR":  SymbolInfo("NEAR",  "NEAR Protocol",    "crypto", coingecko_id="near"),
    "APT":   SymbolInfo("APT",   "Aptos",            "crypto", coingecko_id="aptos"),
    "SUI":   SymbolInfo("SUI",   "Sui",              "crypto", coingecko_id="sui"),
    "SEI":   SymbolInfo("SEI",   "Sei Network",      "crypto", coingecko_id="sei-network"),
    "FTM":   SymbolInfo("FTM",   "Fantom",           "crypto", coingecko_id="fantom"),
    "ALGO":  SymbolInfo("ALGO",  "Algorand",         "crypto", coingecko_id="algorand"),
    "HBAR":  SymbolInfo("HBAR",  "Hedera",           "crypto", coingecko_id="hedera-hashgraph"),
    "VET":   SymbolInfo("VET",   "VeChain",          "crypto", coingecko_id="vechain"),
    "ICP":   SymbolInfo("ICP",   "Internet Computer","crypto", coingecko_id="internet-computer"),
    "FIL":   SymbolInfo("FIL",   "Filecoin",         "crypto", coingecko_id="filecoin"),
    "THETA": SymbolInfo("THETA", "Theta Network",    "crypto", coingecko_id="theta-token"),
    "EOS":   SymbolInfo("EOS",   "EOS",              "crypto", coingecko_id="eos"),
    "XTZ":   SymbolInfo("XTZ",   "Tezos",            "crypto", coingecko_id="tezos"),
    "EGLD":  SymbolInfo("EGLD",  "MultiversX",       "crypto", coingecko_id="elrond-erd-2"),

    # ─── DeFi ───
    "AAVE":  SymbolInfo("AAVE",  "Aave",             "crypto", coingecko_id="aave"),
    "CRV":   SymbolInfo("CRV",   "Curve DAO",        "crypto", coingecko_id="curve-dao-token"),
    "MKR":   SymbolInfo("MKR",   "Maker",            "crypto", coingecko_id="maker"),
    "SNX":   SymbolInfo("SNX",   "Synthetix",        "crypto", coingecko_id="havven"),
    "COMP":  SymbolInfo("COMP",  "Compound",         "crypto", coingecko_id="compound-coin"),
    "SUSHI": SymbolInfo("SUSHI", "SushiSwap",        "crypto", coingecko_id="sushi"),
    "YFI":   SymbolInfo("YFI",   "Yearn Finance",    "crypto", coingecko_id="yearn-finance"),
    "1INCH": SymbolInfo("1INCH", "1inch",            "crypto", coingecko_id="1inch"),
    "BAL":   SymbolInfo("BAL",   "Balancer",         "crypto", coingecko_id="balancer"),
    "LDO":   SymbolInfo("LDO",   "Lido DAO",         "crypto", coingecko_id="lido-dao"),
    "PENDLE":SymbolInfo("PENDLE","Pendle",            "crypto", coingecko_id="pendle"),
    "GMX":   SymbolInfo("GMX",   "GMX",              "crypto", coingecko_id="gmx"),

    # ─── Layer 2 ───
    "ARB":   SymbolInfo("ARB",   "Arbitrum",         "crypto", coingecko_id="arbitrum"),
    "OP":    SymbolInfo("OP",    "Optimism",         "crypto", coingecko_id="optimism"),
    "IMX":   SymbolInfo("IMX",   "Immutable X",      "crypto", coingecko_id="immutable-x"),
    "LRC":   SymbolInfo("LRC",   "Loopring",         "crypto", coingecko_id="loopring"),
    "STRK":  SymbolInfo("STRK",  "StarkNet",         "crypto", coingecko_id="starknet"),
    "ZK":    SymbolInfo("ZK",    "ZKSync",           "crypto", coingecko_id="zksync"),

    # ─── AI & Compute ───
    "RNDR":  SymbolInfo("RNDR",  "Render",           "crypto", coingecko_id="render-token"),
    "FET":   SymbolInfo("FET",   "Fetch.ai",         "crypto", coingecko_id="fetch-ai"),
    "OCEAN": SymbolInfo("OCEAN", "Ocean Protocol",   "crypto", coingecko_id="ocean-protocol"),
    "AGIX":  SymbolInfo("AGIX",  "SingularityNET",   "crypto", coingecko_id="singularitynet"),
    "TAO":   SymbolInfo("TAO",   "Bittensor",        "crypto", coingecko_id="bittensor"),
    "AKT":   SymbolInfo("AKT",   "Akash Network",    "crypto", coingecko_id="akash-network"),

    # ─── Gaming & NFT ───
    "AXS":   SymbolInfo("AXS",   "Axie Infinity",    "crypto", coingecko_id="axie-infinity"),
    "SAND":  SymbolInfo("SAND",  "The Sandbox",      "crypto", coingecko_id="the-sandbox"),
    "MANA":  SymbolInfo("MANA",  "Decentraland",     "crypto", coingecko_id="decentraland"),
    "ENJ":   SymbolInfo("ENJ",   "Enjin Coin",       "crypto", coingecko_id="enjin-coin"),
    "GALA":  SymbolInfo("GALA",  "Gala Games",       "crypto", coingecko_id="gala"),
    "ILV":   SymbolInfo("ILV",   "Illuvium",         "crypto", coingecko_id="illuvium"),
    "MAGIC": SymbolInfo("MAGIC", "Magic",            "crypto", coingecko_id="magic"),
    "GODS":  SymbolInfo("GODS",  "Gods Unchained",   "crypto", coingecko_id="gods-unchained"),
    "PRIME": SymbolInfo("PRIME", "Echelon Prime",    "crypto", coingecko_id="echelon-prime"),
    "YGG":   SymbolInfo("YGG",   "Yield Guild Games","crypto", coingecko_id="yield-guild-games"),
    "RON":   SymbolInfo("RON",   "Ronin",            "crypto", coingecko_id="ronin"),

    # ─── Infrastructure ───
    "GRT":   SymbolInfo("GRT",   "The Graph",        "crypto", coingecko_id="the-graph"),
    "ROSE":  SymbolInfo("ROSE",  "Oasis Network",    "crypto", coingecko_id="oasis-network"),
    "AR":    SymbolInfo("AR",    "Arweave",          "crypto", coingecko_id="arweave"),
    "KAVA":  SymbolInfo("KAVA",  "Kava",             "crypto", coingecko_id="kava"),
    "INJ":   SymbolInfo("INJ",   "Injective",        "crypto", coingecko_id="injective-protocol"),
    "TIA":   SymbolInfo("TIA",   "Celestia",         "crypto", coingecko_id="celestia"),
    "PYTH":  SymbolInfo("PYTH",  "Pyth Network",     "crypto", coingecko_id="pyth-network"),
    "JUP":   SymbolInfo("JUP",   "Jupiter",          "crypto", coingecko_id="jupiter-exchange-solana"),
    "JTO":   SymbolInfo("JTO",   "Jito",             "crypto", coingecko_id="jito-governance-token"),
    "BONK":  SymbolInfo("BONK",  "Bonk",             "crypto", coingecko_id="bonk"),
    "WIF":   SymbolInfo("WIF",   "Dogwifhat",        "crypto", coingecko_id="dogwifcoin"),

    # ─── Meme / Community ───
    "SHIB":  SymbolInfo("SHIB",  "Shiba Inu",        "crypto", coingecko_id="shiba-inu"),
    "PEPE":  SymbolInfo("PEPE",  "Pepe",             "crypto", coingecko_id="pepe"),
    "FLOKI": SymbolInfo("FLOKI", "Floki",            "crypto", coingecko_id="floki"),
    "TURBO": SymbolInfo("TURBO", "Turbo",            "crypto", coingecko_id="turbo"),
    "WLD":   SymbolInfo("WLD",   "Worldcoin",        "crypto", coingecko_id="worldcoin-wld"),
    "BLUR":  SymbolInfo("BLUR",  "Blur",             "crypto", coingecko_id="blur"),

    # ─── Other Crypto ───
    "DYDX":  SymbolInfo("DYDX",  "dYdX",             "crypto", coingecko_id="dydx"),
    "MASK":  SymbolInfo("MASK",  "Mask Network",     "crypto", coingecko_id="mask-network"),
    "ENS":   SymbolInfo("ENS",   "ENS Domains",      "crypto", coingecko_id="ethereum-name-service"),
    "CHZ":   SymbolInfo("CHZ",   "Chiliz",           "crypto", coingecko_id="chiliz"),
    "AUDIO": SymbolInfo("AUDIO", "Audius",           "crypto", coingecko_id="audius"),
    "SUPER": SymbolInfo("SUPER", "SuperVerse",       "crypto", coingecko_id="superverse"),

    # ─── Old Guard ───
    "ZEC":     SymbolInfo("ZEC",     "Zcash",         "crypto", coingecko_id="zcash"),
    "DASHCOIN":SymbolInfo("DASHCOIN","Dash",          "crypto", coingecko_id="dash"),
    "NEO":     SymbolInfo("NEO",     "NEO",           "crypto", coingecko_id="neo"),
    "WAVES":   SymbolInfo("WAVES",   "Waves",         "crypto", coingecko_id="waves"),
    "QTUM":    SymbolInfo("QTUM",    "Qtum",          "crypto", coingecko_id="qtum"),
    "ZIL":     SymbolInfo("ZIL",     "Zilliqa",       "crypto", coingecko_id="zilliqa"),
    "ICX":     SymbolInfo("ICX",     "ICON",          "crypto", coingecko_id="icon"),
    "RLC":     SymbolInfo("RLC",     "iExec RLC",     "crypto", coingecko_id="iexec-rlc"),
    "OMG":     SymbolInfo("OMG",     "OMG Network",   "crypto", coingecko_id="omisego"),
    "BAT":     SymbolInfo("BAT",     "Basic Attention","crypto", coingecko_id="basic-attention-token"),
    "KNC":     SymbolInfo("KNC",     "Kyber Network", "crypto", coingecko_id="kyber-network-crystal"),
    "ZRX":     SymbolInfo("ZRX",     "0x Protocol",   "crypto", coingecko_id="0x"),

    # ─── Collision-safe names ───
    "STACKS":  SymbolInfo("STACKS", "Stacks",         "crypto", coingecko_id="blockstack"),

    # ─── Edge whitelist specials ───
    "GIGA":    SymbolInfo("GIGA",   "GigaChad",       "crypto", coingecko_id="gigachad-2"),
    "IOTX":    SymbolInfo("IOTX",   "IoTeX",          "crypto", coingecko_id="iotex"),
    "ALICE":   SymbolInfo("ALICE",  "My Neighbor Alice","crypto",coingecko_id="my-neighbor-alice"),
    "BRETT":   SymbolInfo("BRETT",  "Brett",           "crypto", coingecko_id="based-brett"),
    "IQ":      SymbolInfo("IQ",     "IQ",              "crypto", coingecko_id="everipedia"),
    "BAND":    SymbolInfo("BAND",   "Band Protocol",   "crypto", coingecko_id="band-protocol"),

    # ─── Additional from crypto_providers.py ───
    "XMR":     SymbolInfo("XMR",    "Monero",          "crypto", coingecko_id="monero"),
    "TON":     SymbolInfo("TON",    "Toncoin",         "crypto", coingecko_id="the-open-network"),
    "TRX":     SymbolInfo("TRX",    "TRON",            "crypto", coingecko_id="tron"),
    "ETC":     SymbolInfo("ETC",    "Ethereum Classic", "crypto", coingecko_id="ethereum-classic"),
}

_STOCK_SYMBOLS: Dict[str, SymbolInfo] = {
    # ─── Tech Giants ───
    "AAPL":  SymbolInfo("AAPL",  "Apple",           "stock", polygon_ticker="AAPL", sector="tech"),
    "MSFT":  SymbolInfo("MSFT",  "Microsoft",       "stock", polygon_ticker="MSFT", sector="tech"),
    "GOOGL": SymbolInfo("GOOGL", "Alphabet",        "stock", polygon_ticker="GOOGL", sector="tech"),
    "AMZN":  SymbolInfo("AMZN",  "Amazon",          "stock", polygon_ticker="AMZN", sector="tech"),
    "META":  SymbolInfo("META",  "Meta Platforms",   "stock", polygon_ticker="META", sector="tech"),
    "NVDA":  SymbolInfo("NVDA",  "NVIDIA",          "stock", polygon_ticker="NVDA", sector="semiconductors"),
    "AMD":   SymbolInfo("AMD",   "AMD",             "stock", polygon_ticker="AMD",  sector="semiconductors"),
    "TSLA":  SymbolInfo("TSLA",  "Tesla",           "stock", polygon_ticker="TSLA", sector="auto"),
    "NFLX":  SymbolInfo("NFLX",  "Netflix",         "stock", polygon_ticker="NFLX", sector="tech"),
    "CRM":   SymbolInfo("CRM",   "Salesforce",      "stock", polygon_ticker="CRM",  sector="tech"),
    "ORCL":  SymbolInfo("ORCL",  "Oracle",          "stock", polygon_ticker="ORCL", sector="tech"),
    "ADBE":  SymbolInfo("ADBE",  "Adobe",           "stock", polygon_ticker="ADBE", sector="tech"),
    "INTC":  SymbolInfo("INTC",  "Intel",           "stock", polygon_ticker="INTC", sector="semiconductors"),
    "CSCO":  SymbolInfo("CSCO",  "Cisco",           "stock", polygon_ticker="CSCO", sector="tech"),
    "IBM":   SymbolInfo("IBM",   "IBM",             "stock", polygon_ticker="IBM",  sector="tech"),

    # ─── Storage & Hardware ───
    "STX":   SymbolInfo("STX",   "Seagate",         "stock", polygon_ticker="STX",  sector="hardware"),
    "WDC":   SymbolInfo("WDC",   "Western Digital",  "stock", polygon_ticker="WDC",  sector="hardware"),
    "NTAP":  SymbolInfo("NTAP",  "NetApp",          "stock", polygon_ticker="NTAP", sector="hardware"),
    "PSTG":  SymbolInfo("PSTG",  "Pure Storage",    "stock", polygon_ticker="PSTG", sector="hardware"),

    # ─── Semiconductors ───
    "AVGO":  SymbolInfo("AVGO",  "Broadcom",        "stock", polygon_ticker="AVGO", sector="semiconductors"),
    "QCOM":  SymbolInfo("QCOM",  "Qualcomm",        "stock", polygon_ticker="QCOM", sector="semiconductors"),
    "TXN":   SymbolInfo("TXN",   "Texas Instruments","stock", polygon_ticker="TXN",  sector="semiconductors"),
    "MU":    SymbolInfo("MU",    "Micron",          "stock", polygon_ticker="MU",   sector="semiconductors"),
    "LRCX":  SymbolInfo("LRCX",  "Lam Research",    "stock", polygon_ticker="LRCX", sector="semiconductors"),
    "AMAT":  SymbolInfo("AMAT",  "Applied Materials","stock", polygon_ticker="AMAT", sector="semiconductors"),
    "KLAC":  SymbolInfo("KLAC",  "KLA Corp",        "stock", polygon_ticker="KLAC", sector="semiconductors"),
    "MRVL":  SymbolInfo("MRVL",  "Marvell",         "stock", polygon_ticker="MRVL", sector="semiconductors"),
    "ON":    SymbolInfo("ON",    "ON Semi",         "stock", polygon_ticker="ON",   sector="semiconductors"),
    "NXPI":  SymbolInfo("NXPI",  "NXP Semi",        "stock", polygon_ticker="NXPI", sector="semiconductors"),
    "ADI":   SymbolInfo("ADI",   "Analog Devices",  "stock", polygon_ticker="ADI",  sector="semiconductors"),
    "MCHP":  SymbolInfo("MCHP",  "Microchip Tech",  "stock", polygon_ticker="MCHP", sector="semiconductors"),

    # ─── AI & Innovation ───
    "PLTR":  SymbolInfo("PLTR",  "Palantir",        "stock", polygon_ticker="PLTR", sector="tech"),
    "AI":    SymbolInfo("AI",    "C3.ai",           "stock", polygon_ticker="AI",   sector="tech"),
    "PATH":  SymbolInfo("PATH",  "UiPath",          "stock", polygon_ticker="PATH", sector="tech"),
    "UPST":  SymbolInfo("UPST",  "Upstart",         "stock", polygon_ticker="UPST", sector="fintech"),
    "COIN":  SymbolInfo("COIN",  "Coinbase",        "stock", polygon_ticker="COIN", sector="fintech"),
    "HOOD":  SymbolInfo("HOOD",  "Robinhood",       "stock", polygon_ticker="HOOD", sector="fintech"),
    "SOFI":  SymbolInfo("SOFI",  "SoFi",            "stock", polygon_ticker="SOFI", sector="fintech"),

    # ─── Healthcare & Biotech ───
    "JNJ":   SymbolInfo("JNJ",   "J&J",             "stock", polygon_ticker="JNJ",  sector="healthcare"),
    "UNH":   SymbolInfo("UNH",   "UnitedHealth",    "stock", polygon_ticker="UNH",  sector="healthcare"),
    "PFE":   SymbolInfo("PFE",   "Pfizer",          "stock", polygon_ticker="PFE",  sector="pharma"),
    "ABBV":  SymbolInfo("ABBV",  "AbbVie",          "stock", polygon_ticker="ABBV", sector="pharma"),
    "MRK":   SymbolInfo("MRK",   "Merck",           "stock", polygon_ticker="MRK",  sector="pharma"),
    "LLY":   SymbolInfo("LLY",   "Eli Lilly",       "stock", polygon_ticker="LLY",  sector="pharma"),
    "AMGN":  SymbolInfo("AMGN",  "Amgen",           "stock", polygon_ticker="AMGN", sector="biotech"),
    "GILD":  SymbolInfo("GILD",  "Gilead",          "stock", polygon_ticker="GILD", sector="biotech"),
    "BMY":   SymbolInfo("BMY",   "Bristol-Myers",   "stock", polygon_ticker="BMY",  sector="pharma"),
    "REGN":  SymbolInfo("REGN",  "Regeneron",       "stock", polygon_ticker="REGN", sector="biotech"),
    "VRTX":  SymbolInfo("VRTX",  "Vertex",          "stock", polygon_ticker="VRTX", sector="biotech"),
    "MRNA":  SymbolInfo("MRNA",  "Moderna",         "stock", polygon_ticker="MRNA", sector="biotech"),
    "BIIB":  SymbolInfo("BIIB",  "Biogen",          "stock", polygon_ticker="BIIB", sector="biotech"),

    # ─── Finance ───
    "JPM":   SymbolInfo("JPM",   "JPMorgan",        "stock", polygon_ticker="JPM",  sector="finance"),
    "BAC":   SymbolInfo("BAC",   "Bank of America", "stock", polygon_ticker="BAC",  sector="finance"),
    "WFC":   SymbolInfo("WFC",   "Wells Fargo",     "stock", polygon_ticker="WFC",  sector="finance"),
    "GS":    SymbolInfo("GS",    "Goldman Sachs",   "stock", polygon_ticker="GS",   sector="finance"),
    "MS":    SymbolInfo("MS",    "Morgan Stanley",  "stock", polygon_ticker="MS",   sector="finance"),
    "C":     SymbolInfo("C",     "Citigroup",       "stock", polygon_ticker="C",    sector="finance"),
    "AXP":   SymbolInfo("AXP",   "American Express","stock", polygon_ticker="AXP",  sector="finance"),
    "V":     SymbolInfo("V",     "Visa",            "stock", polygon_ticker="V",    sector="finance"),
    "MA":    SymbolInfo("MA",    "Mastercard",      "stock", polygon_ticker="MA",   sector="finance"),
    "PYPL":  SymbolInfo("PYPL",  "PayPal",          "stock", polygon_ticker="PYPL", sector="fintech"),
    "SQ":    SymbolInfo("SQ",    "Block (Square)",  "stock", polygon_ticker="SQ",   sector="fintech"),
    "BLK":   SymbolInfo("BLK",   "BlackRock",       "stock", polygon_ticker="BLK",  sector="finance"),
    "SCHW":  SymbolInfo("SCHW",  "Schwab",          "stock", polygon_ticker="SCHW", sector="finance"),

    # ─── Consumer ───
    "NKE":   SymbolInfo("NKE",   "Nike",            "stock", polygon_ticker="NKE",  sector="consumer"),
    "SBUX":  SymbolInfo("SBUX",  "Starbucks",       "stock", polygon_ticker="SBUX", sector="consumer"),
    "MCD":   SymbolInfo("MCD",   "McDonald's",      "stock", polygon_ticker="MCD",  sector="consumer"),
    "KO":    SymbolInfo("KO",    "Coca-Cola",       "stock", polygon_ticker="KO",   sector="consumer"),
    "PEP":   SymbolInfo("PEP",   "PepsiCo",         "stock", polygon_ticker="PEP",  sector="consumer"),
    "WMT":   SymbolInfo("WMT",   "Walmart",         "stock", polygon_ticker="WMT",  sector="retail"),
    "COST":  SymbolInfo("COST",  "Costco",          "stock", polygon_ticker="COST", sector="retail"),
    "TGT":   SymbolInfo("TGT",   "Target",          "stock", polygon_ticker="TGT",  sector="retail"),
    "HD":    SymbolInfo("HD",    "Home Depot",      "stock", polygon_ticker="HD",   sector="retail"),
    "LOW":   SymbolInfo("LOW",   "Lowe's",          "stock", polygon_ticker="LOW",  sector="retail"),
    "DIS":   SymbolInfo("DIS",   "Disney",          "stock", polygon_ticker="DIS",  sector="media"),
    "CMCSA": SymbolInfo("CMCSA", "Comcast",         "stock", polygon_ticker="CMCSA", sector="media"),

    # ─── Energy & Industrial ───
    "XOM":   SymbolInfo("XOM",   "ExxonMobil",      "stock", polygon_ticker="XOM",  sector="energy"),
    "CVX":   SymbolInfo("CVX",   "Chevron",         "stock", polygon_ticker="CVX",  sector="energy"),
    "COP":   SymbolInfo("COP",   "ConocoPhillips",  "stock", polygon_ticker="COP",  sector="energy"),
    "SLB":   SymbolInfo("SLB",   "Schlumberger",    "stock", polygon_ticker="SLB",  sector="energy"),
    "CAT":   SymbolInfo("CAT",   "Caterpillar",     "stock", polygon_ticker="CAT",  sector="industrial"),
    "DE":    SymbolInfo("DE",    "Deere",           "stock", polygon_ticker="DE",   sector="industrial"),
    "HON":   SymbolInfo("HON",   "Honeywell",       "stock", polygon_ticker="HON",  sector="industrial"),
    "GE":    SymbolInfo("GE",    "GE Aerospace",    "stock", polygon_ticker="GE",   sector="industrial"),
    "BA":    SymbolInfo("BA",    "Boeing",          "stock", polygon_ticker="BA",   sector="aerospace"),
    "RTX":   SymbolInfo("RTX",   "RTX Corp",        "stock", polygon_ticker="RTX",  sector="aerospace"),
    "LMT":   SymbolInfo("LMT",   "Lockheed Martin", "stock", polygon_ticker="LMT",  sector="aerospace"),
    "UPS":   SymbolInfo("UPS",   "UPS",             "stock", polygon_ticker="UPS",  sector="logistics"),
    "FDX":   SymbolInfo("FDX",   "FedEx",           "stock", polygon_ticker="FDX",  sector="logistics"),

    # ─── Cybersecurity (V3 validated) ───
    "PANW":  SymbolInfo("PANW",  "Palo Alto",       "stock", polygon_ticker="PANW", sector="cybersecurity"),
    "NET":   SymbolInfo("NET",   "Cloudflare",      "stock", polygon_ticker="NET",  sector="cybersecurity"),
    "FTNT":  SymbolInfo("FTNT",  "Fortinet",        "stock", polygon_ticker="FTNT", sector="cybersecurity"),
    "DDOG":  SymbolInfo("DDOG",  "Datadog",         "stock", polygon_ticker="DDOG", sector="tech"),

    # ─── Others ───
    "ABNB":  SymbolInfo("ABNB",  "Airbnb",          "stock", polygon_ticker="ABNB", sector="travel"),
    "UBER":  SymbolInfo("UBER",  "Uber",            "stock", polygon_ticker="UBER", sector="transport"),
    "LYFT":  SymbolInfo("LYFT",  "Lyft",            "stock", polygon_ticker="LYFT", sector="transport"),
    "DASH":  SymbolInfo("DASH",  "DoorDash",        "stock", polygon_ticker="DASH", sector="tech"),
    "SPOT":  SymbolInfo("SPOT",  "Spotify",         "stock", polygon_ticker="SPOT", sector="media"),
    "ZM":    SymbolInfo("ZM",    "Zoom",            "stock", polygon_ticker="ZM",   sector="tech"),
    "SHOP":  SymbolInfo("SHOP",  "Shopify",         "stock", polygon_ticker="SHOP", sector="ecommerce"),
    "ROKU":  SymbolInfo("ROKU",  "Roku",            "stock", polygon_ticker="ROKU", sector="media"),
    "SNAP":  SymbolInfo("SNAP",  "Snap",            "stock", polygon_ticker="SNAP", sector="social"),
    "PINS":  SymbolInfo("PINS",  "Pinterest",       "stock", polygon_ticker="PINS", sector="social"),
    "TWLO":  SymbolInfo("TWLO",  "Twilio",          "stock", polygon_ticker="TWLO", sector="tech"),
    "OKTA":  SymbolInfo("OKTA",  "Okta",            "stock", polygon_ticker="OKTA", sector="cybersecurity"),

    # ─── V3 Whitelist stocks ───
    "T":     SymbolInfo("T",     "AT&T",            "stock", polygon_ticker="T",    sector="telecom"),
    "BMBL":  SymbolInfo("BMBL",  "Bumble",          "stock", polygon_ticker="BMBL", sector="tech"),
    "XPO":   SymbolInfo("XPO",   "XPO Logistics",   "stock", polygon_ticker="XPO",  sector="logistics"),
}


# ═══════════════════════════════════════════════════════════════════
# COMBINED REGISTRY
# ═══════════════════════════════════════════════════════════════════

SYMBOL_REGISTRY: Dict[str, SymbolInfo] = {**_CRYPTO_SYMBOLS, **_STOCK_SYMBOLS}


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE EXPORTS (replace scattered lists)
# ═══════════════════════════════════════════════════════════════════

# Replaces ghost_scout.py ALL_STOCKS
ALL_STOCKS: list = sorted(_STOCK_SYMBOLS.keys())

# Replaces ghost_scout.py ALL_CRYPTO
ALL_CRYPTO: list = sorted(_CRYPTO_SYMBOLS.keys())

# Replaces ghost_brain.py _KNOWN_CRYPTO
KNOWN_CRYPTO: FrozenSet[str] = frozenset(_CRYPTO_SYMBOLS.keys())

# CoinGecko ID lookup (replaces 4 duplicate maps)
_COINGECKO_ID_MAP: Dict[str, str] = {
    sym: info.coingecko_id
    for sym, info in _CRYPTO_SYMBOLS.items()
    if info.coingecko_id
}


# ═══════════════════════════════════════════════════════════════════
# LOOKUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_coingecko_id(symbol: str) -> Optional[str]:
    """Get CoinGecko ID for a crypto symbol. Returns None for stocks."""
    return _COINGECKO_ID_MAP.get(symbol.upper())


def get_symbol_info(symbol: str) -> Optional[SymbolInfo]:
    """Get full metadata for any symbol."""
    return SYMBOL_REGISTRY.get(symbol.upper())


def is_crypto(symbol: str) -> bool:
    """Check if a symbol is a known crypto asset."""
    info = SYMBOL_REGISTRY.get(symbol.upper())
    return info is not None and info.asset_class == "crypto"


def is_stock(symbol: str) -> bool:
    """Check if a symbol is a known stock."""
    info = SYMBOL_REGISTRY.get(symbol.upper())
    return info is not None and info.asset_class == "stock"


def get_asset_class(symbol: str) -> str:
    """Get asset class for a symbol. Returns 'unknown' if not in registry."""
    info = SYMBOL_REGISTRY.get(symbol.upper())
    return info.asset_class if info else "unknown"


def get_sector(symbol: str) -> Optional[str]:
    """Get sector for a symbol (stocks only)."""
    info = SYMBOL_REGISTRY.get(symbol.upper())
    return info.sector if info else None


def get_symbols_by_sector(sector: str) -> list:
    """Get all symbols in a given sector."""
    return [
        sym for sym, info in SYMBOL_REGISTRY.items()
        if info.sector == sector
    ]


def get_all_coingecko_ids() -> Dict[str, str]:
    """Get full CoinGecko ID mapping (for batch API calls)."""
    return dict(_COINGECKO_ID_MAP)


LOGGER.info(
    f"[REGISTRY] Loaded: {len(_STOCK_SYMBOLS)} stocks, "
    f"{len(_CRYPTO_SYMBOLS)} crypto, "
    f"{len(_COINGECKO_ID_MAP)} CoinGecko IDs"
)
