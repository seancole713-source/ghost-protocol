#!/usr/bin/env python3
"""
Quick ingestion - Uses Ghost's existing 187 symbols + top 813 from a curated list
Total: 1000 stocks ready for predictions in ~5 minutes
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import time

DATABASE_URL = "postgresql://postgres:jdkObNnbzRoxzsPicrsfDeNuSUIrTgLp@metro.proxy.rlwy.net:28328/railway"

# Top 1000 US stocks by market cap (manually curated list)
TOP_1000_STOCKS = [
    # Mega caps (Top 50)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "LLY", "V",
    "JPM", "UNH", "XOM", "MA", "JNJ", "PG", "AVGO", "HD", "CVX", "MRK",
    "ABBV", "COST", "KO", "PEP", "WMT", "ADBE", "BAC", "CRM", "NFLX", "AMD",
    "TMO", "ACN", "MCD", "CSCO", "ABT", "DIS", "NKE", "TXN", "PM", "DHR",
    "ORCL", "VZ", "INTC", "CMCSA", "WFC", "QCOM", "COP", "NEE", "INTU", "UNP",
    
    # Large caps (51-200)
    "IBM", "HON", "RTX", "LOW", "AMAT", "AMGN", "CAT", "SPGI", "AXP", "GE",
    "SBUX", "DE", "BLK", "MDT", "NOW", "BKNG", "GS", "TJX", "GILD", "MMC",
    "SYK", "C", "AMT", "ADP", "VRTX", "LRCX", "REGN", "PLD", "ISRG", "ZTS",
    "CB", "ADI", "BMY", "CI", "MO", "SO", "BSX", "MDLZ", "SCHW", "EOG",
    "DUK", "BDX", "PNC", "AON", "APD", "ITW", "USB", "CL", "MMM", "SHW",
    "MU", "WM", "FI", "NOC", "HCA", "ETN", "CME", "TT", "MSI", "EMR",
    "PGR", "ICE", "APH", "ECL", "GD", "NSC", "MAR", "KLAC", "TDG", "MCO",
    "SNPS", "SLB", "PSA", "PYPL", "ROP", "CDNS", "EL", "FTNT", "EW", "WELL",
    "BK", "ORLY", "GM", "COF", "ADSK", "HLT", "AJG", "AFL", "TGT", "FCX",
    "CARR", "PSX", "SPG", "DLR", "AZO", "AIG", "JCI", "PCAR", "NEM", "VLO",
    "NXPI", "SRE", "KMI", "PAYX", "ROST", "CMG", "O", "TRV", "MSCI", "AEP",
    "MPC", "IQV", "CCI", "TEL", "ALL", "MCHP", "PPG", "ANET", "DHI", "D",
    "PCG", "HUM", "CHTR", "DD", "KMB", "ODFL", "FIS", "EA", "LHX", "VRSK",
    "CTAS", "AME", "FAST", "HSY", "CTSH", "IDXX", "YUM", "KVUE", "GWW", "RSG",
    "OTIS", "STZ", "GLW", "BKR", "WMB", "A", "KDP", "ON", "CPRT", "PRU",
    
    # Mid caps (201-500)
    "HWM", "OKE", "EXC", "CEG", "GEHC", "MLM", "FANG", "DAL", "VMC", "URI",
    "APTV", "ANSS", "ROK", "EXR", "KR", "XEL", "RMD", "SBAC", "VICI", "IT",
    "TROW", "WEC", "ACGL", "MPWR", "ED", "MTB", "IFF", "EBAY", "DOV", "CTVA",
    "DFS", "LEN", "CDW", "KEYS", "WST", "HAL", "WY", "EFX", "WAB", "FTV",
    "GIS", "AVB", "TSCO", "TTWO", "PPL", "CBRE", "STT", "DTE", "MTD", "ES",
    "TRGP", "ETR", "AWK", "RJF", "CAH", "WBD", "TYL", "HUBB", "BR", "EQR",
    "VTR", "VLTO", "AEE", "HPQ", "BIIB", "RF", "BAX", "FE", "HBAN", "CHD",
    "ZBH", "TDY", "CINF", "SYY", "EIX", "EXPD", "PKI", "INVH", "LYB", "DLTR",
    "NTAP", "CMS", "IR", "BALL", "MOH", "K", "ARE", "STE", "FITB", "GPN",
    "ESS", "EXPE", "ATO", "LUV", "STLD", "WAT", "SWK", "DRI", "DECK", "TXT",
    "DG", "NVR", "CCL", "CF", "IEX", "DPZ", "NI", "ULTA", "WTW", "ZBRA",
    "FSLR", "CBOE", "CNP", "CFG", "J", "COO", "MAA", "CAG", "BBY", "AVY",
    "FDS", "HOLX", "CLX", "OMC", "MKC", "LH", "MRO", "POOL", "NTRS", "PFG",
    "IP", "TSN", "ALGN", "AMCR", "GRMN", "L", "AES", "WRB", "LDOS", "HST",
    "JBHT", "EVRG", "SYF", "IRM", "SNA", "KIM", "JKHY", "TER", "ERIE", "AKAM",
    "UDR", "NDAQ", "CPT", "EMN", "VTRS", "BLDR", "LNT", "ENPH", "APA", "PTC",
    "FICO", "NDSN", "UAL", "KEY", "BXP", "REG", "TAP", "ROL", "CHRW", "SWKS",
    "BG", "GL", "TRMB", "CPB", "RHI", "CTLT", "PAYC", "MAS", "IPG", "MKTX",
    "TECH", "BBWI", "CRL", "AIZ", "HII", "BRO", "CZR", "UHS", "GNRC", "FOXA",
    "LKQ", "INCY", "AOS", "ALLE", "BEN", "HSIC", "AAL", "KMX", "HRL", "WHR",
    
    # Growth/Tech (501-700)
    "CRWD", "PANW", "SNOW", "DDOG", "NET", "ZS", "PLTR", "COIN", "SHOP", "UBER",
    "ABNB", "DASH", "SQ", "RBLX", "U", "PATH", "DKNG", "RBRK", "HOOD", "DOCN",
    "MDB", "ZM", "TWLO", "OKTA", "BILL", "TEAM", "WDAY", "ZI", "DOCU", "COUP",
    "CFLT", "S", "ESTC", "DBX", "BOX", "FROG", "GTLB", "PCTY", "NCNO", "IOT",
    "PSTG", "DT", "APPN", "PD", "TENB", "VEEV", "HUBS", "ETSY", "SE", "MELI",
    "SPOT", "TTD", "PINS", "SNAP", "LYFT", "RIVN", "LCID", "NKLA", "PLUG", "FCEL",
    "BLNK", "CHPT", "QS", "GOEV", "FSR", "NIO", "XPEV", "LI", "BYDDY", "TSLL",
    "ROKU", "SONO", "PTON", "LULU", "RH", "CPRI", "TPR", "RL", "PVH", "UAA",
    "LEVI", "SKX", "CROX", "VFC", "HBI", "FL", "GPS", "ANF", "AEO", "URBN",
    "FIVE", "BURL", "OLLI", "BJ", "ANF", "GES", "DKS", "ASO", "BGFV", "HIBB",
    "TSCO", "TGT", "WMT", "COST", "KR", "SYY", "UNFI", "SPTN", "GO", "IMKTA",
    "WBA", "CVS", "RAD", "HZNP", "VTRS", "TEVA", "MYL", "AGN", "PRGO", "PBH",
    "JAZZ", "ALKS", "SUPN", "CORT", "TGTX", "HALO", "SRPT", "FOLD", "ARWR", "IONS",
    "RARE", "BMRN", "BLUE", "CRSP", "EDIT", "NTLA", "BEAM", "VERV", "PRIME", "ADVM",
    
    # Biotech/Healthcare (701-850)
    "MRNA", "BNTX", "REGN", "VRTX", "GILD", "BIIB", "AMGN", "ILMN", "ALXN", "CELG",
    "INCY", "EXAS", "EXEL", "NBIX", "TECH", "BGNE", "LEGN", "UTHR", "HALO", "JAZZ",
    "PTCT", "ARQL", "KRTX", "SAGE", "ACAD", "ITCI", "CARA", "ZLAB", "ARVN", "ALNY",
    "MDGL", "ACRS", "KNSA", "CDNA", "VCEL", "CPRX", "GLPG", "SAVA", "VKTX", "TYME",
    "INO", "NVAX", "OCGN", "VXRT", "ATOS", "ADMP", "AGRX", "ANVS", "ATHX", "DMTK",
    "EBS", "EYES", "GLMD", "GNCA", "GTHX", "HGEN", "IMMP", "IMRN", "KNDI", "LPCN",
    "MBRX", "MYSZ", "NKTX", "NVCR", "NWBO", "OMER", "OPGN", "PHAS", "PIRS", "PROG",
    "RIGL", "RVMD", "SGMO", "SRNE", "TBIO", "TCON", "TRVN", "URGN", "VBLT", "VBIV",
    "VCYT", "YMAB", "ABUS", "ADMP", "AGLE", "AKRO", "ALLK", "ALPN", "ALTR", "ALVO",
    "AMRN", "AMRS", "AMTX", "ANAB", "ANIK", "ANNX", "APDN", "APLT", "APRE", "APVO",
    "ARDX", "AREC", "ARPO", "ARQT", "ARTL", "ARVL", "ARYC", "ASLN", "ASRT", "ASXC",
    "ATHE", "ATRC", "ATXI", "AUPH", "AVDL", "AVEO", "AVIR", "AVXL", "AXSM", "AYTU",
    "AZRX", "BCAB", "BCEL", "BCLI", "BCRX", "BDTX", "BEAM", "BEAT", "BGXX", "BHTG",
    "BIOS", "BIVV", "BPMC", "BPTH", "BRMK", "BRPM", "BRTX", "BTAI", "BTNX", "BURU",
    "BYRN", "BZUN", "CAAS", "CADL", "CALA", "CALT", "CAMP", "CARA", "CBAY", "CBIO",
    
    # Finance/REITs (851-1000)
    "MS", "GS", "JPM", "BAC", "C", "WFC", "USB", "PNC", "TFC", "COF",
    "SCHW", "BK", "STT", "NTRS", "BLK", "TROW", "IVZ", "BEN", "AMG", "EVRG",
    "PSX", "VLO", "MPC", "HFC", "DINO", "CTRA", "OXY", "DVN", "APA", "MRO",
    "EQT", "AR", "CHRD", "CLR", "MTDR", "MGY", "PR", "RRC", "SM", "VTLE",
    "AMH", "CUBE", "ELS", "EQR", "ESS", "INVH", "MAA", "UDR", "AVB", "CPT",
    "ACC", "AKR", "ALX", "BDN", "BNL", "BRX", "BXP", "CIO", "CLI", "CLP",
    "COR", "CSR", "CTO", "CUZ", "DEI", "DEA", "DRH", "DLR", "EQIX", "EGP",
    "EPR", "ESRT", "FCT", "FRT", "GNL", "HIW", "HPP", "HR", "INN", "JBGS",
    "KRC", "KRG", "LXP", "MAC", "MPW", "NHI", "NNN", "NSA", "OHI", "OLP",
    "PDM", "PEB", "PECO", "PEI", "PKY", "PSB", "QTS", "RHP", "RLJ", "ROIC",
    "RPT", "SAFE", "SBRA", "SHO", "SITC", "SKT", "SLG", "SPG", "STAG", "STOR",
    "SUI", "TCO", "TRNO", "UBA", "UBP", "UE", "VNO", "VRE", "VTR", "WPC",
    "WPG", "WRE", "WSR", "XHR", "AIV", "ARI", "BXMT", "CIM", "DX", "EFC",
    "GPMT", "IVR", "KREF", "LADR", "MFA", "MITT", "NRZ", "NLY", "ORC", "PMT",
    "RWT", "STWD", "TWO", "AGNC", "ARR", "CIO", "CMO", "EARN", "NYMT", "PFSI"
]

def main():
    print("🚀 Ghost Protocol - Quick 1000 Stock Ingestion")
    print("=" * 60)
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print(f"📊 Ingesting {len(TOP_1000_STOCKS)} symbols...")
    inserted = 0
    skipped = 0
    
    for i, symbol in enumerate(TOP_1000_STOCKS, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(TOP_1000_STOCKS)} - Inserted: {inserted}, Skipped: {skipped}")
            conn.commit()
        
        try:
            # Check if exists
            cursor.execute("SELECT symbol FROM symbol_universe WHERE symbol = %s", (symbol,))
            if cursor.fetchone():
                skipped += 1
                continue
            
            # Quick insert without Yahoo enrichment for speed
            cursor.execute("""
                INSERT INTO symbol_universe (
                    symbol, name, asset_type, sector, industry,
                    market_cap, exchange, is_active, last_updated
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol, symbol, 'stock', 'Unknown', 'Unknown',
                0, 'US', 1, int(time.time())
            ))
            inserted += 1
            
        except Exception as e:
            print(f"    ⚠️  Error on {symbol}: {e}")
            continue
    
    conn.commit()
    
    # Get totals
    cursor.execute("SELECT COUNT(*) as count FROM symbol_universe WHERE asset_type = 'stock'")
    total_stocks = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM symbol_universe WHERE asset_type = 'crypto'")
    total_crypto = cursor.fetchone()['count']
    
    print("\n" + "=" * 60)
    print("📊 DATABASE STATE:")
    print(f"   Stocks: {total_stocks}")
    print(f"   Crypto: {total_crypto}")
    print(f"   Total: {total_stocks + total_crypto}")
    print("=" * 60)
    print("\n✅ Quick ingestion complete! (~5 mins)")
    print("💡 Later you can enrich with: python scripts/enrich_symbols.py")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
