import pandas as pd
import ssl
import urllib.request

# Disable SSL verification for downloading public data
ssl._create_default_https_context = ssl._create_unverified_context

# Step 1 — Download listings
nasdaq_url  = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
nyse_url    = "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv"

df1 = pd.read_csv(nasdaq_url, sep="|", dtype=str, skipfooter=1, engine='python')
df1 = df1[['Symbol','Security Name']]
df1 = df1[df1['Symbol'].str.len()>0]

df2 = pd.read_csv(nyse_url, dtype=str)
df2 = df2[['ACT Symbol','Company Name']]
df2.columns = ['Symbol','Security Name']
df2 = df2[df2['Symbol'].str.len()>0]

df = pd.concat([df1, df2], ignore_index=True)
df = df.drop_duplicates(subset=['Symbol'])

# Step 2 — Filter out ineligible securities
blacklist = [
    "ETF","ETN","Trust","Preferred","Warrant","Notes","Bond","Fund",
    "Income","Index","ADR","Depositary"
]
pattern = "|".join(blacklist)
df_filtered = df[~df['Security Name'].str.contains(pattern, case=False, na=False)]

# Step 3 — Select top ~1,000
df_final = df_filtered.head(1000)

# Export
output_path = "supported_tickers_draft.csv"
df_final.to_csv(output_path, index=False)

print("supported_tickers_draft.csv generated successfully")
