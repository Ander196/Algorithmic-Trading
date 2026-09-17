"""
ticker_loader.py

Fetches stock information (ticker, name, exchange, sector, industry) from
US and European exchanges and uploads to Supabase `stocks` table.

Usage:
    python ticker_loader.py

Environment variables required in .env:
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_ANON_KEY=your-anon-key
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
import supabase
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(".env.secrets")


# Ticker lists to fetch - can be extended
# US Indices component stocks
US_TICKERS = [
    # Tech/Communication
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CRM",
    "CSCO", "ADBE", "ACN", "IBM", "INTC", "AMD", "QCOM", "TXN", "NOW", "INTU",
    "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "PANW", "CRWD", "FTNT", "NET",
    "ZS", "PLTR", "SNOW", "DDOG", "HUBS", "MDB", "TEAM", "WDAY", "OKTA", "SPLK",
    "NXPI", "ON", "MPWR", "ADI", "MCHP", "XEL", "CTAS", "CDW", "MNST", "KDP",
    "GFS", "AS", "CPRT", "PAYC", "CDNS", "ANSS", "PTC", "SWKS", "QRVO", "KEYS",
    "TER", "VEON", "LBTYK", "WBD", "FOX", "DIS", "CMCSA", "CHTR", "T", "VZ",
    # Consumer/Retail
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "DG", "DLTR",
    "BBY", "ROST", "BURL", "DHI", "LEN", "PHM", "DRI", "TAP", "MNST", "KDP",
    "EL", "CL", "PG", "KO", "PEP", "PM", "MO", "GIS", "K", "HSY",
    "MDLZ", "KHC", "CLX", "HRL", "SYY", "DGX", "LH", "IQV", "IDXX", "REGN",
    "VRTX", "GILD", "BIIB", "MRNA", "NVAX", "CTVA", "FANG", "DVN", "EOG", "PXD",
    "SLB", "HAL", "BKR", "SLB", "OXY", "COP", "PSX", "VLO", "MPC", "DK",
    # Financials
    "BRK.B", "SPGI", "C", "USB", "PNC", "TFC", "COF", "AFL", "MET", "PRU",
    "L", "TRV", "AON", "MMC", "WTW", "BRO", "FDS", "NDAQ", "ICE", "CME",
    "CB", "TRG", "AJG", "AIG", "SNA", "BLK", "BK", "STT", "NTRS", "SPGI",
    # Industrial
    "CAT", "DE", "BA", "GE", "HON", "UPS", "FDX", "RTX", "LMT", "NOC",
    "GD", "LHX", "HII", "LDOS", "SAIC", "CARR", "OTIS", "WM", "RSG", "NSC",
    "CSX", "UNP", "ETN", "EMR", "ROK", "PH", "ITW", "SWK", "IR", "CMI",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "VRTX", "BIIB", "REGN", "MRNA", "ISRG", "DXCM", "HOLX", "IDXX",
    "ALGN", "ZBH", "STE", "Hologic", "EW", "MDT", "SYK", "BSX", "EW", "ZHT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "OXY",
    "HAL", "BKR", "DVN", "FANG", "EQT", "OKE", "WMB", "KMI", "TRGP", "ET",
    # Utilities/REITs
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "PEG",
    "PSA", "AMT", "PLD", "EQIX", "SPG", "O", "WELL", "DLR", "AVB", "EQR",
    # Materials
    "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "STLD", "RS", "PPG", "ALB",
    "CE", "CTVA", "FMC", "MOS", "CF", "SMG",
    # Indexes
    "SPY", "QQQ"
]

# European stocks (major exchanges)
EU_TICKERS = [
    # UK (LSE)
    "SHEL.L", "HSBA.L", "AZN.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "BATS.L",
    "REL.L", "NVO.L", "VOD.L", "LSEG.L", "NG.L", "BT-A.L", " SSE.L", "PRU.L",
    # Germany (XETRA)
    "SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "MBG.DE", "BMW.DE", "BAYN.DE", "ADS.DE",
    "DBK.DE", "DTE.DE", "ENR.DE", "VNA.DE", "RWE.DE", "FRE.DE", "HEI.DE", "MTX.DE",
    # France (Euronext)
    "OR.PA", "MC.PA", "CAP.PA", "SAN.PA", "BNP.PA", "GLE.PA", "ACA.PA", "CS.PA",
    "AIR.PA", "VLO.PA", "ENGI.PA", "FDJ.PA", "RMS.PA", "KER.PA", "LR.PA", "TTE.PA",
    # Netherlands (Euronext)
    "ASML.AS", "INGA.AS", "UNA.AS", "ADYEN.AS", "EXM.AS", "HEIA.AS", "PHIA.AS",
    # Belgium (Euronext)
    "ABI.BR", "UCB.BR", "KBC.BR", "ENGB.BR", "COLR.BR",
    # Spain (BME)
    "SAN.MC", "BBVA.MC", "ITX.MC", "TEF.MC", "REP.MC", "FER.MC", "ACS.MC", "AENA.MC",
    # Italy (FTSE MIB)
    "ISP.MI", "UCG.MI", "ENEL.MI", "ENI.MI", "TIT.MI", "G.MI", "PRY.MI", "MONC.MI",
    # Switzerland (SIX)
    "NESN.SW", "ROG.SW", "NVN.SW", "UBS.SW", "CSGN.SW", "ZURN.SW", "SLHN.SW", "SGSN.SW",
    # Sweden (Nasdaq Stockholm)
    "VOLV-B.SE", "ERIC-B.SE", "SEB-A.SE", "SHB-A.SE", "TEL2-B.SE", "ALFA.SE", "SKA-B.SE",
    # Denmark (Nasdaq Copenhagen)
    "NOVO-B.DK", "DSV.DK", "CARL-B.DK", "ORSTED.DK", "PNDORA.DK",
    # Norway (Oslo Bors)
    "EQNR.OL", "DNB.OL", "TEL.OL", "YAR.OL", "ORK.OL", "MOWI.OL", "SALM.OL",
    # Finland (Nasdaq Helsinki)
    "NDA-FI.HE", "SAMPO.HE", "KESKOB.HE", "STERV.HE", "METSO.HE", "NESTE.HE",
    # Portugal (Euronext Lisbon)
    "EDP.LS", "EGLW.LS", "GALP.LS", "JMT.LS", "NHL.LS",
    # Austria (WBAG)
    "EOS.VI", "AND.VI", "CAI.VI", "ANDR.VI", "OMV.VI", "VER.VI", "WIE.VI",
    # Ireland (Euronext Dublin)
    "CRG.IE", "KRON.IE", "EG7.IE", "ALH.IE", "HSIF.IE",
    # Poland (Warsaw Stock Exchange)
    "PKN.PW", "PKO.PW", "PEO.PW", "PZU.PW", "KGH.PW", "CDR.PW", "ALE.PW",
    # Czech (Prague Stock Exchange)
    "CEZ.PR", "KOMB.PR", "MONET.PR", "ERBA.PR",
]

# Combined ticker lists
ALL_TICKERS = list(set(US_TICKERS + EU_TICKERS))


def fetchTickerInfo(ticker: str, retries: int = 2) -> dict | None:
    """Fetch stock info from yfinance with retry logic."""
    for attempt in range(retries):
        try:
            info = yf.Ticker(ticker).info
            if info is None or info.get("regularMarketPrice") is None:
                return None

            # Extract relevant fields
            return {
                "ticker": ticker,
                "name": info.get("shortName") or info.get("longName", "Unknown"),
                "exchange": info.get("exchange", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
            }
        except Exception:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            continue
    return None


def fetchAllTickers(tickers: list[str], batchSize: int = 50, delay: float = 0.5) -> pd.DataFrame:
    """Fetch info for all tickers with batching and rate limiting."""
    results = []
    total = len(tickers)

    print(f"Fetching info for {total} tickers...")

    for i in range(0, total, batchSize):
        batch = tickers[i:i + batchSize]
        print(f"Processing batch {i // batchSize + 1}/{(total + batchSize - 1) // batchSize}: {batch[:3]}...")

        for ticker in batch:
            info = fetchTickerInfo(ticker)
            if info:
                results.append(info)
            time.sleep(delay)

        # Progress indicator
        progress = min(i + batchSize, total)
        print(f"  Progress: {progress}/{total} ({100 * progress / total:.1f}%)")

    df = pd.DataFrame(results)
    print(f"Successfully fetched {len(results)}/{total} tickers")
    return df


def getExistingTickers(client: supabase.Client) -> set[str]:
    """Get set of tickers already in database."""
    response = client.table("stocks").select("ticker, is_active, created_at").execute()
    return {row["ticker"] for row in response.data}


def getNowTimestamptz() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def uploadToSupabase(df: pd.DataFrame, client: supabase.Client) -> tuple[int, int]:
    """
    Upload tickers to Supabase stocks table.
    - Inserts new tickers with is_active=True
    - Marks missing tickers as is_active=False
    Returns (new_count, deactivated_count)
    """
    existingTickers = getExistingTickers(client)
    now = getNowTimestamptz()
    newCount = 0
    deactivatedCount = 0

    print(f"\nExisting tickers in DB: {len(existingTickers)}")
    print(f"New tickers to upload: {len(df)}")

    # `created_at` is the universe-run marker consumed by train-only. Refresh
    # it for every ticker in this load, not only newly discovered symbols.
    records = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        records.append({
            "ticker": ticker, "name": row["name"], "exchange": row["exchange"],
            "sector": row["sector"], "industry": row["industry"],
            "is_active": True, "created_at": now,
        })
        if ticker not in existingTickers:
            newCount += 1

    if records:
        client.table("stocks").upsert(records, on_conflict="ticker").execute()
        print(f"  Upserted {len(records)} tickers for execution {now}")

    # Deactivate tickers no longer in the fetched list
    fetchedTickers = set(df["ticker"].tolist())
    missingTickers = existingTickers - fetchedTickers

    if missingTickers:
        print(f"Deactivating {len(missingTickers)} tickers no longer in fetched list...")
        for ticker in missingTickers:
            client.table("stocks").update({"is_active": False}).eq("ticker", ticker).execute()
            deactivatedCount += 1
        print(f"  Deactivated {deactivatedCount} tickers")

    return newCount, deactivatedCount


def main() -> None:
    """Main entry point."""
    # Load environment variables
    supabaseUrl = os.getenv("SUPABASE_URL")
    supabaseKey = os.getenv("SUPABASE_ANON_KEY")

    if not supabaseUrl or not supabaseKey:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")

    # Initialize Supabase client
    client = supabase.create_client(supabaseUrl, supabaseKey)

    # Fetch ticker info
    df = fetchAllTickers(ALL_TICKERS)

    if df.empty:
        print("No tickers fetched, nothing to upload")
        return

    # Upload to Supabase
    newCount, deactivatedCount = uploadToSupabase(df, client)

    print(f"\nDone! New: {newCount}, Deactivated: {deactivatedCount}")


if __name__ == "__main__":
    main()
