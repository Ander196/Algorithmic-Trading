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

from data.excluded_tickers import EXCLUDED_TICKERS, is_excluded_ticker, normalize_ticker
from data.retired_tickers import RETIRED_TICKERS

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
    "GFS", "AS", "CPRT", "PAYC", "CDNS", "ANSS", "PTC", "SWKS", "KEYS",
    "TER", "VEON", "LBTYK", "FOX", "DIS", "CMCSA", "CHTR", "T", "VZ",
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
    """Fetch stock info unless the ticker is explicitly excluded."""
    ticker = normalize_ticker(ticker)
    if is_excluded_ticker(ticker):
        return None

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
    """Fetch non-excluded ticker info with batching and rate limiting."""
    tickers = [
        normalize_ticker(t)
        for t in tickers
        if t
        and not is_excluded_ticker(t)
        and normalize_ticker(t) not in RETIRED_TICKERS
    ]
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


def fetchRetiredTickerInfo() -> pd.DataFrame:
    """Register retired symbols only when Yahoo Finance returns price history.

    Delisted symbols often no longer expose usable quote metadata through the
    info endpoint. Their static descriptive fields are curated in
    data.retired_tickers and verified against the history endpoint instead.
    The returned rows are explicitly historical-only and must remain inactive.
    """
    results = []
    tickers = list(RETIRED_TICKERS)
    print(f"\nChecking Yahoo Finance history for {len(tickers)} retired tickers...")

    for ticker in tickers:
        try:
            history = yf.Ticker(ticker).history(
                period="max",
                auto_adjust=False,
                actions=False,
            )
            if history is None or history.empty:
                print(f"  {ticker}: no Yahoo Finance history; not registering")
                continue

            results.append({
                "ticker": ticker,
                **RETIRED_TICKERS[ticker],
                "historical_only": True,
            })
            first_date = history.index.min().date()
            last_date = history.index.max().date()
            print(
                f"  {ticker}: {len(history)} Yahoo rows "
                f"({first_date} to {last_date}); historical-only"
            )
        except Exception as exc:
            print(f"  {ticker}: Yahoo history check failed; not registering ({exc})")
        time.sleep(0.3)

    return pd.DataFrame(results)


def getExistingTickers(client: supabase.Client) -> set[str]:
    """Get set of tickers already in database."""
    response = client.table("stocks").select("ticker, is_active, created_at").execute()
    return {row["ticker"] for row in response.data}


def getNowTimestamptz() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def uploadToSupabase(df: pd.DataFrame, client: supabase.Client) -> tuple[int, int]:
    """
    Upload current and historical-only tickers to the Supabase stocks table.
    - Current tickers are stored with is_active=True.
    - Curated retired tickers are stored with is_active=False.
    - Tickers absent from the combined fetch are marked inactive.
    Returns (new_count, deactivated_count)
    """
    if not df.empty:
        df = df.loc[~df["ticker"].map(is_excluded_ticker)].copy()

    existingTickers = getExistingTickers(client)
    newCount = 0
    deactivatedCount = 0

    # Do not deactivate the entire universe if a catalog fetch returns no
    # eligible symbols. Still ensure the permanent blocklist stays inactive.
    if df.empty:
        excluded_existing = {
            ticker for ticker in existingTickers if is_excluded_ticker(ticker)
        }
        for ticker in excluded_existing:
            client.table("stocks").update({"is_active": False}).eq(
                "ticker", ticker
            ).execute()
        return 0, len(excluded_existing)

    now = getNowTimestamptz()
    print(f"\nExisting tickers in DB: {len(existingTickers)}")
    print(f"New tickers to upload: {len(df)}")

    # `created_at` is the universe-run marker consumed by train-only. Refresh
    # it for every ticker in this load, not only newly discovered symbols.
    records = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        historical_flag = row.get("historical_only", False)
        historical_only = bool(historical_flag) if pd.notna(historical_flag) else False
        record = {
            "ticker": ticker, "name": row["name"], "exchange": row["exchange"],
            "sector": row["sector"], "industry": row["industry"],
            "is_active": not historical_only,
        }
        # Avoid refreshing the live-universe run marker for historical-only
        # rows. Omitted created_at preserves existing timestamps on upsert;
        # new rows receive the database default.
        if not historical_only:
            record["created_at"] = now
        records.append(record)
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

    # Enforce the exclusion in the database too; do not let a catalog refresh
    # reactivate symbols whose price history was intentionally removed.
    client.table("stocks").update({"is_active": False}).in_(
        "ticker", sorted(EXCLUDED_TICKERS)
    ).execute()

    # Fetch the current universe and a separate, verified historical-only
    # catalog. Retired symbols remain inactive but are available for backfills.
    live_df = fetchAllTickers(ALL_TICKERS)
    retired_df = fetchRetiredTickerInfo()
    df = pd.concat([live_df, retired_df], ignore_index=True)

    if df.empty:
        print("No current or historical tickers fetched, nothing to upload")
        return

    print(
        f"Prepared {len(live_df)} current tickers and "
        f"{len(retired_df)} retired tickers with available history"
    )

    # Upload current symbols as active and retired symbols as inactive.
    newCount, deactivatedCount = uploadToSupabase(df, client)

    print(f"\nDone! New: {newCount}, Deactivated: {deactivatedCount}")


if __name__ == "__main__":
    main()
