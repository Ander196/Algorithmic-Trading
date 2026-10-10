-- Run this first in the Supabase SQL Editor.
-- Read-only preview: records belonging to excluded tickers, records before
-- 2000-01-01, and the unique combined delete set.

WITH excluded_tickers AS (
    SELECT unnest(ARRAY[
        'ABI.BR', 'ACA.PA', 'ITX.MC', 'UCG.MI',
        'AZN.L', 'BATS.L', 'BP.L', 'BT-A.L', 'GSK.L', 'HSBA.L',
        'LSEG.L', 'NG.L', 'PRU.L', 'REL.L', 'RIO.L', 'SHEL.L',
        'ULVR.L', 'VOD.L'
    ]::text[]) AS ticker
),
ticker_counts AS (
    SELECT
        'excluded ticker'::text AS category,
        p.ticker,
        COUNT(*) AS rows_to_delete,
        MIN(p.price_date) AS first_date,
        MAX(p.price_date) AS last_date
    FROM public.stock_prices p
    WHERE UPPER(BTRIM(p.ticker)) IN (SELECT ticker FROM excluded_tickers)
    GROUP BY p.ticker
),
old_history AS (
    SELECT
        'before 2000 (all tickers)'::text AS category,
        '(all tickers)'::text AS ticker,
        COUNT(*) AS rows_to_delete,
        MIN(price_date) AS first_date,
        MAX(price_date) AS last_date
    FROM public.stock_prices
    WHERE price_date < DATE '2000-01-01'
),
combined AS (
    SELECT
        'unique combined delete set'::text AS category,
        '(unique rows)'::text AS ticker,
        COUNT(*) AS rows_to_delete,
        MIN(price_date) AS first_date,
        MAX(price_date) AS last_date
    FROM public.stock_prices
    WHERE price_date < DATE '2000-01-01'
       OR UPPER(BTRIM(ticker)) IN (SELECT ticker FROM excluded_tickers)
)
SELECT * FROM ticker_counts
UNION ALL
SELECT * FROM old_history
UNION ALL
SELECT * FROM combined
ORDER BY category, ticker;
