-- DESTRUCTIVE CLEANUP for public.stock_prices.
-- First run scripts/sql/preview_price_history_cleanup.sql and inspect the counts.
-- This removes every row for the 18 excluded tickers and all rows dated before
-- 2000-01-01 (the cutoff date itself is retained).
-- Also deactivates the tickers in public.stocks so the execution universe drops them.
-- Run in Supabase SQL Editor as a single script/transaction.

BEGIN;

UPDATE public.stocks
SET is_active = FALSE
WHERE UPPER(BTRIM(ticker)) = ANY(ARRAY[
    'ABI.BR', 'ACA.PA', 'ITX.MC', 'UCG.MI',
    'AZN.L', 'BATS.L', 'BP.L', 'BT-A.L', 'GSK.L', 'HSBA.L',
    'LSEG.L', 'NG.L', 'PRU.L', 'REL.L', 'RIO.L', 'SHEL.L',
    'ULVR.L', 'VOD.L'
]::text[]);

WITH deleted AS (
    DELETE FROM public.stock_prices
    WHERE price_date < DATE '2000-01-01'
       OR UPPER(BTRIM(ticker)) = ANY(ARRAY[
            'ABI.BR', 'ACA.PA', 'ITX.MC', 'UCG.MI',
            'AZN.L', 'BATS.L', 'BP.L', 'BT-A.L', 'GSK.L', 'HSBA.L',
            'LSEG.L', 'NG.L', 'PRU.L', 'REL.L', 'RIO.L', 'SHEL.L',
            'ULVR.L', 'VOD.L'
       ]::text[])
    RETURNING ticker, price_date
)
SELECT
    COUNT(*) AS deleted_rows_total,
    COUNT(*) FILTER (
        WHERE UPPER(BTRIM(ticker)) = ANY(ARRAY[
            'ABI.BR', 'ACA.PA', 'ITX.MC', 'UCG.MI',
            'AZN.L', 'BATS.L', 'BP.L', 'BT-A.L', 'GSK.L', 'HSBA.L',
            'LSEG.L', 'NG.L', 'PRU.L', 'REL.L', 'RIO.L', 'SHEL.L',
            'ULVR.L', 'VOD.L'
        ]::text[])
    ) AS deleted_rows_for_excluded_tickers,
    COUNT(*) FILTER (
        WHERE price_date < DATE '2000-01-01'
          AND UPPER(BTRIM(ticker)) <> ALL(ARRAY[
            'ABI.BR', 'ACA.PA', 'ITX.MC'UCG.MI',
            'AZN.L', 'BATS.L', 'BP.L', 'BT-A.L', 'GSK.L', 'HSBA.L',
            'LSEG.L', 'NG.L', 'PRU.L', 'REL.L', 'RIO.L', 'SHEL.L',
            'ULVR.L', 'VOD.L'
        ]::text[])
    ) AS deleted_pre_2000_rows_for_other_tickers,
    MIN(price_date) AS earliest_deleted_date,
    MAX(price_date) AS latest_deleted_date
FROM deleted;

COMMIT;
