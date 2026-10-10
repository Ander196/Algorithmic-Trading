-- DESTRUCTIVE: remove all price rows for tickers with confirmed invalid history.
-- This also deactivates those symbols in public.stocks to keep them out of the
-- current execution universe. Run scripts/sql/preview_price_history_cleanup.sql first.

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
    WHERE UPPER(BTRIM(ticker)) = ANY(ARRAY[
        'ABI.BR', 'ACA.PA', 'ITX.MC', 'UCG.MI',
        'AZN.L', 'BATS.L', 'BP.L', 'BT-A.L', 'GSK.L', 'HSBA.L',
        'LSEG.L', 'NG.L', 'PRU.L', 'REL.L', 'RIO.L', 'SHEL.L',
        'ULVR.L', 'VOD.L'
    ]::text[])
    RETURNING ticker, price_date
)
SELECT
    ticker,
    COUNT(*) AS deleted_rows,
    MIN(price_date) AS first_deleted_date,
    MAX(price_date) AS last_deleted_date
FROM deleted
GROUP BY ticker
ORDER BY ticker;

COMMIT;
