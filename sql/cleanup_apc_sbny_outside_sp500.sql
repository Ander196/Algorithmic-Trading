-- One-time cleanup: remove APC/SBNY prices outside their S&P 500 membership.
-- APC: membership [2000-01-01, 2019-08-09)
-- SBNY: membership [2021-12-20, 2023-03-15)
--
-- Run in Supabase SQL Editor after reviewing the preview result.
-- The backup table is retained so deleted rows can be inspected/restored.
-- This script does not delete valid APC/SBNY rows within their membership.

BEGIN;

-- Preview the exact rows targeted by the cleanup.
SELECT
    ticker,
    COUNT(*) AS rows_to_remove,
    MIN(price_date) AS first_price_date,
    MAX(price_date) AS last_price_date
FROM public.stock_prices
WHERE
    (ticker = 'APC' AND NOT (
        price_date >= DATE '2000-01-01'
        AND price_date < DATE '2019-08-09'
    ))
    OR
    (ticker = 'SBNY' AND NOT (
        price_date >= DATE '2021-12-20'
        AND price_date < DATE '2023-03-15'
    ))
GROUP BY ticker
ORDER BY ticker;

-- Keep a restorable copy of the complete rows before deleting them.
CREATE TABLE IF NOT EXISTS public.stock_prices_sp500_cleanup_backup
    (LIKE public.stock_prices INCLUDING DEFAULTS);

INSERT INTO public.stock_prices_sp500_cleanup_backup
SELECT *
FROM public.stock_prices
WHERE
    (ticker = 'APC' AND NOT (
        price_date >= DATE '2000-01-01'
        AND price_date < DATE '2019-08-09'
    ))
    OR
    (ticker = 'SBNY' AND NOT (
        price_date >= DATE '2021-12-20'
        AND price_date < DATE '2023-03-15'
    ))
ON CONFLICT DO NOTHING;

DELETE FROM public.stock_prices
WHERE
    (ticker = 'APC' AND NOT (
        price_date >= DATE '2000-01-01'
        AND price_date < DATE '2019-08-09'
    ))
    OR
    (ticker = 'SBNY' AND NOT (
        price_date >= DATE '2021-12-20'
        AND price_date < DATE '2023-03-15'
    ));

COMMIT;

-- Post-cleanup verification: this should return zero rows.
SELECT ticker, COUNT(*) AS out_of_membership_rows
FROM public.stock_prices
WHERE
    (ticker = 'APC' AND NOT (
        price_date >= DATE '2000-01-01'
        AND price_date < DATE '2019-08-09'
    ))
    OR
    (ticker = 'SBNY' AND NOT (
        price_date >= DATE '2021-12-20'
        AND price_date < DATE '2023-03-15'
    ))
GROUP BY ticker;
