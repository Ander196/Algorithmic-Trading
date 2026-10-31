-- One-time cleanup: remove APC/SBNY/QRVO prices outside their S&P membership.
-- APC: membership [2000-01-01, 2019-08-09); last traded 2019-08-08.
-- SBNY: membership [2021-12-20, 2023-03-15); last traded 2023-03-10.
-- SBNY price window therefore ends exclusively on 2023-03-11.
-- QRVO: membership [2015-06-12, 2024-12-23); final included date 2024-12-20.
--
-- Run in Supabase SQL Editor after reviewing the preview result.
-- The backup table is retained so deleted rows can be inspected/restored.
-- This script does not delete valid APC/SBNY/QRVO rows within their membership.

BEGIN;

-- Run preview_apc_sbny_sp500_cleanup.sql first, then execute this transaction.

-- Keep a restorable copy of the complete rows before deleting them.
CREATE TABLE IF NOT EXISTS public.stock_prices_sp500_cleanup_backup
    (LIKE public.stock_prices INCLUDING ALL);

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
        AND price_date < DATE '2023-03-11'
    ))
    OR
    (ticker = 'QRVO' AND NOT (
        price_date >= DATE '2015-06-12'
        AND price_date < DATE '2024-12-23'
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
        AND price_date < DATE '2023-03-11'
    ))
    OR
    (ticker = 'QRVO' AND NOT (
        price_date >= DATE '2015-06-12'
        AND price_date < DATE '2024-12-23'
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
        AND price_date < DATE '2023-03-11'
    ))
    OR
    (ticker = 'QRVO' AND NOT (
        price_date >= DATE '2015-06-12'
        AND price_date < DATE '2024-12-23'
    ))
GROUP BY ticker;
