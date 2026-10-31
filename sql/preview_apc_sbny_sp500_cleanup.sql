-- Read-only preview of APC/SBNY/QRVO rows outside their valid S&P 500 price windows.
-- APC: [2000-01-01, 2019-08-09), last traded 2019-08-08.
-- SBNY: [2021-12-20, 2023-03-11), last traded 2023-03-10.
-- QRVO: [2015-06-12, 2024-12-23), final included date 2024-12-20.

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
        AND price_date < DATE '2023-03-11'
    ))
    OR
    (ticker = 'QRVO' AND NOT (
        price_date >= DATE '2015-06-12'
        AND price_date < DATE '2024-12-23'
    ))
GROUP BY ticker
ORDER BY ticker;

-- This query must return zero rows after running cleanup_apc_sbny_outside_sp500.sql.
