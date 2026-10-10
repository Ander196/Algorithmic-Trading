-- DESTRUCTIVE: remove all stock_prices observations before 2000-01-01.
-- The cutoff date itself is retained. Run the read-only preview first.

BEGIN;

WITH deleted AS (
    DELETE FROM public.stock_prices
    WHERE price_date < DATE '2000-01-01'
    RETURNING ticker, price_date
)
SELECT
    COUNT(*) AS deleted_rows,
    MIN(price_date) AS earliest_deleted_date,
    MAX(price_date) AS latest_deleted_date
FROM deleted;

COMMIT;
