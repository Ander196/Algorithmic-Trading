-- READ ONLY: detailed examples of rows with invalid OHLCV fields.
-- Returns at most 5000 rows; see audit_ohlcv_reason_summary.sql for full counts.
-- Reasons can overlap: inspect each row's reasons array before deciding what to repair.

WITH reasoned AS (
    SELECT
        p.ticker,
        p.price_date,
        p.open,
        p.high,
        p.low,
        p.close,
        p.adj_close,
        p.volume,
        ARRAY_REMOVE(ARRAY[
            CASE
                WHEN p.open IS NULL OR p.high IS NULL
                  OR p.low IS NULL OR p.close IS NULL
                THEN 'missing_ohlc_values'
            END,
            CASE WHEN p.volume IS NULL THEN 'missing_volume' END,
            CASE
                WHEN p.open::text ~* '^(nan|[+-]?infinity)$'
                  OR p.high::text ~* '^(nan|[+-]?infinity)$'
                  OR p.low::text ~* '^(nan|[+-]?infinity)$'
                  OR p.close::text ~* '^(nan|[+-]?infinity)$'
                THEN 'non_finite_ohlc_values'
            END,
            CASE
                WHEN p.volume::text ~* '^(nan|[+-]?infinity)$'
                THEN 'non_finite_volume'
            END,
            CASE
                WHEN (p.open IS NOT NULL AND p.open <= 0)
                  OR (p.high IS NOT NULL AND p.high <= 0)
                  OR (p.low IS NOT NULL AND p.low <= 0)
                  OR (p.close IS NOT NULL AND p.close <= 0)
                THEN 'non_positive_ohlc_values'
            END,
            CASE WHEN p.volume < 0 THEN 'negative_volume' END,
            CASE
                WHEN p.adj_close IS NOT NULL
                 AND p.adj_close::text ~* '^(nan|[+-]?infinity)$'
                THEN 'non_finite_adj_close'
            END,
            CASE
                WHEN p.adj_close IS NOT NULL AND p.adj_close <= 0
                THEN 'non_positive_adj_close'
            END,
            CASE
                WHEN p.open > 0 AND p.high > 0 AND p.low > 0 AND p.close > 0
                 AND p.open::text !~* '^(nan|[+-]?infinity)$'
                 AND p.high::text !~* '^(nan|[+-]?infinity)$'
                 AND p.low::text !~* '^(nan|[+-]?infinity)$'
                 AND p.close::text !~* '^(nan|[+-]?infinity)$'
                 AND p.high < GREATEST(p.open, p.low, p.close)
                THEN 'high_below_ohlc_max'
            END,
            CASE
                WHEN p.open > 0 AND p.high > 0 AND p.low > 0 AND p.close > 0
                 AND p.open::text !~* '^(nan|[+-]?infinity)$'
                 AND p.high::text !~* '^(nan|[+-]?infinity)$'
                 AND p.low::text !~* '^(nan|[+-]?infinity)$'
                 AND p.close::text !~* '^(nan|[+-]?infinity)$'
                 AND p.low > LEAST(p.open, p.high, p.close)
                THEN 'low_above_ohlc_min'
            END
        ], NULL) AS reasons
    FROM public.stock_prices AS p
)
SELECT
    ticker,
    price_date,
    open,
    high,
    low,
    close,
    adj_close,
    volume,
    reasons
FROM reasoned
WHERE CARDINALITY(reasons) > 0
ORDER BY price_date, ticker
LIMIT 5000;
