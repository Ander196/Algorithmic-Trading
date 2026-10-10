-- READ ONLY: counts of invalid rows by ticker and validation reason.
-- Reason counts are non-mutually-exclusive: one row may have multiple reasons.
-- No data is modified by this query.

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
),
invalid AS (
    SELECT r.ticker, r.price_date, unnest(r.reasons) AS reason
    FROM reasoned AS r
    WHERE CARDINALITY(r.reasons) > 0
),
ticker_totals AS (
    SELECT
        ticker,
        COUNT(DISTINCT price_date) AS total_invalid_rows_for_ticker
    FROM reasoned
    WHERE CARDINALITY(reasons) > 0
    GROUP BY ticker
)
SELECT
    i.ticker,
    t.total_invalid_rows_for_ticker,
    i.reason,
    COUNT(DISTINCT i.price_date) AS rows_with_reason,
    MIN(i.price_date) AS first_date,
    MAX(i.price_date) AS last_date
FROM invalid AS i
JOIN ticker_totals AS t USING (ticker)
GROUP BY i.ticker, t.total_invalid_rows_for_ticker, i.reason
ORDER BY i.ticker, i.reason;
