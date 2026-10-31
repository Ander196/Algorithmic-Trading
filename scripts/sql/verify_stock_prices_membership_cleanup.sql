-- READ-ONLY post-cleanup verification for batch
-- a6a2a11d-5dd6-4aa1-bb3e-202610310001
--
-- Expected immediately after the cleanup, before any unrelated new ingestion:
--   public.stock_prices total rows: 2,507,757
--   archival backup rows:          234,487
--   live rows matching original backup id: 0
--   live rows matching original ticker/date key: 0
--
-- If another ingestion job ran after cleanup, total_stock_prices may be higher.
-- The batch-specific checks remain valid and should still report zero live keys.
--
-- This script performs SELECT statements only.

WITH
params AS (
    SELECT 'a6a2a11d-5dd6-4aa1-bb3e-202610310001'::uuid AS batch_id
),
batch AS (
    SELECT b.*
    FROM public.stock_prices_membership_cleanup_backup AS b
    CROSS JOIN params AS p
    WHERE b.backup_batch_id = p.batch_id
),
batch_metrics AS (
    SELECT
        COUNT(*) AS backup_rows,
        COUNT(DISTINCT original_id) AS distinct_original_ids,
        COUNT(DISTINCT (ticker, price_date)) AS distinct_ticker_dates,
        COUNT(*) FILTER (
            WHERE classification = 'before_first_membership'
        ) AS before_first_rows,
        COUNT(*) FILTER (
            WHERE classification = 'between_membership_periods'
        ) AS between_period_rows,
        COUNT(*) FILTER (
            WHERE classification = 'after_final_membership'
        ) AS after_final_rows
    FROM batch
),
live_metrics AS (
    SELECT
        COUNT(*) AS total_stock_prices_rows,
        COUNT(*) FILTER (
            WHERE EXISTS (
                SELECT 1
                FROM batch b
                WHERE b.original_id = p.id
            )
        ) AS remaining_original_ids,
        COUNT(*) FILTER (
            WHERE EXISTS (
                SELECT 1
                FROM batch b
                WHERE b.ticker = p.ticker
                  AND b.price_date = p.price_date
            )
        ) AS remaining_original_keys
    FROM public.stock_prices AS p
),
sample_expected (ticker, expected_rows, purpose) AS (
    VALUES
        ('EQT', 3516::bigint, 'keep both valid membership periods'),
        ('STLD', 967::bigint, 'keep prices from S&P 500 entry onward'),
        ('ASML.AS', 6887::bigint, 'never_member scope intentionally retained'),
        ('QQQ', 6748::bigint, 'manual-review benchmark retained'),
        ('OKTA', 2405::bigint, 'future membership pending source update')
),
sample_actual AS (
    SELECT
        e.ticker,
        e.expected_rows,
        e.purpose,
        COUNT(p.id) AS actual_rows
    FROM sample_expected AS e
    LEFT JOIN public.stock_prices AS p
      ON p.ticker = e.ticker
    GROUP BY e.ticker, e.expected_rows, e.purpose
),
results AS (
    SELECT
        'batch_integrity'::text AS check_group,
        'backup_rows'::text AS check_name,
        m.backup_rows::bigint AS actual_value,
        234487::bigint AS expected_value,
        (m.backup_rows = 234487) AS passed
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'distinct_original_ids',
        m.distinct_original_ids,
        234487,
        (m.distinct_original_ids = 234487)
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'distinct_ticker_dates',
        m.distinct_ticker_dates,
        234487,
        (m.distinct_ticker_dates = 234487)
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'before_first_membership',
        m.before_first_rows,
        215403,
        (m.before_first_rows = 215403)
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'between_membership_periods',
        m.between_period_rows,
        19054,
        (m.between_period_rows = 19054)
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'after_final_membership',
        m.after_final_rows,
        30,
        (m.after_final_rows = 30)
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'live_table',
        'total_stock_prices_rows',
        l.total_stock_prices_rows,
        2507757,
        (l.total_stock_prices_rows = 2507757)
    FROM live_metrics AS l

    UNION ALL
    SELECT
        'live_table',
        'remaining_original_ids',
        l.remaining_original_ids,
        0,
        (l.remaining_original_ids = 0)
    FROM live_metrics AS l

    UNION ALL
    SELECT
        'live_table',
        'remaining_original_ticker_date_keys',
        l.remaining_original_keys,
        0,
        (l.remaining_original_keys = 0)
    FROM live_metrics AS l

    UNION ALL
    SELECT
        'retained_ticker_samples',
        s.ticker,
        s.actual_rows,
        s.expected_rows,
        (s.actual_rows = s.expected_rows)
    FROM sample_actual AS s
)
SELECT
    check_group,
    check_name,
    actual_value,
    expected_value,
    passed
FROM results
ORDER BY check_group, check_name;

-- Detailed sample report, easier to inspect manually.
SELECT
    s.ticker,
    s.purpose,
    s.expected_rows,
    s.actual_rows,
    (s.actual_rows = s.expected_rows) AS passed
FROM sample_actual AS s
ORDER BY s.ticker;

-- Per-category backup retention and live-key check.
SELECT
    b.classification,
    COUNT(*) AS backed_up_rows,
    COUNT(p.id) AS matching_live_ticker_date_keys,
    COUNT(*) - COUNT(p.id) AS absent_from_live_table
FROM batch AS b
LEFT JOIN public.stock_prices AS p
  ON p.ticker = b.ticker
 AND p.price_date = b.price_date
GROUP BY b.classification
ORDER BY b.classification;
