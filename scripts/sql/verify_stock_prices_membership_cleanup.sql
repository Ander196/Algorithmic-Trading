-- READ-ONLY post-cleanup verification for batch
-- a6a2a11d-5dd6-4aa1-bb3e-202610310001
--
-- Run this file as one SQL statement. The output is a unified list of checks.
-- Immediately after cleanup (before later ingestion), stock_prices was expected
-- to contain 2,507,757 rows. The total check uses a minimum because new valid
-- rows may have been ingested since cleanup.
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
                SELECT 1 FROM batch b WHERE b.original_id = p.id
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
category_actual AS (
    SELECT
        b.classification,
        COUNT(*) AS backed_up_rows,
        COUNT(p.id) AS matching_live_ticker_date_keys
    FROM batch AS b
    LEFT JOIN public.stock_prices AS p
      ON p.ticker = b.ticker
     AND p.price_date = b.price_date
    GROUP BY b.classification
),
results AS (
    SELECT
        'batch_integrity'::text AS check_group,
        'backup_rows'::text AS check_name,
        m.backup_rows::bigint AS actual_value,
        234487::bigint AS expected_value,
        (m.backup_rows = 234487) AS passed,
        'Must match all rows in the archived batch'::text AS note
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'distinct_original_ids',
        m.distinct_original_ids,
        234487,
        (m.distinct_original_ids = 234487),
        'One unique original primary key per backup row'
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'distinct_ticker_dates',
        m.distinct_ticker_dates,
        234487,
        (m.distinct_ticker_dates = 234487),
        'One unique ticker/date key per backup row'
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'before_first_membership',
        m.before_first_rows,
        215403,
        (m.before_first_rows = 215403),
        'Original audited class count'
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'between_membership_periods',
        m.between_period_rows,
        19054,
        (m.between_period_rows = 19054),
        'Original audited class count'
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'batch_integrity',
        'after_final_membership',
        m.after_final_rows,
        30,
        (m.after_final_rows = 30),
        'Original audited class count'
    FROM batch_metrics AS m

    UNION ALL
    SELECT
        'live_table',
        'total_stock_prices_rows',
        l.total_stock_prices_rows,
        2507757,
        (l.total_stock_prices_rows >= 2507757),
        'Expected count immediately after cleanup; later valid ingestion may increase it'
    FROM live_metrics AS l

    UNION ALL
    SELECT
        'live_table',
        'remaining_original_ids',
        l.remaining_original_ids,
        0,
        (l.remaining_original_ids = 0),
        'No original backup primary key may remain in stock_prices'
    FROM live_metrics AS l

    UNION ALL
    SELECT
        'live_table',
        'remaining_original_ticker_date_keys',
        l.remaining_original_keys,
        0,
        (l.remaining_original_keys = 0),
        'No original backup ticker/date pair may remain in stock_prices'
    FROM live_metrics AS l

    UNION ALL
    SELECT
        'retained_ticker_samples',
        s.ticker,
        s.actual_rows,
        s.expected_rows,
        (s.actual_rows = s.expected_rows),
        s.purpose
    FROM sample_actual AS s

    UNION ALL
    SELECT
        'per_category_live_keys',
        c.classification,
        c.matching_live_ticker_date_keys,
        0,
        (c.matching_live_ticker_date_keys = 0),
        'All archived keys in this class should be absent from live stock_prices'
    FROM category_actual AS c

    UNION ALL
    SELECT
        'per_category_backup_rows',
        c.classification,
        c.backed_up_rows,
        CASE c.classification
            WHEN 'before_first_membership' THEN 215403
            WHEN 'between_membership_periods' THEN 19054
            WHEN 'after_final_membership' THEN 30
        END,
        CASE c.classification
            WHEN 'before_first_membership' THEN c.backed_up_rows = 215403
            WHEN 'between_membership_periods' THEN c.backed_up_rows = 19054
            WHEN 'after_final_membership' THEN c.backed_up_rows = 30
            ELSE FALSE
        END,
        'Archived rows in the backup batch by classification'
    FROM category_actual AS c
)
SELECT
    check_group,
    check_name,
    actual_value,
    expected_value,
    passed,
    note
FROM results
ORDER BY check_group, check_name;
