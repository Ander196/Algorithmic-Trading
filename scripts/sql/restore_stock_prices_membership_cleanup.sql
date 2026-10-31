-- STEP 3/3: Restore a previously archived membership-cleanup batch.
--
-- Restore is idempotent. It preserves original stock_prices.id values and the
-- original OHLCV/inserted_at fields. It refuses conflicting IDs or ticker/date
-- keys rather than overwriting current rows.

BEGIN;

LOCK TABLE public.stock_prices IN SHARE ROW EXCLUSIVE MODE NOWAIT;

DO $restore$
DECLARE
    v_batch uuid := 'a6a2a11d-5dd6-4aa1-bb3e-202610310001'::uuid;
    v_backup_count bigint;
    v_conflicts bigint;
    v_restored bigint;
    v_final_count bigint;
BEGIN
    SELECT COUNT(*) INTO v_backup_count
    FROM public.stock_prices_membership_cleanup_backup
    WHERE backup_batch_id = v_batch;

    IF v_backup_count <> 234487 THEN
        RAISE EXCEPTION
            'Restore batch incomplete: expected %, found %.',
            234487, v_backup_count;
    END IF;

    -- If an original id already exists, it must be the exact original row.
    SELECT COUNT(*) INTO v_conflicts
    FROM public.stock_prices_membership_cleanup_backup AS b
    JOIN public.stock_prices AS p
      ON p.id = b.original_id
    WHERE b.backup_batch_id = v_batch
      AND ROW(
            p.ticker, p.price_date, p.open, p.high, p.low, p.close,
            p.adj_close, p.volume, p.inserted_at
          ) IS DISTINCT FROM ROW(
            b.ticker, b.price_date, b.open, b.high, b.low, b.close,
            b.adj_close, b.volume, b.original_inserted_at
          );

    IF v_conflicts <> 0 THEN
        RAISE EXCEPTION 'Found % conflicting original IDs; restore aborted.', v_conflicts;
    END IF;

    -- Do not allow a ticker/date pair to be restored under a different id.
    SELECT COUNT(*) INTO v_conflicts
    FROM public.stock_prices_membership_cleanup_backup AS b
    JOIN public.stock_prices AS p
      ON p.ticker = b.ticker
     AND p.price_date = b.price_date
     AND p.id <> b.original_id
    WHERE b.backup_batch_id = v_batch;

    IF v_conflicts <> 0 THEN
        RAISE EXCEPTION 'Found % conflicting ticker/date keys; restore aborted.', v_conflicts;
    END IF;

    INSERT INTO public.stock_prices (
        id, ticker, price_date, open, high, low, close,
        adj_close, volume, inserted_at
    )
    OVERRIDING SYSTEM VALUE
    SELECT
        b.original_id, b.ticker, b.price_date, b.open, b.high, b.low, b.close,
        b.adj_close, b.volume, b.original_inserted_at
    FROM public.stock_prices_membership_cleanup_backup AS b
    WHERE b.backup_batch_id = v_batch
      AND NOT EXISTS (
          SELECT 1
          FROM public.stock_prices AS p
          WHERE p.id = b.original_id
      );

    GET DIAGNOSTICS v_restored = ROW_COUNT;

    SELECT COUNT(*) INTO v_final_count
    FROM public.stock_prices_membership_cleanup_backup AS b
    JOIN public.stock_prices AS p
      ON p.id = b.original_id
    WHERE b.backup_batch_id = v_batch
      AND ROW(
            p.ticker, p.price_date, p.open, p.high, p.low, p.close,
            p.adj_close, p.volume, p.inserted_at
          ) IS NOT DISTINCT FROM ROW(
            b.ticker, b.price_date, b.open, b.high, b.low, b.close,
            b.adj_close, b.volume, b.original_inserted_at
          );

    IF v_final_count <> 234487 THEN
        RAISE EXCEPTION
            'Restore verification failed: expected %, found %.',
            234487, v_final_count;
    END IF;

    PERFORM setval(
        pg_get_serial_sequence('public.stock_prices', 'id'),
        GREATEST((SELECT COALESCE(MAX(id), 1) FROM public.stock_prices), 1),
        EXISTS (SELECT 1 FROM public.stock_prices)
    );

    RAISE NOTICE 'Restored % missing rows; verified % rows in total.',
        v_restored, v_final_count;
END
$restore$;

COMMIT;

SELECT
    COUNT(*) AS verified_restored_rows,
    COUNT(*) FILTER (WHERE p.id IS NULL) AS still_missing_rows
FROM public.stock_prices_membership_cleanup_backup AS b
LEFT JOIN public.stock_prices AS p
  ON p.id = b.original_id
WHERE b.backup_batch_id = 'a6a2a11d-5dd6-4aa1-bb3e-202610310001'::uuid;
