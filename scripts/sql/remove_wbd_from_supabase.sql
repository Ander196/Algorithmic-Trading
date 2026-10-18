-- DESTRUCTIVE: permanently remove WBD-specific rows from the tables defined in storage/schema.sql.
-- Run scripts/sql/preview_wbd_removal.sql first and review all row counts.
-- This script is transactional and idempotent. It does not drop tables or schema objects.

BEGIN;

DO $$
DECLARE
    deleted_count bigint;
BEGIN
    -- FK restrict: delete user requests before deleting the stocks catalog row.
    DELETE FROM public.custom_requests
     WHERE UPPER(BTRIM(ticker)) = 'WBD';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'custom_requests: deleted % row(s)', deleted_count;

    DELETE FROM public.signals
     WHERE UPPER(BTRIM(ticker)) = 'WBD';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'signals: deleted % row(s)', deleted_count;

    DELETE FROM public.stock_prices
     WHERE UPPER(BTRIM(ticker)) = 'WBD';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'stock_prices: deleted % row(s)', deleted_count;

    DELETE FROM public.hmm_results
     WHERE UPPER(BTRIM(ticker)) = 'WBD';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'hmm_results: deleted % row(s)', deleted_count;

    DELETE FROM public.stock_risk_models
     WHERE UPPER(BTRIM(ticker)) = 'WBD';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'stock_risk_models: deleted % row(s)', deleted_count;

    -- allocation_results has an FK to market_regime_models; remove rows that
    -- use WBD as a stock OR depend on a market model whose market_ticker is WBD.
    DELETE FROM public.allocation_results AS allocations
     WHERE UPPER(BTRIM(allocations.ticker)) = 'WBD'
        OR allocations.market_model_version IN (
            SELECT model_version FROM public.market_regime_models
            WHERE UPPER(BTRIM(market_ticker)) = 'WBD'
        );
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'allocation_results: deleted % row(s)', deleted_count;

    DELETE FROM public.hmm_market_results
     WHERE UPPER(BTRIM(market_ticker)) = 'WBD';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'hmm_market_results: deleted % row(s)', deleted_count;

    -- Delete regime results referencing a WBD market model before the model.
    DELETE FROM public.market_regime_results AS results
     WHERE UPPER(BTRIM(results.market_ticker)) = 'WBD'
        OR results.market_model_version IN (
            SELECT model_version FROM public.market_regime_models
            WHERE UPPER(BTRIM(market_ticker)) = 'WBD'
        );
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'market_regime_results: deleted % row(s)', deleted_count;

    DELETE FROM public.market_regime_models
     WHERE UPPER(BTRIM(market_ticker)) = 'WBD';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'market_regime_models: deleted % row(s)', deleted_count;

    -- Final catalogue delete; FK children in stock_prices/signals are already gone.
    DELETE FROM public.stocks
     WHERE UPPER(BTRIM(ticker)) = 'WBD';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'stocks: deleted % row(s)', deleted_count;
END $$;

-- Postcondition: all ticker-keyed WBD rows below should be zero.
SELECT table_name, remaining_rows
FROM (
    SELECT 'stocks' AS table_name, COUNT(*)::bigint AS remaining_rows
      FROM public.stocks WHERE UPPER(BTRIM(ticker)) = 'WBD'
    UNION ALL
    SELECT 'stock_prices', COUNT(*)::bigint
      FROM public.stock_prices WHERE UPPER(BTRIM(ticker)) = 'WBD'
    UNION ALL
    SELECT 'signals', COUNT(*)::bigint
      FROM public.signals WHERE UPPER(BTRIM(ticker)) = 'WBD'
    UNION ALL
    SELECT 'custom_requests', COUNT(*)::bigint
      FROM public.custom_requests WHERE UPPER(BTRIM(ticker)) = 'WBD'
    UNION ALL
    SELECT 'hmm_results', COUNT(*)::bigint
      FROM public.hmm_results WHERE UPPER(BTRIM(ticker)) = 'WBD'
    UNION ALL
    SELECT 'stock_risk_models', COUNT(*)::bigint
      FROM public.stock_risk_models WHERE UPPER(BTRIM(ticker)) = 'WBD'
    UNION ALL
    SELECT 'allocation_results', COUNT(*)::bigint
      FROM public.allocation_results WHERE UPPER(BTRIM(ticker)) = 'WBD'
    UNION ALL
    SELECT 'hmm_market_results', COUNT(*)::bigint
      FROM public.hmm_market_results WHERE UPPER(BTRIM(market_ticker)) = 'WBD'
    UNION ALL
    SELECT 'market_regime_results', COUNT(*)::bigint
      FROM public.market_regime_results WHERE UPPER(BTRIM(market_ticker)) = 'WBD'
    UNION ALL
    SELECT 'market_regime_models', COUNT(*)::bigint
      FROM public.market_regime_models WHERE UPPER(BTRIM(market_ticker)) = 'WBD'
) AS remaining
ORDER BY table_name;

COMMIT;
