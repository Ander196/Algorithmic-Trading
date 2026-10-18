-- READ-ONLY preflight for removing WBD from the project's Supabase schema.
-- Review every count before running scripts/sql/remove_wbd_from_supabase.sql.

SELECT table_name, rows_to_delete
FROM (
    SELECT 'stocks' AS table_name, COUNT(*)::bigint AS rows_to_delete
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
      FROM public.allocation_results AS allocations
     WHERE UPPER(BTRIM(allocations.ticker)) = 'WBD'
        OR allocations.market_model_version IN (
            SELECT model_version FROM public.market_regime_models
            WHERE UPPER(BTRIM(market_ticker)) = 'WBD'
        )
    UNION ALL
    SELECT 'hmm_market_results', COUNT(*)::bigint
      FROM public.hmm_market_results WHERE UPPER(BTRIM(market_ticker)) = 'WBD'
    UNION ALL
    SELECT 'market_regime_results', COUNT(*)::bigint
      FROM public.market_regime_results AS results
     WHERE UPPER(BTRIM(results.market_ticker)) = 'WBD'
        OR results.market_model_version IN (
            SELECT model_version FROM public.market_regime_models
            WHERE UPPER(BTRIM(market_ticker)) = 'WBD'
        )
    UNION ALL
    SELECT 'market_regime_models', COUNT(*)::bigint
      FROM public.market_regime_models WHERE UPPER(BTRIM(market_ticker)) = 'WBD'
) AS counts
ORDER BY table_name;
