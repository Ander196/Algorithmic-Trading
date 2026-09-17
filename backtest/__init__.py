"""
Backtest Package

Walk-forward backtesting system with HMM-based regime detection.
"""
from backtest.backtester import (
    BacktestConfig,
    BacktestRecord,
    BacktestResult,
    WalkForwardBacktester,
    run_walk_forward_backtest,
)

from backtest.performance import (
    PerformanceMetrics,
    RegimeMetrics,
    ConfidenceMetrics,
    BenchmarkMetrics,
    FullPerformanceReport,
    calculate_core_metrics,
    calculate_regime_metrics,
    calculate_confidence_metrics,
    calculate_benchmark_metrics,
    generate_full_report,
    analyze_backtest,
    print_report,
    save_results,
)

from backtest.stress_test import (
    StressTestConfig,
    CrashInjectionResult,
    GapRiskResult,
    RegimeMisclassificationResult,
    StressTestReport,
    run_all_stress_tests,
    print_stress_report,
)

__all__ = [
    # Backtester
    "BacktestConfig",
    "BacktestRecord",
    "BacktestResult",
    "WalkForwardBacktester",
    "run_walk_forward_backtest",
    # Performance
    "PerformanceMetrics",
    "RegimeMetrics",
    "ConfidenceMetrics",
    "BenchmarkMetrics",
    "FullPerformanceReport",
    "calculate_core_metrics",
    "calculate_regime_metrics",
    "calculate_confidence_metrics",
    "calculate_benchmark_metrics",
    "generate_full_report",
    "analyze_backtest",
    "print_report",
    "save_results",
    # Stress Testing
    "StressTestConfig",
    "CrashInjectionResult",
    "GapRiskResult",
    "RegimeMisclassificationResult",
    "StressTestReport",
    "run_all_stress_tests",
    "print_stress_report",
]