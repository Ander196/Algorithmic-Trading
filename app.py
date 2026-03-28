"""
Regime-Based Trading Dashboard
A Streamlit dashboard for regime detection and trading strategy signals.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import time

from data_loader import fetchHourlyData
from indicators import calculateAllIndicators
from hmm_model import RegimeDetector
from backtester import TradingBacktester

st.set_page_config(
    page_title="Regime Trading System",
    page_icon="📈",
    layout="wide"
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0a0a14; }
    .stApp { background-color: #0a0a14; }
    .regime-bull { color: #00ff88; font-weight: bold; }
    .regime-bear { color: #ff4757; font-weight: bold; }
    .regime-crash { color: #ff0040; font-weight: bold; }
    .regime-neutral { color: #ffd93d; font-weight: bold; }
    .signal-long { background-color: #00ff88; color: #000; padding: 10px 20px; border-radius: 5px; font-weight: bold; }
    .signal-cash { background-color: #ff4757; color: #fff; padding: 10px 20px; border-radius: 5px; font-weight: bold; }
    .metric-card { background-color: #1a1a2e; padding: 15px; border-radius: 10px; border: 1px solid #2a2a4e; }
    .trade-log { font-size: 12px; }
    div[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def loadData(ticker: str, days: int):
    """Load and cache stock data."""
    return fetchHourlyData(ticker, days)


def plotCandlestickWithRegimes(df: pd.DataFrame, regimes: pd.Series) -> go.Figure:
    """Create candlestick chart with regime-colored background."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('', 'Volume')
    )

    df_plot = df.tail(500).copy()
    regimes_plot = regimes.tail(500) if regimes is not None else None

    colors = {
        'Bull': 'rgba(0, 255, 136, 0.15)',
        'Bear': 'rgba(255, 71, 87, 0.15)',
        'Crash': 'rgba(255, 0, 64, 0.2)',
        'Neutral': 'rgba(255, 217, 61, 0.08)'
    }

    if regimes_plot is not None:
        for regime in ['Bull', 'Bear', 'Crash', 'Neutral']:
            mask = regimes_plot == regime
            if mask.any():
                fig.add_hrect(
                    y0=df_plot.loc[mask, 'low'].min() * 0.998,
                    y1=df_plot.loc[mask, 'high'].max() * 1.002,
                    fillcolor=colors.get(regime, 'rgba(128,128,128,0.1)'),
                    line_width=0,
                    row='all', col=1,
                    secondary_y=False
                )

    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4757',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4757'
        ),
        row=1, col=1
    )

    if 'ema50' in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['ema50'],
                mode='lines',
                name='EMA 50',
                line=dict(color='#4ecdc4', width=1)
            ),
            row=1, col=1
        )

    if 'ema200' in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['ema200'],
                mode='lines',
                name='EMA 200',
                line=dict(color='#ffd93d', width=1)
            ),
            row=1, col=1
        )

    colors_vol = ['#00ff88' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else '#ff4757'
                  for i in range(len(df_plot))]

    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=df_plot['volume'],
            name='Volume',
            marker_color=colors_vol,
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a14',
        plot_bgcolor='#0a0a14',
        font=dict(color='#ffffff'),
        height=600,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            gridcolor='#1a1a2e',
            zerolinecolor='#2a2a4e'
        ),
        yaxis2=dict(
            gridcolor='#1a1a2e',
            zerolinecolor='#2a2a4e'
        )
    )

    fig.update_xaxes(
        gridcolor='#1a1a2e',
        zerolinecolor='#2a2a4e',
        row=1, col=1
    )
    fig.update_xaxes(
        gridcolor='#1a1a2e',
        zerolinecolor='#2a2a4e',
        row=2, col=1
    )

    return fig


def plotEquityCurve(equity_df: pd.DataFrame, initial_capital: float) -> go.Figure:
    """Plot equity curve with drawdown."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df['equity'],
            mode='lines',
            name='Strategy Equity',
            line=dict(color='#00ff88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=[initial_capital] * len(equity_df),
            mode='lines',
            name='Initial Capital',
            line=dict(color='#ffd93d', width=1, dash='dash')
        )
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a14',
        font=dict(color='#ffffff'),
        height=300,
        margin=dict(l=60, r=40, t=30, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        yaxis=dict(
            gridcolor='#1a1a2e',
            zerolinecolor='#2a2a4e',
            title='Capital ($)'
        ),
        xaxis=dict(
            gridcolor='#1a1a2e',
            zerolinecolor='#2a2a4e',
            rangeslider_visible=False
        )
    )

    return fig


def main():
    st.title("📈 Regime-Based Trading System")

    with st.sidebar:
        st.header("Configuration")
        ticker = st.selectbox(
            "Select Stock",
            ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY'],
            index=0
        )
        initial_capital = st.number_input("Initial Capital ($)", value=1000, min_value=100)
        leverage = st.number_input("Leverage", value=2.5, min_value=1.0, max_value=10.0)
        st.divider()
        st.caption("Strategy Settings")
        st.write(f"Min Confirmations: 7/8")
        st.write(f"Cooldown: 48 hours")
        st.write(f"Leverage: {leverage}x")
        st.divider()
        st.caption("Regime Detection")
        st.write("HMM Components: 7")
        st.write("Features: Returns, Range, Volume Vol")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with st.spinner(f'Loading {ticker} data...'):
        try:
            df = loadData(ticker, 730)
            df = calculateAllIndicators(df)

            detector = RegimeDetector(n_components=7)
            detector.fit(df)
            regimes = detector.predictRegimes(df)
            df['regime'] = regimes

            backtester = TradingBacktester(
                initial_capital=initial_capital,
                leverage=leverage,
                cooldown_hours=48,
                min_confirmations=7
            )

            results = backtester.runBacktest(df)
            metrics = results['metrics']
            trades = results['trades']
            equity_curve = results['equity_curve']

            current_regime = detector.getCurrentRegime(df)
            signal, confirmations_met, confirmations_list = backtester.getCurrentSignal(df, current_regime)

            latest = df.iloc[-1]

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    regime_colors = {
        'Bull': '#00ff88',
        'Bear': '#ff4757',
        'Crash': '#ff0040',
        'Neutral': '#ffd93d'
    }

    with col1:
        st.markdown("### Current Signal")
        if signal == 'LONG':
            st.markdown(
                f'<div style="background-color:#00ff88;color:#000;padding:15px 30px;border-radius:10px;text-align:center;font-size:24px;font-weight:bold;">LONG</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color:#ff4757;color:#fff;padding:15px 30px;border-radius:10px;text-align:center;font-size:24px;font-weight:bold;">CASH</div>',
                unsafe_allow_html=True
            )
        st.markdown(f"**Confirmations: {confirmations_met}/8**")

    with col2:
        st.markdown("### Detected Regime")
        regime_color = regime_colors.get(current_regime, '#ffffff')
        st.markdown(
            f'<div style="color:{regime_color};font-size:32px;font-weight:bold;text-align:center;">{current_regime.upper()}</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"Latest Price: **${latest['close']:.2f}**")

    with col3:
        st.metric(
            "Total Return",
            f"{metrics['total_return']:.2f}%",
            delta=f"Alpha: {metrics['alpha']:.2f}%"
        )

    with col4:
        st.metric(
            "Final Capital",
            f"${metrics['final_capital']:.2f}",
            delta=f"vs ${initial_capital} initial"
        )

    st.divider()

    st.plotly_chart(
        plotCandlestickWithRegimes(df, regimes),
        use_container_width=True
    )

    st.plotly_chart(
        plotEquityCurve(equity_curve, initial_capital),
        use_container_width=True
    )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    with col6:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
    with col7:
        st.metric("Total Trades", metrics['total_trades'])
    with col8:
        st.metric("Winning Trades", metrics['winning_trades'])

    st.divider()

    with st.expander("Active Confirmations (8 Total)", expanded=False):
        confirmation_status = [
            ("RSI < 90", latest['rsi'] < 90, f"Current: {latest['rsi']:.1f}"),
            ("Momentum > 1%", latest['momentum'] > 1.0, f"Current: {latest['momentum']:.2f}%"),
            ("Volatility < 6%", latest['volatility'] < 6.0, f"Current: {latest['volatility']:.2f}%"),
            ("Volume > SMA20", latest['volumeAboveSma'], f"Current: {latest['volume']/latest['volumeSma20']:.2f}x"),
            ("ADX > 25", latest['adx'] > 25, f"Current: {latest['adx']:.1f}"),
            ("Price > EMA50", latest['priceAboveEma50'], f"Current: ${latest['close']:.2f} vs EMA50: ${latest['ema50']:.2f}"),
            ("Price > EMA200", latest['priceAboveEma200'], f"Current: ${latest['close']:.2f} vs EMA200: ${latest['ema200']:.2f}"),
            ("MACD > Signal", latest['macdAboveSignal'], f"MACD: {latest['macd']:.2f} vs Signal: {latest['macdSignal']:.2f}")
        ]

        for name, met, detail in confirmation_status:
            status = "✅" if met else "❌"
            color = "#00ff88" if met else "#ff4757"
            st.markdown(f"{status} **{name}** - {detail}")

    st.divider()

    st.subheader("Trade Log")
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['pnl_pct'] = trades_df['pnl'] / initial_capital * 100
        trades_df = trades_df.sort_values('exit_time', ascending=False).head(20)

        col_trade1, col_trade2 = st.columns(2)
        with col_trade1:
            st.dataframe(
                trades_df[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'exit_reason']],
                use_container_width=True,
                hide_index=True
            )
        with col_trade2:
            st.markdown("### Trade Statistics")
            avg_pnl = trades_df['pnl'].mean()
            best_pnl = trades_df['pnl'].max()
            worst_pnl = trades_df['pnl'].min()
            st.metric("Average PnL", f"${avg_pnl:.2f}")
            st.metric("Best Trade", f"${best_pnl:.2f}")
            st.metric("Worst Trade", f"${worst_pnl:.2f}")
    else:
        st.info("No trades executed yet.")

    st.divider()

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: {ticker} hourly")


if __name__ == "__main__":
    main()
