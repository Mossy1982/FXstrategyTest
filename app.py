"""
Forex Risk Management & Position Sizing Dashboard
Author: Senior Quantitative Developer
Purpose: Risk mitigation-first dashboard for ZAR-based trading account
Tech Stack: Streamlit, Plotly, Pandas, NumPy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from risk_engine import RiskCalculationEngine
from position_sizing import PositionSizer
from monte_carlo import MonteCarloSimulator

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Forex Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Forex Risk Management & Position Sizing Dashboard")
st.markdown("**Risk Mitigation First | ZAR Base Currency | Dynamic Parameter Adjustment**")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None


# ============================================================================
# SECTION 1: CONTROL PANEL (SIDEBAR)
# ============================================================================
st.sidebar.header("⚙️ CONTROL PANEL")
st.sidebar.markdown("---")

with st.sidebar:
    st.subheader("Account Parameters")
    account_balance_zar = st.number_input(
        "Account Balance (ZAR)",
        min_value=1000.0,
        value=50000.0,
        step=1000.0,
        help="Your trading account balance in South African Rand"
    )
    
    risk_percentage = st.slider(
        "Risk per Trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Maximum percentage of account to risk on a single trade"
    )
    
    st.subheader("Exchange Rate")
    usd_zar_rate = st.number_input(
        "USD/ZAR Exchange Rate",
        min_value=5.0,
        max_value=25.0,
        value=18.50,
        step=0.01,
        help="Current USD/ZAR exchange rate for risk capital conversion"
    )
    
    st.subheader("Trading Parameters")
    asset = st.selectbox(
        "Asset Selection",
        ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "S&P 500"],
        help="Select the trading instrument"
    )
    
    stop_loss_pips = st.number_input(
        "Stop Loss (Pips/Ticks)",
        min_value=1,
        max_value=500,
        value=50,
        step=1,
        help="Stop loss distance in pips/ticks from entry"
    )
    
    st.subheader("Historical Performance Metrics")
    win_rate = st.slider(
        "Historical Win Rate (%)",
        min_value=10.0,
        max_value=90.0,
        value=55.0,
        step=1.0,
        help="Historical percentage of winning trades"
    )
    
    reward_to_risk = st.number_input(
        "Reward-to-Risk Ratio (R:R)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Average profit per winning trade / loss per losing trade"
    )
    
    st.markdown("---")
    st.sidebar.info(
        "💡 **Tip:** Adjust parameters in real-time to see immediate impact on "
        "position sizing and risk metrics."
    )

# ============================================================================
# INITIALIZE CALCULATION ENGINES
# ============================================================================
try:
    risk_engine = RiskCalculationEngine(
        account_balance_zar=account_balance_zar,
        risk_percentage=risk_percentage,
        usd_zar_rate=usd_zar_rate
    )

    position_sizer = PositionSizer(asset)

    # Calculate key metrics
    zar_risk_capital = risk_engine.calculate_zar_risk()
    usd_risk_capital = risk_engine.convert_to_usd(zar_risk_capital)
    lot_size = position_sizer.calculate_lot_size(
        usd_risk_capital=usd_risk_capital,
        stop_loss_pips=stop_loss_pips
    )
    ev_per_trade = risk_engine.calculate_expected_value(
        win_rate=win_rate / 100,
        reward_to_risk=reward_to_risk
    )
    kelly_fraction = risk_engine.calculate_kelly_criterion(
        win_rate=win_rate / 100,
        reward_to_risk=reward_to_risk
    )
    half_kelly = kelly_fraction / 2

except Exception as e:
    st.error(f"Error initializing calculation engines: {e}")
    st.stop()

# ============================================================================
# SECTION 2: SIZING CONSOLE (KPI METRICS)
# ============================================================================
st.markdown("---")
st.subheader("📈 SIZING CONSOLE - KEY PERFORMANCE INDICATORS")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Max ZAR Risk per Trade",
        value=f"ZAR {zar_risk_capital:,.2f}",
        delta=f"{risk_percentage}% of account"
    )

with col2:
    st.metric(
        label="Max USD Risk per Trade",
        value=f"USD {usd_risk_capital:,.2f}",
        delta=f"@ {usd_zar_rate} rate"
    )

with col3:
    st.metric(
        label="Recommended Lot Size",
        value=f"{lot_size:.3f}",
        delta=f"{asset}"
    )

with col4:
    ev_color = "green" if ev_per_trade > 0 else "red"
    st.metric(
        label="Expected Value per Trade",
        value=f"USD {ev_per_trade:,.4f}",
        delta="Positive" if ev_per_trade > 0 else "Negative"
    )

# ============================================================================
# KELLY CRITERION ANALYSIS
# ============================================================================
st.markdown("---")
st.subheader("🎲 KELLY CRITERION & SIZING BENCHMARKS")

col_kelly1, col_kelly2, col_kelly3 = st.columns(3)

with col_kelly1:
    st.metric(
        label="Full Kelly Fraction",
        value=f"{kelly_fraction:.4f}",
        delta="Aggressive (not recommended)"
    )

with col_kelly2:
    st.metric(
        label="Half Kelly Fraction",
        value=f"{half_kelly:.4f}",
        delta="Conservative (recommended)"
    )

with col_kelly3:
    adjusted_lot = lot_size * half_kelly
    st.metric(
        label="Half-Kelly Adjusted Lot Size",
        value=f"{adjusted_lot:.3f}",
        delta=f"Reduced risk"
    )

kelly_explanation = f"""
**Kelly Criterion Interpretation:**
- **Win Rate:** {win_rate:.1f}%
- **R:R Ratio:** {reward_to_risk:.2f}:1
- **Full Kelly:** {kelly_fraction:.4f} (Highly aggressive, risk of ruin exists)
- **Half Kelly:** {half_kelly:.4f} (Conservative, smoother equity curve)
- **Quarter Kelly:** {kelly_fraction/4:.4f} (Ultra-conservative)

*Recommendation: Use Half-Kelly for production trading to minimize drawdown risk.*
"""
st.info(kelly_explanation)

# ============================================================================
# SECTION 3: MONTE CARLO SIMULATION
# ============================================================================
st.markdown("---")
st.subheader("🎯 DRAWDOWN & RUIN SIMULATION (Monte Carlo)")

st.markdown(
    "Running 10,000 simulations over 100 trades to model equity curve behavior..."
)

# Initialize simulator with proper initial capital
initial_capital_for_sim = max(usd_risk_capital * 50, 10000)  # Minimum $10k for meaningful simulation

simulator = MonteCarloSimulator(
    initial_capital_usd=initial_capital_for_sim,
    win_rate=win_rate / 100,
    reward_to_risk=reward_to_risk,
    num_simulations=10000,
    num_trades=100
)

# Run simulation
try:
    simulation_results = simulator.run_simulation()
    st.session_state.simulation_results = simulation_results

    # Extract key statistics
    median_final_balance = np.median(simulation_results["final_balances"])
    percentile_5_final = np.percentile(simulation_results["final_balances"], 5)
    percentile_95_final = np.percentile(simulation_results["final_balances"], 95)
    probability_20_drawdown = simulation_results["probability_20_percent_drawdown"]
    max_drawdown_avg = np.mean(simulation_results["max_drawdowns"])

    # ========================================================================
    # SIMULATION STATISTICS
    # ========================================================================
    col_stat1, col_stat2, col_stat3 = st.columns(3)

    with col_stat1:
        pct_return = ((median_final_balance / initial_capital_for_sim) - 1) * 100
        st.metric(
            label="Median Final Balance",
            value=f"USD {median_final_balance:,.0f}",
            delta=f"+{pct_return:.1f}%"
        )

    with col_stat2:
        st.metric(
            label="Avg Max Drawdown",
            value=f"{max_drawdown_avg:.2f}%",
            delta="Worst-case per simulation"
        )

    with col_stat3:
        risk_color = "🔴" if probability_20_drawdown > 0.30 else "🟡" if probability_20_drawdown > 0.15 else "🟢"
        st.metric(
            label=f"{risk_color} Probability of -20% Drawdown",
            value=f"{probability_20_drawdown * 100:.2f}%",
            delta="Risk of hitting 20% loss"
        )

    # ========================================================================
    # DETAILED SIMULATION TABLE
    # ========================================================================
    st.markdown("**Simulation Summary Statistics:**")
    summary_table = pd.DataFrame({
        "Metric": [
            "Initial Capital",
            "Median Final Balance",
            "5th Percentile",
            "95th Percentile",
            "Average Max Drawdown",
            "Probability of 20% Loss",
            "Win Rate",
            "Reward-to-Risk"
        ],
        "Value": [
            f"USD {initial_capital_for_sim:,.0f}",
            f"USD {median_final_balance:,.0f}",
            f"USD {percentile_5_final:,.0f}",
            f"USD {percentile_95_final:,.0f}",
            f"{max_drawdown_avg:.2f}%",
            f"{probability_20_drawdown * 100:.2f}%",
            f"{win_rate:.1f}%",
            f"{reward_to_risk:.2f}:1"
        ]
    })
    st.dataframe(summary_table, use_container_width=True)

    # ========================================================================
    # EQUITY CURVE VISUALIZATION
    # ========================================================================
    st.markdown("---")
    st.subheader("📉 EQUITY CURVE: Median vs. Drawdown Scenarios")

    # Calculate percentile curves
    median_curve = np.median(simulation_results["equity_curves"], axis=0)
    percentile_5_curve = np.percentile(simulation_results["equity_curves"], 5, axis=0)
    percentile_95_curve = np.percentile(simulation_results["equity_curves"], 95, axis=0)

    trade_numbers = np.arange(0, len(median_curve))

    fig_equity = go.Figure()

    # Add worst-case (5th percentile)
    fig_equity.add_trace(go.Scatter(
        x=trade_numbers,
        y=percentile_5_curve,
        name="5th Percentile (Worst Case)",
        line=dict(color="red", width=2, dash="dash"),
        fill=None
    ))

    # Add best-case (95th percentile)
    fig_equity.add_trace(go.Scatter(
        x=trade_numbers,
        y=percentile_95_curve,
        name="95th Percentile (Best Case)",
        line=dict(color="green", width=2, dash="dash"),
        fill="tonexty",
        fillcolor="rgba(0, 255, 0, 0.1)"
    ))

    # Add median
    fig_equity.add_trace(go.Scatter(
        x=trade_numbers,
        y=median_curve,
        name="Median Equity Curve",
        line=dict(color="blue", width=3),
        fill="tonexty",
        fillcolor="rgba(255, 0, 0, 0.1)"
    ))

    fig_equity.update_layout(
        title="Monte Carlo Simulation: 10,000 Scenarios over 100 Trades",
        xaxis_title="Trade Number",
        yaxis_title="Account Balance (USD)",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig_equity, use_container_width=True)

    # ========================================================================
    # DRAWDOWN DISTRIBUTION
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 DRAWDOWN DISTRIBUTION")

    fig_drawdown = go.Figure(data=[
        go.Histogram(
            x=simulation_results["max_drawdowns"],
            nbinsx=50,
            name="Max Drawdown %",
            marker_color="indianred"
        )
    ])

    fig_drawdown.add_vline(
        x=20.0,
        line_dash="dash",
        line_color="red",
        annotation_text="20% Loss Threshold",
        annotation_position="top right"
    )

    fig_drawdown.update_layout(
        title="Distribution of Maximum Drawdowns Across Simulations",
        xaxis_title="Maximum Drawdown (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig_drawdown, use_container_width=True)

    # ========================================================================
    # RISK RECOMMENDATIONS
    # ========================================================================
    st.markdown("---")
    st.subheader("⚠️ RISK ANALYSIS & RECOMMENDATIONS")

    risk_warnings = []
    risk_suggestions = []

    # Risk warnings and suggestions
    if probability_20_drawdown > 0.30:
        risk_warnings.append(
            "🔴 **HIGH RISK:** Probability of 20% drawdown exceeds 30%. "
            "Consider reducing risk per trade or improving win rate."
        )
    elif probability_20_drawdown > 0.15:
        risk_warnings.append(
            "🟡 **MODERATE RISK:** Probability of 20% drawdown is between 15-30%. "
            "Monitor closely and consider risk reduction strategies."
        )
    else:
        risk_suggestions.append(
            "🟢 **LOW RISK:** Probability of 20% drawdown is below 15%. "
            "Current risk parameters are within acceptable range."
        )

    if ev_per_trade <= 0:
        risk_warnings.append(
            "🔴 **NEGATIVE EXPECTANCY:** Your win rate and R:R do not generate positive EV. "
            "Improve win rate or increase R:R before trading."
        )
    else:
        risk_suggestions.append(
            f"✅ Positive expectancy of USD {ev_per_trade:.4f} per trade detected."
        )

    if kelly_fraction > 0.25:
        risk_suggestions.append(
            f"💡 Full Kelly ({kelly_fraction:.4f}) is aggressive. Use Half-Kelly ({half_kelly:.4f}) "
            "for smoother equity curves."
        )

    if reward_to_risk < 1.5:
        risk_warnings.append(
            "⚠️ R:R ratio below 1.5:1 reduces profit potential. "
            "Target R:R >= 2:1 for optimal risk-adjusted returns."
        )

    if win_rate < 50:
        risk_warnings.append(
            "⚠️ Win rate below 50% requires high R:R to be profitable. "
            "Current setup relies on quality trade management."
        )

    for warning in risk_warnings:
        st.warning(warning)

    for suggestion in risk_suggestions:
        st.success(suggestion)

except Exception as e:
    st.error(f"Error running Monte Carlo simulation: {e}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p><b>Forex Risk Management Dashboard v1.0</b> | Risk Mitigation First | ZAR Base Currency</p>
    <p>Disclaimer: This tool is for educational and risk analysis purposes. Always validate calculations independently.</p>
    </div>
    """,
    unsafe_allow_html=True
)
