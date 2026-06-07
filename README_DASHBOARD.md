# 📊 Forex Risk Management & Position Sizing Dashboard

A robust, production-ready web application for managing forex trading risk with a focus on ZAR-based accounts. Built with Python, Streamlit, Plotly, and Pandas.

## 🎯 Purpose

**Risk mitigation is the core priority** of this application. It provides:
- Dynamic position sizing based on risk capital and stop loss distance
- Expected value calculations using historical win rate and R:R ratio
- Kelly Criterion optimization for bet sizing
- Monte Carlo simulation (10,000 scenarios) to model drawdown probability
- Real-time risk analysis and recommendations

## ✨ Key Features

### 1. **Control Panel (Sidebar)**
- Account Balance (ZAR)
- Risk per Trade (%)
- USD/ZAR Exchange Rate
- Asset Selection (EUR/USD, GBP/USD, USD/JPY, XAU/USD, S&P 500)
- Stop Loss (Pips/Ticks)
- Historical Win Rate (%)
- Reward-to-Risk Ratio (R:R)

### 2. **Sizing Console (Main Top)**
Large KPI metrics displaying:
- Max ZAR Risk per Trade
- Max USD Risk per Trade
- Recommended Lot Size
- Expected Value per Trade
- Full Kelly & Half-Kelly Fractions
- Kelly-Adjusted Lot Size

### 3. **Monte Carlo Simulation (Main Bottom)**
- 10,000 simulations over 100 trades
- Median equity curve with 5th/95th percentile bounds
- Drawdown distribution histogram
- Probability of -20% account loss
- Comprehensive statistics table

### 4. **Risk Analysis & Recommendations**
Automated warnings and suggestions based on:
- Probability of 20% drawdown
- Expected value sign (positive/negative)
- Kelly fraction aggressiveness
- R:R ratio adequacy
- Win rate viability

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone/Download Repository**
```bash
git clone https://github.com/Mossy1982/FXstrategyTest.git
cd FXstrategyTest
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Dashboard**
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
forex-risk-dashboard/
├── app.py                 # Main Streamlit dashboard
├── risk_engine.py         # Risk calculation engine (ZAR/USD conversion, EV, Kelly)
├── position_sizing.py     # Position sizing engine (asset specs & lot calculation)
├── monte_carlo.py         # Monte Carlo simulator (10,000 scenarios)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎲 Kelly Criterion Recommendations

- **Full Kelly:** Aggressive, high ruin risk (not recommended for live trading)
- **Half-Kelly:** Conservative, smooth equity curves (recommended)
- **Quarter-Kelly:** Ultra-conservative, minimal drawdown (safest)

## 💡 Usage Example

**Scenario:** ZAR 50,000 account, 2% risk per trade, EUR/USD, 50-pip stop loss

**Results:**
- Lot Size: ~0.108 standard lots (10,800 units)
- Expected Value: +$0.55 per trade ✅ (Positive)
- Half-Kelly Fraction: ~0.15 (Conservative sizing)
- Probability of -20% Drawdown: ~12% (Low Risk) ✅

---

**Built with ❤️ by a Senior Quantitative Developer**  
*Risk Mitigation First | ZAR Base Currency | Dynamic Parameter Adjustment*
