import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Forex Backtester Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Forex Order Flow Strategy Backtester")
st.markdown("---")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.header("⚙️ Strategy Parameters")

# TRADING CAPITAL - Changed to ZAR
initial_capital = st.sidebar.number_input(
    "Initial Trading Capital (ZAR)",
    min_value=1000,
    max_value=1000000,
    value=150000,
    step=1000,
    help="Starting account balance for backtesting in South African Rand"
)

max_capital_exposure = st.sidebar.number_input(
    "Maximum Capital Exposure per Trade (R)",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    help="Maximum risk per trade in currency units"
)

min_rrr = st.sidebar.slider(
    "Minimum Risk-to-Reward Ratio",
    min_value=1.0,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="Minimum target yield as multiple of initial risk"
)

volume_surge_multiplier = st.sidebar.slider(
    "Volume Surge Multiplier",
    min_value=1.0,
    max_value=5.0,
    value=2.5,
    step=0.1,
    help="Delta must be X times the 20-period rolling average"
)

# TRADE DIRECTION FILTER
st.sidebar.markdown("---")
st.sidebar.header("📍 Trade Direction")
trade_direction = st.sidebar.radio(
    "Select Trade Type:",
    options=["Both (Long & Short)", "Long Only ↑", "Short Only ↓"],
    help="Choose which direction(s) to trade"
)

# STOP LOSS METHOD
st.sidebar.markdown("---")
st.sidebar.header("🛑 Stop Loss Configuration")
stop_loss_method = st.sidebar.radio(
    "Stop Loss Method:",
    options=["Fixed Percentage", "Average True Range (ATR)"],
    help="Choose how to calculate stop loss"
)

if stop_loss_method == "Fixed Percentage":
    stop_loss_percent = st.sidebar.slider(
        "Stop Loss %",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Percentage below/above entry price for stop loss"
    )
    atr_period = 14  # Default, won't be used
    atr_multiplier = 1.0  # Default, won't be used
else:
    atr_period = st.sidebar.slider(
        "ATR Period",
        min_value=5,
        max_value=50,
        value=14,
        step=1,
        help="Number of candles for ATR calculation"
    )
    atr_multiplier = st.sidebar.slider(
        "ATR Multiplier",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Multiply ATR by this value for stop loss distance"
    )
    stop_loss_percent = 2.0  # Default, won't be used

# CURRENCY PAIR SELECTION - Added XAU/USD
st.sidebar.markdown("---")
st.sidebar.header("💱 Currency Pair")
major_pairs = {
    "EUR/USD": {"symbol": "EURUSD", "description": "Euro vs US Dollar"},
    "GBP/USD": {"symbol": "GBPUSD", "description": "British Pound vs US Dollar"},
    "USD/JPY": {"symbol": "USDJPY", "description": "US Dollar vs Japanese Yen"},
    "USD/CHF": {"symbol": "USDCHF", "description": "US Dollar vs Swiss Franc"},
    "AUD/USD": {"symbol": "AUDUSD", "description": "Australian Dollar vs US Dollar"},
    "USD/CAD": {"symbol": "USDCAD", "description": "US Dollar vs Canadian Dollar"},
    "NZD/USD": {"symbol": "NZDUSD", "description": "New Zealand Dollar vs US Dollar"},
    "XAU/USD": {"symbol": "XAUUSD", "description": "Gold vs US Dollar"},
}
selected_pair = st.sidebar.selectbox(
    "Select Major Currency Pair:",
    options=list(major_pairs.keys()),
    help="Choose which pair to analyze"
)

pair_info = major_pairs[selected_pair]

st.sidebar.markdown("---")
st.sidebar.header("📊 Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload OHLCV CSV File",
    type=['csv'],
    help="CSV should contain: datetime, open, high, low, close, volume, delta, absorption"
)

# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

@st.cache_data
def load_sample_data():
    """Generate synthetic OHLCV data for demonstration"""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
    
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.5) + 1.3500
    
    data = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(1000) * 0.0003,
        'high': prices + np.abs(np.random.randn(1000)) * 0.001,
        'low': prices - np.abs(np.random.randn(1000)) * 0.001,
        'close': prices,
        'volume': np.random.randint(10000, 100000, 1000),
        'delta': np.random.randint(-5000, 5000, 1000),
        'absorption': np.random.randint(1000, 20000, 1000)
    })
    
    data['close'] = prices
    return data

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    df = df.copy()
    
    # Calculate True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR
    df['atr'] = df['tr'].rolling(window=period, min_periods=1).mean()
    
    return df['atr']

def process_data(df, stop_loss_method, stop_loss_percent=2.0, atr_period=14, atr_multiplier=1.5):
    """Process and validate OHLCV data"""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate delta 20-period rolling average
    df['delta_ma20'] = df['delta'].rolling(window=20, min_periods=1).mean()
    
    # Calculate volume surge signal
    df['volume_signal'] = df['delta'].abs() >= (df['delta_ma20'].abs() * volume_surge_multiplier)
    
    # Calculate ATR if needed
    if stop_loss_method == "Average True Range (ATR)":
        df['atr'] = calculate_atr(df, period=atr_period)
    
    # Initialize trade tracking columns
    df['trade_signal'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['pnl'] = 0.0
    
    return df

# Load data
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        data = process_data(data, stop_loss_method, stop_loss_percent, atr_period, atr_multiplier)
        st.sidebar.success("✅ Data loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {str(e)}")
        st.warning("Using sample data for demonstration...")
        data = load_sample_data()
        data = process_data(data, stop_loss_method, stop_loss_percent, atr_period, atr_multiplier)
else:
    st.info("📂 Upload a CSV file or using sample data for demo...")
    data = load_sample_data()
    data = process_data(data, stop_loss_method, stop_loss_percent, atr_period, atr_multiplier)

# ============================================================================
# TRADING LOGIC & BACKTESTING ENGINE
# ============================================================================

class ForexOrderFlowStrategy:
    """
    Forex Order Flow Strategy implementing:
    - Dynamic position sizing with capital exposure limits
    - Risk-to-reward validation
    - Session lockout (Asian open)
    - 6-candle invalidation rule
    - Volume normalization
    - Break-even risk elimination
    - Trade direction filtering (Long/Short/Both)
    - ATR-based or fixed percentage stop loss
    """
    
    def __init__(self, data, initial_capital, max_exposure, min_rrr, volume_multiplier, 
                 trade_direction="Both (Long & Short)", stop_loss_method="Fixed Percentage",
                 stop_loss_percent=2.0, atr_period=14, atr_multiplier=1.5):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.max_exposure = max_exposure
        self.min_rrr = min_rrr
        self.volume_multiplier = volume_multiplier
        self.trade_direction = trade_direction
        self.stop_loss_method = stop_loss_method
        self.stop_loss_percent = stop_loss_percent
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        # Trading parameters
        self.pip_value = 0.0001  # For standard pairs
        self.standard_lot_size = 100000
        self.pip_value_per_lot = 10  # $10 per pip per standard lot
        
        self.trades = []
        self.equity = initial_capital
        self.equity_curve = [self.equity]
        self.position = None
        self.current_pnl = 0
        
    def is_session_lockout(self, timestamp):
        """Check if current time is during Asian open lockout (Sunday-Monday, first 4 hours)"""
        day = timestamp.dayofweek
        hour = timestamp.hour
        
        # Sunday (6) 20:00-23:59 + Monday (0) 00:00-03:59
        is_sunday_evening = (day == 6 and hour >= 20)
        is_monday_morning = (day == 0 and hour < 4)
        
        return is_sunday_evening or is_monday_morning
    
    def calculate_lot_size(self, entry, stop_loss):
        """
        Dynamic position sizing: Lot Size = R100 / (Distance in Pips × Pip Value)
        """
        distance_pips = abs(entry - stop_loss) / self.pip_value
        
        if distance_pips == 0:
            return 0
        
        # Calculate lot size needed for exactly R100 exposure
        lot_size = self.max_exposure / (distance_pips * (self.pip_value_per_lot / self.standard_lot_size))
        
        # Minimum viable lot size
        min_lot = 0.01
        if lot_size < min_lot:
            return 0  # Abort: too risky
        
        return min(lot_size, 10)  # Cap at 10 standard lots
    
    def validate_trade(self, entry, stop_loss, take_profit, direction):
        """
        Validate trade setup:
        1. Check trade direction filter
        2. Position sizing within exposure limits
        3. Risk-to-reward ratio meets minimum threshold
        """
        # Check direction filter
        if self.trade_direction == "Long Only ↑" and direction == -1:
            return False, "Short trades disabled - Long Only mode"
        if self.trade_direction == "Short Only ↓" and direction == 1:
            return False, "Long trades disabled - Short Only mode"
        
        # Calculate lot size
        lot_size = self.calculate_lot_size(entry, stop_loss)
        if lot_size == 0:
            return False, "Position size exceeds exposure limit"
        
        # Calculate risk and reward
        risk_distance = abs(entry - stop_loss)
        reward_distance = abs(take_profit - entry)
        
        if risk_distance == 0:
            return False, "Invalid stop loss"
        
        rrr = reward_distance / risk_distance
        if rrr < self.min_rrr:
            return False, f"RRR {rrr:.2f} < Minimum {self.min_rrr}"
        
        return True, lot_size
    
    def calculate_stop_loss(self, entry, direction, idx):
        """Calculate stop loss based on selected method"""
        if self.stop_loss_method == "Average True Range (ATR)":
            if 'atr' in self.data.columns and not pd.isna(self.data.loc[idx, 'atr']):
                atr_value = self.data.loc[idx, 'atr']
                if direction == 1:  # Long
                    stop_loss = entry - (atr_value * self.atr_multiplier)
                else:  # Short
                    stop_loss = entry + (atr_value * self.atr_multiplier)
                return stop_loss
        
        # Default to fixed percentage
        if direction == 1:  # Long
            stop_loss = entry * (1 - self.stop_loss_percent / 100)
        else:  # Short
            stop_loss = entry * (1 + self.stop_loss_percent / 100)
        
        return stop_loss
    
    def backtest(self):
        """Execute backtesting logic"""
        candle_counter = 0
        
        for i in range(len(self.data)):
            candle = self.data.iloc[i]
            timestamp = candle['datetime']
            
            # Check session lockout
            if self.is_session_lockout(timestamp):
                if self.position:
                    self._close_position(i, 'Session Lockout')
                continue
            
            # Check for 6-candle invalidation
            if self.position:
                candle_counter += 1
                if candle_counter >= 6:
                    self._close_position(i, '6-Candle Invalidation')
                    candle_counter = 0
            
            # Volume surge signal
            if self.data.loc[i, 'volume_signal'] and not self.position:
                # Simple entry signal: volume surge + momentum
                if i > 20:
                    entry_signal = self._generate_entry_signal(i)
                    if entry_signal:
                        self._open_position(i, entry_signal)
                        candle_counter = 0
            
            # Manage open position
            if self.position:
                self._manage_position(i)
        
        # Close any remaining position
        if self.position:
            self._close_position(len(self.data) - 1, 'End of Data')
        
        self.data['equity'] = np.cumsum(self.data['pnl']) + self.initial_capital
        return self._generate_metrics()
    
    def _generate_entry_signal(self, idx):
        """Generate entry signal based on order flow"""
        candle = self.data.iloc[idx]
        
        # Signal: High volume + delta surge
        if candle['volume_signal']:
            direction = 1 if candle['delta'] > 0 else -1  # Long if positive delta
            return {
                'entry': candle['close'],
                'direction': direction,
                'timestamp': candle['datetime'],
                'index': idx
            }
        return None
    
    def _open_position(self, idx, signal):
        """Open a new position"""
        candle = self.data.iloc[idx]
        entry = signal['entry']
        direction = signal['direction']
        
        # Calculate stop loss using selected method
        stop_loss = self.calculate_stop_loss(entry, direction, idx)
        take_profit = entry + (direction * abs(entry - stop_loss) * self.min_rrr)
        
        # Validate trade
        valid, lot_size = self.validate_trade(entry, stop_loss, take_profit, direction)
        if not valid:
            return
        
        self.position = {
            'entry_idx': idx,
            'entry_price': entry,
            'entry_time': signal['timestamp'],
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'max_profit': 0,
            'entry_risk': abs(entry - stop_loss)
        }
        
        self.data.loc[idx, 'trade_signal'] = direction
        self.data.loc[idx, 'entry_price'] = entry
    
    def _manage_position(self, idx):
        """Manage open position: check SL, TP, break-even"""
        if not self.position:
            return
        
        candle = self.data.iloc[idx]
        direction = self.position['direction']
        entry = self.position['entry_price']
        initial_risk = self.position['entry_risk']
        
        # Check if price hits stop loss
        if direction == 1:  # Long
            if candle['low'] <= self.position['stop_loss']:
                self._close_position(idx, 'Stop Loss')
                return
            
            current_profit = candle['close'] - entry
            
            # Move stop to break-even if 1R profit achieved
            if current_profit >= initial_risk and self.position['stop_loss'] < entry:
                self.position['stop_loss'] = entry
            
            # Check take profit
            if candle['high'] >= self.position['take_profit']:
                self._close_position(idx, 'Take Profit')
                return
        
        else:  # Short
            if candle['high'] >= self.position['stop_loss']:
                self._close_position(idx, 'Stop Loss')
                return
            
            current_profit = entry - candle['close']
            
            # Move stop to break-even if 1R profit achieved
            if current_profit >= initial_risk and self.position['stop_loss'] > entry:
                self.position['stop_loss'] = entry
            
            # Check take profit
            if candle['low'] <= self.position['take_profit']:
                self._close_position(idx, 'Take Profit')
                return
    
    def _close_position(self, idx, reason):
        """Close open position and record P&L"""
        if not self.position:
            return
        
        candle = self.data.iloc[idx]
        exit_price = candle['close']
        
        direction = self.position['direction']
        entry = self.position['entry_price']
        
        # Calculate P&L
        if direction == 1:
            pnl = (exit_price - entry) * self.position['lot_size'] * self.standard_lot_size * self.pip_value / self.pip_value
        else:
            pnl = (entry - exit_price) * self.position['lot_size'] * self.standard_lot_size * self.pip_value / self.pip_value
        
        # Record trade
        self.trades.append({
            'entry_time': self.position['entry_time'],
            'exit_time': candle['datetime'],
            'entry_price': entry,
            'exit_price': exit_price,
            'direction': 'Long' if direction == 1 else 'Short',
            'lot_size': self.position['lot_size'],
            'pnl': pnl,
            'reason': reason,
            'risk': self.position['entry_risk'],
            'reward': abs(exit_price - entry)
        })
        
        self.data.loc[idx, 'exit_price'] = exit_price
        self.data.loc[idx, 'pnl'] = pnl
        
        self.equity += pnl
        self.equity_curve.append(self.equity)
        
        self.position = None
    
    def _generate_metrics(self):
        """Generate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'net_profit': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'long_trades': 0,
                'short_trades': 0,
                'trades_df': pd.DataFrame()
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        long_trades = len(trades_df[trades_df['direction'] == 'Long'])
        short_trades = len(trades_df[trades_df['direction'] == 'Short'])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        net_profit = trades_df['pnl'].sum()
        
        # Calculate max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'win_rate': win_rate,
            'net_profit': net_profit,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades_df': trades_df,
            'equity_curve': self.equity_curve
        }

# ============================================================================
# STRATEGY ANALYST - NEW FEATURE
# ============================================================================

class StrategyAnalyst:
    """AI-powered analyst to identify strategy pitfalls and improvements"""
    
    def __init__(self, metrics, strategy_params):
        self.metrics = metrics
        self.strategy_params = strategy_params
        self.pitfalls = []
        self.recommendations = []
    
    def analyze(self):
        """Run comprehensive strategy analysis"""
        if self.metrics['total_trades'] == 0:
            return self._no_trades_analysis()
        
        self._check_win_rate()
        self._check_profit_factor()
        self._check_drawdown()
        self._check_trade_balance()
        self._check_rrr_efficiency()
        self._check_average_trades()
        self._check_position_sizing()
        
        return {
            'pitfalls': self.pitfalls,
            'recommendations': self.recommendations,
            'risk_level': self._assess_risk_level()
        }
    
    def _no_trades_analysis(self):
        """Handle case when no trades were generated"""
        return {
            'pitfalls': [
                "⚠️ No trades were generated during backtest",
                "Strategy parameters may be too restrictive"
            ],
            'recommendations': [
                "📝 Reduce minimum Risk-to-Reward Ratio",
                "📝 Lower Volume Surge Multiplier threshold",
                "📝 Increase Maximum Capital Exposure per trade",
                "📝 Verify uploaded data contains valid signals"
            ],
            'risk_level': 'N/A'
        }
    
    def _check_win_rate(self):
        """Analyze win rate"""
        wr = self.metrics['win_rate']
        
        if wr < 30:
            self.pitfalls.append(f"❌ Win Rate critically low at {wr:.1f}% (Below 30%)")
            self.recommendations.append("🎯 Improve entry signal quality or strengthen filter conditions")
        elif wr < 40:
            self.pitfalls.append(f"⚠️ Win Rate below ideal at {wr:.1f}% (30-40%)")
            self.recommendations.append("📈 Refine volume signal thresholds or add confirmation indicators")
        elif wr > 70:
            self.pitfalls.append(f"⚠️ Win Rate suspiciously high at {wr:.1f}% - possible overfitting")
            self.recommendations.append("🔍 Backtest on out-of-sample data or recent time periods")
    
    def _check_profit_factor(self):
        """Analyze profit factor"""
        pf = self.metrics['profit_factor']
        
        if pf < 1.0:
            self.pitfalls.append(f"❌ Profit Factor {pf:.2f} - Strategy is unprofitable")
            self.recommendations.append("🛑 Increase stop loss distance or improve entry timing")
        elif pf < 1.5:
            self.pitfalls.append(f"⚠️ Profit Factor {pf:.2f} - Marginal profitability")
            self.recommendations.append("💰 Increase minimum RRR or tighten trade selection filters")
        elif pf > 3.0:
            self.pitfalls.append(f"⚠️ Profit Factor {pf:.2f} - May indicate overfitting")
            self.recommendations.append("📊 Validate on different market conditions or time periods")
    
    def _check_drawdown(self):
        """Analyze maximum drawdown"""
        dd = abs(self.metrics['max_drawdown'])
        
        if dd > 30:
            self.pitfalls.append(f"❌ Max Drawdown {dd:.2f}% - Excessive risk exposure")
            self.recommendations.append("🛡️ Reduce position size or implement tighter stop losses")
        elif dd > 20:
            self.pitfalls.append(f"⚠️ Max Drawdown {dd:.2f}% - High drawdown risk")
            self.recommendations.append("💾 Consider reducing per-trade exposure or risk management")
        elif dd < 5:
            self.recommendations.append("✅ Excellent risk management with low drawdown")
    
    def _check_trade_balance(self):
        """Analyze long vs short trade balance"""
        total = self.metrics['total_trades']
        long_ratio = self.metrics['long_trades'] / total if total > 0 else 0
        
        if long_ratio > 0.85 or long_ratio < 0.15:
            self.pitfalls.append(f"⚠️ Imbalanced trade direction: {long_ratio*100:.0f}% Long / {(1-long_ratio)*100:.0f}% Short")
            self.recommendations.append("⚖️ Strategy may be biased to one direction - check for market regime dependency")
    
    def _check_rrr_efficiency(self):
        """Analyze Risk-to-Reward efficiency"""
        trades_df = self.metrics['trades_df']
        if len(trades_df) == 0:
            return
        
        actual_rrr = trades_df['reward'].mean() / trades_df['risk'].mean() if trades_df['risk'].mean() > 0 else 0
        min_rrr = self.strategy_params.get('min_rrr', 1.5)
        
        if actual_rrr < min_rrr * 0.8:
            self.pitfalls.append(f"⚠️ Actual RRR {actual_rrr:.2f} underperforming minimum {min_rrr:.2f}")
            self.recommendations.append("📍 Adjust take profit targets to achieve better reward/risk ratio")
    
    def _check_average_trades(self):
        """Analyze trade frequency"""
        total = self.metrics['total_trades']
        
        if total < 5:
            self.pitfalls.append(f"⚠️ Only {total} trades generated - insufficient data")
            self.recommendations.append("📈 Consider relaxing entry filters for more trading opportunities")
        elif total > 100:
            self.recommendations.append("✅ Adequate trade sample size for statistical validity")
    
    def _check_position_sizing(self):
        """Analyze position sizing quality"""
        trades_df = self.metrics['trades_df']
        if len(trades_df) > 0 and 'lot_size' in trades_df.columns:
            avg_lot = trades_df['lot_size'].mean()
            if avg_lot < 0.1:
                self.pitfalls.append(f"⚠️ Average position size very small ({avg_lot:.3f} lots)")
                self.recommendations.append("💪 Consider increasing maximum exposure per trade")
    
    def _assess_risk_level(self):
        """Assess overall risk level"""
        score = 0
        
        # Win rate (0-25 points)
        wr = self.metrics['win_rate']
        score += min(25, wr / 2)
        
        # Profit factor (0-25 points)
        pf = self.metrics['profit_factor']
        score += min(25, pf * 8)
        
        # Drawdown (0-25 points) - lower is better
        dd = abs(self.metrics['max_drawdown'])
        score += max(0, 25 - (dd * 0.8))
        
        # Trade count (0-25 points)
        trades = self.metrics['total_trades']
        score += min(25, trades / 4)
        
        if score < 40:
            return "🔴 HIGH RISK - Needs significant improvement"
        elif score < 60:
            return "🟡 MEDIUM RISK - Has potential but needs refinement"
        else:
            return "🟢 LOW RISK - Strategy shows promise"

# ============================================================================
# RUN BACKTEST
# ============================================================================

st.subheader("📊 Running Backtest...")

with st.spinner("Analyzing order flow and executing strategy..."):
    strategy = ForexOrderFlowStrategy(
        data,
        initial_capital,
        max_capital_exposure,
        min_rrr,
        volume_surge_multiplier,
        trade_direction=trade_direction,
        stop_loss_method=stop_loss_method,
        stop_loss_percent=stop_loss_percent,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier
    )
    metrics = strategy.backtest()
    
    # Run Strategy Analysis
    analyst = StrategyAnalyst(metrics, {
        'min_rrr': min_rrr,
        'volume_multiplier': volume_surge_multiplier,
        'stop_loss_method': stop_loss_method
    })
    analysis = analyst.analyze()

# ============================================================================
# DISPLAY RESULTS - HEADER INFO
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"💱 **Pair:** {selected_pair}\n\n{pair_info['description']}")

with col2:
    st.info(f"🧮 **Account (ZAR):**\n\nStarting: R{initial_capital:,.0f}\n\nFinal: R{metrics['equity_curve'][-1]:,.2f}" if metrics['equity_curve'] else f"🧮 **Account:**\n\nStarting: R{initial_capital:,.0f}")

with col3:
    mode_text = "🔄 Both" if trade_direction == "Both (Long & Short)" else trade_direction
    st.info(f"🎯 **Mode:** {mode_text}\n\n📍 **SL:** {stop_loss_method}")

st.markdown("---")

# ============================================================================
# STRATEGY ANALYST REPORT - NEW SECTION
# ============================================================================

st.subheader("🔍 Strategy Analyst Report")

analyst_col1, analyst_col2 = st.columns([1, 1])

with analyst_col1:
    st.markdown("### 🚨 Identified Pitfalls")
    if analysis['pitfalls']:
        for pitfall in analysis['pitfalls']:
            st.markdown(f"- {pitfall}")
    else:
        st.markdown("✅ No critical pitfalls detected!")

with analyst_col2:
    st.markdown("### 💡 Recommendations")
    if analysis['recommendations']:
        for i, rec in enumerate(analysis['recommendations'], 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.markdown("✅ Strategy parameters look good!")

st.markdown(f"### 📊 Risk Assessment: {analysis['risk_level']}")
st.markdown("---")

# ============================================================================
# KEY METRICS
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Net Profit",
        f"R{metrics['net_profit']:.2f}",
        delta=f"{(metrics['net_profit']/initial_capital*100):.1f}%" if metrics['net_profit'] != 0 else "0%"
    )

with col2:
    st.metric(
        "Win Rate",
        f"{metrics['win_rate']:.1f}%",
        delta=f"{metrics['winning_trades']}W / {metrics['losing_trades']}L"
    )

with col3:
    st.metric(
        "Max Drawdown",
        f"{metrics['max_drawdown']:.2f}%",
        delta="Negative" if metrics['max_drawdown'] < 0 else "Neutral"
    )

with col4:
    st.metric(
        "Profit Factor",
        f"{metrics['profit_factor']:.2f}x",
        delta=f"{metrics['total_trades']} trades"
    )

st.markdown("---")

# ============================================================================
# EQUITY CURVE CHART
# ============================================================================

st.subheader("📈 Equity Curve")

if metrics['equity_curve']:
    equity_df = pd.DataFrame({
        'Trade': range(len(metrics['equity_curve'])),
        'Equity': metrics['equity_curve']
    })
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df['Trade'],
        y=equity_df['Equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#00CC96', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.1)'
    ))
    
    # Add starting capital line
    fig_equity.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Capital"
    )
    
    fig_equity.update_layout(
        title="Account Equity Over Time (ZAR)",
        xaxis_title="Trade #",
        yaxis_title="Equity (ZAR)",
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
else:
    st.warning("No trades generated. Adjust parameters or upload different data.")

# ============================================================================
# PRICE ACTION & TRADES CHART
# ============================================================================

st.subheader("🕯️ Price Action with Trade Entries/Exits")

# Display recent 100 candles with trades overlaid
display_data = data.iloc[-100:].copy()

fig_price = go.Figure()

# Add candlestick chart
fig_price.add_trace(go.Candlestick(
    x=display_data['datetime'],
    open=display_data['open'],
    high=display_data['high'],
    low=display_data['low'],
    close=display_data['close'],
    name='Price',
    increasing_line_color='#00CC96',
    decreasing_line_color='#FF6B6B'
))

# Add entry points - safely handle column access
if 'entry_price' in display_data.columns:
    entries = display_data[display_data['entry_price'].notna()]
    if not entries.empty:
        fig_price.add_trace(go.Scatter(
            x=entries['datetime'],
            y=entries['entry_price'],
            mode='markers',
            name='Entry',
            marker=dict(size=10, color='#00D9FF', symbol='triangle-up')
        ))

# Add exit points - safely handle column access
if 'exit_price' in display_data.columns:
    exits = display_data[display_data['exit_price'].notna()]
    if not exits.empty:
        fig_price.add_trace(go.Scatter(
            x=exits['datetime'],
            y=exits['exit_price'],
            mode='markers',
            name='Exit',
            marker=dict(size=10, color='#FF6B6B', symbol='triangle-down')
        ))

fig_price.update_layout(
    title="Recent Price Action with Trade Signals (Last 100 Candles)",
    yaxis_title="Price",
    xaxis_title="Time",
    template='plotly_dark',
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig_price, use_container_width=True)

# ============================================================================
# VOLUME & DELTA ANALYSIS
# ============================================================================

st.subheader("📊 Volume & Delta Analysis")

col1, col2 = st.columns(2)

with col1:
    display_vol = data.iloc[-100:].copy()
    
    fig_volume = go.Figure()
    
    colors = ['#00CC96' if row['close'] >= row['open'] else '#FF6B6B' 
              for _, row in display_vol.iterrows()]
    
    fig_volume.add_trace(go.Bar(
        x=display_vol['datetime'],
        y=display_vol['volume'],
        name='Volume',
        marker_color=colors,
        marker_line_width=0
    ))
    
    fig_volume.update_layout(
        title="Trading Volume",
        yaxis_title="Volume",
        xaxis_title="Time",
        template='plotly_dark',
        height=350,
        hovermode='x unified',
        showlegend=False
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)

with col2:
    fig_delta = go.Figure()
    
    fig_delta.add_trace(go.Bar(
        x=display_vol['datetime'],
        y=display_vol['delta'],
        name='Delta',
        marker=dict(
            color=display_vol['delta'],
            colorscale='RdYlGn',
            showscale=False
        )
    ))
    
    # Add threshold line
    fig_delta.add_hline(
        y=display_vol['delta_ma20'].iloc[-1] * volume_surge_multiplier,
        line_dash="dash",
        line_color="cyan",
        annotation_text=f"Threshold (MA20 × {volume_surge_multiplier})"
    )
    
    fig_delta.update_layout(
        title="Order Flow Delta",
        yaxis_title="Delta",
        xaxis_title="Time",
        template='plotly_dark',
        height=350,
        hovermode='x unified',
        showlegend=False
    )
    
    st.plotly_chart(fig_delta, use_container_width=True)

# ============================================================================
# TRADE DIRECTION BREAKDOWN
# ============================================================================

if metrics['total_trades'] > 0:
    st.subheader("📊 Trade Direction Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Long Trades", metrics['long_trades'])
    
    with col2:
        st.metric("Short Trades", metrics['short_trades'])
    
    with col3:
        if metrics['long_trades'] > 0:
            long_wins = len(metrics['trades_df'][(metrics['trades_df']['direction'] == 'Long') & (metrics['trades_df']['pnl'] > 0)])
            long_win_rate = (long_wins / metrics['long_trades'] * 100)
            st.metric("Long Win Rate", f"{long_win_rate:.1f}%")
        else:
            st.metric("Long Win Rate", "N/A")

# ============================================================================
# TRADE DETAILS TABLE
# ============================================================================

st.subheader("📋 Trade Details")

if not metrics['trades_df'].empty:
    trades_display = metrics['trades_df'].copy()
    trades_display['entry_time'] = trades_display['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    trades_display['exit_time'] = trades_display['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    trades_display['entry_price'] = trades_display['entry_price'].round(5)
    trades_display['exit_price'] = trades_display['exit_price'].round(5)
    trades_display['pnl'] = trades_display['pnl'].round(2)
    trades_display['risk'] = trades_display['risk'].round(5)
    trades_display['reward'] = trades_display['reward'].round(5)
    
    # Reorder columns
    trades_display = trades_display[[
        'entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price',
        'lot_size', 'risk', 'reward', 'pnl', 'reason'
    ]]
    
    st.dataframe(
        trades_display,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No trades were executed. Adjust parameters or provide different data.")

# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================

st.subheader("📊 Performance Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric("Total Trades", metrics['total_trades'])
    st.metric("Winning Trades", metrics['winning_trades'])
    st.metric("Losing Trades", metrics['losing_trades'])

with summary_col2:
    st.metric("Average Win", f"R{metrics['avg_win']:.2f}")
    st.metric("Average Loss", f"R{metrics['avg_loss']:.2f}")
    st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}x")

with summary_col3:
    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
    st.metric("Return on Account", f"{(metrics['net_profit']/initial_capital*100):.2f}%")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📌 Strategy Rules Summary
- **Trading Capital**: R{:,.0f} (ZAR)
- **Dynamic Position Sizing**: R{}/trade with lot calculation based on risk
- **Risk-to-Reward Validation**: Minimum {}:1 ratio enforced
- **Volume Normalization**: Delta surge at {}× rolling 20-period average
- **Session Lockout**: First 4 hours of Sunday/Monday Asian open blocked
- **6-Candle Invalidation**: Positions auto-close if unprofitable within 30 minutes
- **Break-Even Risk Elimination**: Stop moved to entry after 1R profit achieved
- **Trade Direction**: {}
- **Stop Loss Method**: {}
- **Currency Pair**: {}

### ⚠️ Disclaimer
This backtester is for educational purposes only. Past performance does not guarantee future results.
Always conduct your own analysis before live trading.
""".format(initial_capital, max_capital_exposure, min_rrr, volume_surge_multiplier, 
          trade_direction, stop_loss_method, selected_pair))
