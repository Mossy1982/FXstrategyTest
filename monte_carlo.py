"""
Monte Carlo Simulation Engine
Generates 10,000 equity curve scenarios to model drawdown risk
"""

import numpy as np


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading equity curves.
    
    Runs multiple simulations to model:
    - Distribution of final balances
    - Maximum drawdown per scenario
    - Probability of hitting specific loss thresholds
    - Equity curve percentiles (median, best case, worst case)
    """
    
    def __init__(self, initial_capital_usd, win_rate, reward_to_risk, 
                 num_simulations=10000, num_trades=100):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            initial_capital_usd (float): Starting capital in USD
            win_rate (float): Historical win rate as decimal (e.g., 0.55)
            reward_to_risk (float): Average R:R ratio (e.g., 2.0 for 2:1)
            num_simulations (int): Number of simulation scenarios (default 10,000)
            num_trades (int): Number of trades per simulation (default 100)
        """
        self.initial_capital_usd = initial_capital_usd
        self.win_rate = win_rate
        self.reward_to_risk = reward_to_risk
        self.num_simulations = num_simulations
        self.num_trades = num_trades
        
        # Pre-calculate trade outcomes
        self.avg_win = reward_to_risk  # Normalized: win = R:R
        self.avg_loss = -1.0  # Normalized: loss = -1
    
    def run_simulation(self):
        """
        Execute Monte Carlo simulation.
        
        Returns:
            dict: Simulation results containing:
                - equity_curves: 2D array of equity paths (num_simulations × num_trades)
                - final_balances: Array of final account values
                - max_drawdowns: Array of max drawdown percentages per simulation
                - probability_20_percent_drawdown: Probability of hitting -20% loss
        """
        # Initialize arrays to store results
        equity_curves = np.zeros((self.num_simulations, self.num_trades + 1))
        final_balances = np.zeros(self.num_simulations)
        max_drawdowns = np.zeros(self.num_simulations)
        
        # Set initial capital for all simulations
        equity_curves[:, 0] = self.initial_capital_usd
        
        # Run simulations
        for sim in range(self.num_simulations):
            balance = self.initial_capital_usd
            peak_balance = balance
            max_dd = 0.0
            
            for trade in range(self.num_trades):
                # Generate random outcome: win or loss
                is_win = np.random.random() < self.win_rate
                
                # Calculate P&L as percentage of current balance
                # Normalized: each trade risked 1% of capital
                risk_per_trade = balance * 0.01  # 1% risk
                
                if is_win:
                    pnl = risk_per_trade * self.avg_win
                else:
                    pnl = risk_per_trade * self.avg_loss
                
                # Update balance
                balance += pnl
                
                # Track peak for drawdown calculation
                if balance > peak_balance:
                    peak_balance = balance
                
                # Calculate current drawdown
                if peak_balance > 0:
                    current_dd = (peak_balance - balance) / peak_balance
                    if current_dd > max_dd:
                        max_dd = current_dd
                
                # Store equity value
                equity_curves[sim, trade + 1] = balance
            
            # Store final results for this simulation
            final_balances[sim] = balance
            max_drawdowns[sim] = max_dd * 100  # Convert to percentage
        
        # Calculate probability of hitting 20% drawdown
        drawdowns_exceeding_20 = np.sum(max_drawdowns >= 20.0)
        probability_20_dd = drawdowns_exceeding_20 / self.num_simulations
        
        return {
            "equity_curves": equity_curves,
            "final_balances": final_balances,
            "max_drawdowns": max_drawdowns,
            "probability_20_percent_drawdown": probability_20_dd
        }
    
    def calculate_statistics(self, equity_curves, final_balances, max_drawdowns):
        """
        Calculate detailed statistics from simulation results.
        
        Args:
            equity_curves (ndarray): Equity paths
            final_balances (ndarray): Final balance distribution
            max_drawdowns (ndarray): Maximum drawdown distribution
        
        Returns:
            dict: Comprehensive statistics
        """
        return {
            "initial_capital": self.initial_capital_usd,
            "median_final_balance": np.median(final_balances),
            "mean_final_balance": np.mean(final_balances),
            "percentile_5_final": np.percentile(final_balances, 5),
            "percentile_95_final": np.percentile(final_balances, 95),
            "std_dev_final": np.std(final_balances),
            "max_final_balance": np.max(final_balances),
            "min_final_balance": np.min(final_balances),
            "median_max_drawdown": np.median(max_drawdowns),
            "mean_max_drawdown": np.mean(max_drawdowns),
            "max_worst_dd": np.max(max_drawdowns),
            "min_worst_dd": np.min(max_drawdowns)
        }
