"""
Risk Calculation Engine
Handles all core risk mathematics for the portfolio
"""

import numpy as np


class RiskCalculationEngine:
    """
    Core risk calculation engine for ZAR-based trading accounts.
    
    Responsible for:
    - ZAR risk capital calculation
    - USD conversion
    - Expected value computation
    - Kelly Criterion calculation
    """
    
    def __init__(self, account_balance_zar, risk_percentage, usd_zar_rate):
        """
        Initialize the risk engine.
        
        Args:
            account_balance_zar (float): Account balance in ZAR
            risk_percentage (float): Risk per trade as percentage (e.g., 2.0 for 2%)
            usd_zar_rate (float): Current USD/ZAR exchange rate
        """
        self.account_balance_zar = account_balance_zar
        self.risk_percentage = risk_percentage
        self.usd_zar_rate = usd_zar_rate
    
    def calculate_zar_risk(self):
        """
        Calculate maximum risk capital in ZAR per trade.
        
        Formula: Account Balance (ZAR) * Risk Percentage / 100
        
        Returns:
            float: Maximum ZAR risk per trade
        """
        zar_risk = self.account_balance_zar * (self.risk_percentage / 100)
        return zar_risk
    
    def convert_to_usd(self, zar_amount):
        """
        Convert ZAR amount to USD using current exchange rate.
        
        Formula: ZAR Amount / USD-ZAR Rate
        
        Args:
            zar_amount (float): Amount in ZAR
        
        Returns:
            float: Equivalent amount in USD
        """
        usd_amount = zar_amount / self.usd_zar_rate
        return usd_amount
    
    def calculate_expected_value(self, win_rate, reward_to_risk):
        """
        Calculate expected value per trade.
        
        Formula: (Win Rate * Reward) - ((1 - Win Rate) * Risk)
        Where Risk = 1 (normalized) and Reward = R:R ratio
        
        Args:
            win_rate (float): Historical win rate as decimal (e.g., 0.55 for 55%)
            reward_to_risk (float): Average R:R ratio (e.g., 2.0 for 2:1)
        
        Returns:
            float: Expected value as decimal (negative = negative expectancy)
        """
        # EV = (Win% × Reward) - (Loss% × Risk)
        # Normalized: Risk = 1 USD, Reward = R:R ratio
        ev = (win_rate * reward_to_risk) - ((1 - win_rate) * 1.0)
        return ev
    
    def calculate_kelly_criterion(self, win_rate, reward_to_risk):
        """
        Calculate Kelly Criterion for optimal fraction sizing.
        
        Formula: f* = (bp - q) / b
        Where:
            b = odds (R:R ratio)
            p = win probability
            q = loss probability (1 - p)
        
        Args:
            win_rate (float): Historical win rate as decimal
            reward_to_risk (float): Average R:R ratio
        
        Returns:
            float: Kelly fraction (should divide by 2 for Half-Kelly in practice)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly formula: f* = (p * b - q) / b
        p = win_rate
        q = 1 - win_rate
        b = reward_to_risk
        
        kelly = (p * b - q) / b
        
        # Ensure Kelly is non-negative
        return max(kelly, 0.0)
    
    def calculate_probability_of_ruin(self, win_rate, reward_to_risk, kelly_fraction):
        """
        Estimate probability of ruin using Kelly-derived metrics.
        
        Simplified formula: Ruin Risk ≈ e^(-2 * f * Edge * n)
        Where f = Kelly fraction, Edge = EV, n = number of trades
        
        Args:
            win_rate (float): Historical win rate as decimal
            reward_to_risk (float): Average R:R ratio
            kelly_fraction (float): Applied Kelly fraction
        
        Returns:
            float: Estimated probability of ruin (0-1)
        """
        ev = self.calculate_expected_value(win_rate, reward_to_risk)
        
        # If negative expectancy, ruin is certain
        if ev <= 0:
            return 1.0
        
        # Simplified Gambler's ruin for trading
        # P(ruin) = ((1-p)/p)^(kelly_fraction * bankroll_in_units)
        # For practical purposes, low kelly fractions have near-zero ruin probability
        
        if kelly_fraction <= 0:
            return 0.0
        
        # Risk of ruin ≈ (q/p)^(kelly_fraction)
        p = win_rate
        q = 1 - win_rate
        
        if p == 0:
            return 1.0
        
        ruin_prob = (q / p) ** kelly_fraction
        return min(ruin_prob, 1.0)
