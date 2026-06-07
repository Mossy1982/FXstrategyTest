"""
Position Sizing Engine
Calculates lot sizes based on asset characteristics and risk parameters
"""


class PositionSizer:
    """
    Dynamically calculates position sizes based on asset pip values and risk capital.
    
    Supports major forex pairs, precious metals, and indices.
    """
    
    # Asset specifications: pip_value per standard lot
    ASSET_SPECS = {
        "EUR/USD": {
            "pip_value": 10.0,  # 1 pip = $10 per standard lot (100,000 units)
            "standard_lot": 100000,
            "description": "Euro/US Dollar"
        },
        "GBP/USD": {
            "pip_value": 10.0,  # 1 pip = $10 per standard lot
            "standard_lot": 100000,
            "description": "British Pound/US Dollar"
        },
        "USD/JPY": {
            "pip_value": 9.13,  # Approximately $9.13 per pip (varies with rate)
            "standard_lot": 100000,
            "description": "US Dollar/Japanese Yen"
        },
        "XAU/USD": {
            "pip_value": 100.0,  # 1 pip = $100 per standard lot (100 ounces)
            "standard_lot": 100,
            "description": "Gold/US Dollar (per 100 oz)"
        },
        "S&P 500": {
            "pip_value": 50.0,  # 1 pip = $50 per standard contract
            "standard_lot": 1,
            "description": "S&P 500 Index"
        }
    }
    
    def __init__(self, asset):
        """
        Initialize position sizer for a specific asset.
        
        Args:
            asset (str): Asset symbol (e.g., "EUR/USD")
        
        Raises:
            ValueError: If asset not in supported list
        """
        if asset not in self.ASSET_SPECS:
            raise ValueError(f"Asset {asset} not supported. Choose from: {list(self.ASSET_SPECS.keys())}")
        
        self.asset = asset
        self.asset_info = self.ASSET_SPECS[asset]
    
    def calculate_lot_size(self, usd_risk_capital, stop_loss_pips):
        """
        Calculate position size (in lots) based on risk capital and stop loss.
        
        Formula:
            Risk in USD = Lot Size × Pip Value × Stop Loss Pips
            Lot Size = Risk in USD / (Pip Value × Stop Loss Pips)
        
        Args:
            usd_risk_capital (float): Maximum USD to risk on the trade
            stop_loss_pips (float): Stop loss distance in pips/ticks
        
        Returns:
            float: Position size in lots (can be fractional for micro-lots)
        
        Example:
            usd_risk_capital = 200 USD
            stop_loss_pips = 50 pips
            pip_value = 10 (EUR/USD)
            lot_size = 200 / (10 * 50) = 0.4 standard lots
        """
        pip_value = self.asset_info["pip_value"]
        
        if stop_loss_pips <= 0:
            return 0.0
        
        # Lot size = Risk Capital / (Pip Value × Stop Loss Pips)
        lot_size = usd_risk_capital / (pip_value * stop_loss_pips)
        
        return lot_size
    
    def calculate_units_to_trade(self, lot_size):
        """
        Convert lot size to number of units to trade.
        
        Args:
            lot_size (float): Position size in lots
        
        Returns:
            float: Number of units to trade
        
        Example:
            lot_size = 0.4 standard lots
            standard_lot = 100,000 units
            units = 0.4 × 100,000 = 40,000 units
        """
        standard_lot = self.asset_info["standard_lot"]
        units = lot_size * standard_lot
        return units
    
    def calculate_pip_value_for_lot_size(self, lot_size):
        """
        Calculate the actual pip value for a given lot size.
        
        Args:
            lot_size (float): Position size in lots
        
        Returns:
            float: USD value per pip for this lot size
        """
        base_pip_value = self.asset_info["pip_value"]
        return lot_size * base_pip_value
    
    def get_asset_info(self):
        """
        Get full specification for the current asset.
        
        Returns:
            dict: Asset specifications
        """
        return self.asset_info
