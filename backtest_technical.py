import pandas as pd
from backtesting import Backtest, Strategy
from data.fetcher import get_market_data_enhanced
from analysis.technical import calculate_enhanced_technical_analysis, generate_technical_signals

# --- 1. Define the Backtesting Strategy ---
class TechnicalSignalStrategy(Strategy):
    def init(self):
        # Pre-calculate all technical indicators and signals
        print("Pre-calculating indicators and signals...")
        self.signals = []
        # We need to iterate through the data history to generate a signal for each day
        for i in range(len(self.data.Close)):
            if i < 50:  # Need at least 50 days of data for calculations
                self.signals.append("HOLD")
                continue
            
            # Create a data slice and analysis_results dictionary for each day
            current_data_slice = self.data.df.iloc[:i+1]
            current_price = self.data.Close[i]

            # This is a simplified analysis_results structure for backtesting
            # It mimics the structure the signal function expects
            analysis_results = {
                'current_price': current_price,
                'enhanced_indicators': calculate_enhanced_technical_analysis(current_data_slice)
            }
            
            signal = generate_technical_signals(analysis_results)
            self.signals.append(signal)
        
        print("Calculation complete.")

    def next(self):
        # Get the signal for the current day
        current_signal = self.signals[len(self.data.Close)-1]
        
        # --- Trading Logic ---
        if "BUY" in current_signal and not self.position:
            self.buy() # Enter long position
            
        elif "SELL" in current_signal and self.position:
            self.position.close() # Exit long position

# --- 2. Main Execution Block ---
if __name__ == "__main__":
    # Fetch historical data
    print("Fetching historical data for backtest...")
    # Using a longer period for a more meaningful backtest
    hist_data = get_market_data_enhanced(symbol='SPY', period='5y')
    
    if hist_data is not None:
        print("Data fetched successfully. Starting backtest...")
        
        # Configure and run the backtest
        bt = Backtest(hist_data, TechnicalSignalStrategy, cash=100000, commission=.002)
        
        stats = bt.run()
        
        # Print the results
        print("\n--- Backtest Results ---")
        print(stats)
        
        print("\n--- Trades ---")
        print(stats['_trades'])
        
        # Plot the backtest
        bt.plot()
        
    else:
        print("Could not fetch data. Backtest aborted.")
