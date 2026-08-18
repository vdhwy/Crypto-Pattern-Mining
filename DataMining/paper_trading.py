import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Alpaca SDK Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv

# Import your custom strategy modules
from pip_pattern_miner import PIPPatternMiner
from perceptually_important import find_pips

load_dotenv() 

# Pull the keys securely into your script
API_KEY = os.getenv("ALPACA_PAPER_API_KEY")
SECRET_KEY = os.getenv("ALPACA_PAPER_SECRET_KEY")

# Safety check to ensure the keys loaded correctly
if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing API Keys! Check your .env file.")

SYMBOL = "BTC/USD"
TRADE_QTY = 0.01
SYMBOL = "BTC/USD"
TRADE_QTY = 0.01  # How much BTC to buy/sell per trade

class LivePIPMinerBot:
    def __init__(self):
        print("Initializing Alpaca Clients...")
        # paper=True ensures we are NOT using real money!
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self.data_client = CryptoHistoricalDataClient()
        
        # Strategy Parameters
        self.n_pips = 5
        self.lookback = 24
        self.hold_period = 6
        self.hold_counter = 0
        self.current_position = 0 # 1 for Long, -1 for Short, 0 for Flat

        # The AI Model
        self.miner = PIPPatternMiner(
            n_pips=self.n_pips, 
            lookback=self.lookback, 
            hold_period=self.hold_period
        )
        self.trained = False

    def fetch_historical_data(self, days_back: int) -> np.array:
        """Fetches hourly historical data from Alpaca."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)

        request_params = CryptoBarsRequest(
            symbol_or_symbols=[SYMBOL],
            timeframe=TimeFrame.Hour,
            start=start_time,
            end=end_time
        )
        
        bars = self.data_client.get_crypto_bars(request_params)
        df = bars.df
        
        # Alpaca returns a multi-index dataframe. We extract just the close prices.
        # We also apply the log transformation exactly like your backtest!
        close_prices = df['close'].to_numpy()
        log_prices = np.log(close_prices)
        
        return log_prices

    def initial_training(self):
        """Trains the model on the last 1 year of data before starting live trading."""
        print("Fetching 365 days of historical data for initial training...")
        train_data = self.fetch_historical_data(days_back=365)
        
        print(f"Training PIP Miner on {len(train_data)} hourly candles. This may take a minute...")
        self.miner.train(train_data, n_reps=-1)
        self.trained = True
        print(f"Training Complete! Found {len(self.miner._pip_clusters)} profitable clusters.")

    def close_all_positions(self):
        """Closes any open positions for our symbol."""
        print(f"Closing out all open positions for {SYMBOL}...")
        self.trading_client.close_all_positions(cancel_orders=True)
        self.current_position = 0

    def execute_trade(self, signal: float):
        """Places a market order via Alpaca based on the signal."""
        try:
            if signal == 1.0:
                print(f"Submitting LONG Market Order for {TRADE_QTY} {SYMBOL}...")
                order_data = MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=TRADE_QTY,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )
                self.trading_client.submit_order(order_data=order_data)
                self.current_position = 1
                self.hold_counter = self.hold_period
                
            elif signal == -1.0:
                print(f"Submitting SHORT Market Order for {TRADE_QTY} {SYMBOL}...")
                order_data = MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=TRADE_QTY,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
                self.trading_client.submit_order(order_data=order_data)
                self.current_position = -1
                self.hold_counter = self.hold_period

        except Exception as e:
            print(f"Error executing trade: {e}")

    def run(self):
        """The main continuous trading loop."""
        self.initial_training()
        
        # Ensure we start flat
        self.close_all_positions()

        print("\n" + "="*40)
        print("🟢 LIVE PAPER TRADING BOT ACTIVATED 🟢")
        print("="*40)

        while True:
            # 1. Wait until exactly the top of the next hour
            now = datetime.utcnow()
            next_hour = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
            wait_seconds = (next_hour - now).total_seconds()
            
            print(f"[{now.strftime('%H:%M:%S')} UTC] Waiting {int(wait_seconds/60)} minutes for the next hourly candle...")
            time.sleep(wait_seconds)
            
            print(f"\n[{datetime.utcnow().strftime('%H:%M:%S')} UTC] New hourly candle detected. Analyzing market...")

            # 2. Are we currently in a trade? Handle the Hold Period.
            if self.hold_counter > 0:
                self.hold_counter -= 1
                print(f"Holding current position. {self.hold_counter} hours remaining.")
                
                if self.hold_counter == 0:
                    print("Hold period complete. Exiting position.")
                    self.close_all_positions()
                
                # If we are holding, we skip looking for new patterns
                continue

            # 3. We are flat. Look for new patterns.
            # We fetch a little extra (5 days) to ensure we have a clean lookback window
            recent_data = self.fetch_historical_data(days_back=5)
            
            # Grab the last 24 hours
            window = recent_data[-self.lookback:]
            
            if len(window) < self.lookback:
                print("Not enough data fetched to form a window. Skipping this hour.")
                continue

            # Calculate PIPs and Predict
            _, pips_y = find_pips(window, self.n_pips, 3)
            signal = self.miner.predict(pips_y)

            if signal == 1.0:
                print("🔥 LONG SIGNAL DETECTED 🔥")
                self.execute_trade(1.0)
            elif signal == -1.0:
                print("🩸 SHORT SIGNAL DETECTED 🩸")
                self.execute_trade(-1.0)
            else:
                print("No profitable pattern detected. Remaining flat.")


if __name__ == "__main__":
    bot = LivePIPMinerBot()
    # Runs infinitely until you kill the terminal (Ctrl+C)
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nBot stopped manually. Shutting down gracefully...")
        # Optional: bot.close_all_positions()