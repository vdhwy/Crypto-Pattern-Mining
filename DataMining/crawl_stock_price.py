import yfinance as yf
import pandas as pd
import os

# Note: Removed the duplicate 'NVDA' from the list
STOCKS = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'META', 'NVDA', 'SNP']

def fetch_1h_data(ticker_symbol):
    print(f"Connecting to Yahoo Finance for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    
    # Fetch the historical data
    print(f"Fetching 1-hour candle data for {ticker_symbol} (max 730 days)...")
    df = ticker.history(period="730d", interval="1h")
     
    # Check if the dataframe is empty
    if df.empty:
        print(f"No data found for {ticker_symbol}. This could be due to a network error or rate limiting.")
        return
        
    # Keep only the standard candlestick columns
    df = df[['Open', 'High', 'Low', 'Close']]
    
    # Rename the index (the first column in the CSV) to 'date'
    df.index.name = 'date'
    
    # Rename the rest of the columns to lowercase
    df.rename(columns={
        "Open": "open", 
        "High": "high", 
        "Low": "low", 
        "Close": "close", 
        "Volume": "volume"
    }, inplace=True)
    
    
    # --- NEW: Folder handling ---
    folder_name = 'stock_data'
    
    # Create the directory if it doesn't already exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Join the folder name and file name safely
    csv_filename = os.path.join(folder_name, f"{ticker_symbol}_1h_candles.csv")
    
    # Save to the CSV file inside the target folder
    df.to_csv(csv_filename)
    
    print(f"Success! Extracted {len(df)} rows of data.")
    print(f"Data has been saved locally to: {csv_filename}\n")

if __name__ == "__main__":
    # Loop through the list and pass each ticker to the function
    for ticker in STOCKS:
        fetch_1h_data(ticker)