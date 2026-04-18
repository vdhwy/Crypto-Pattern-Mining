import requests
import pandas as pd
import time
from datetime import datetime

def get_binance_historical_data(symbol, interval, start_str, end_str):
    # Convert date strings to milliseconds for the Binance API
    start_ts = int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, '%Y-%m-%d').timestamp() * 1000)
    
    url = 'https://api.binance.com/api/v3/klines'
    limit = 1000 # Binance maximum allowed limit per request
    all_data = []
    
    print(f"Fetching {symbol} data from {start_str} to {end_str}...")
    
    # Loop to paginate through time until we reach the end date
    while start_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.text}")
            break
            
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        
        # Update start_ts to the last fetched candle's timestamp + 1 millisecond
        # to avoid fetching the exact same candle twice
        start_ts = data[-1][0] + 1
        
        # Briefly pause to respect Binance's API rate limits
        time.sleep(0.1)
        
    # Format the raw JSON array into a Pandas DataFrame
    columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ]
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Convert timestamp columns to readable UTC datetime objects
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    
    # Drop the 'Ignore' column and convert price/volume strings to floats
    df.drop(columns=['Ignore'], inplace=True)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'Quote Asset Volume', 'Taker Buy Base Asset Volume', 
                    'Taker Buy Quote Asset Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    df = df[["Open Time","Open","High","Low","Close","Volume"]]
    df.rename(columns={"Open Time": "date"}, inplace=True)
    df.rename(columns={"Open": "open"}, inplace=True)
    df.rename(columns={"High": "high"}, inplace=True)
    df.rename(columns={"Low": "low"}, inplace=True)
    df.rename(columns={"Close": "close"}, inplace=True)
    df.rename(columns={"Volume": "volume"}, inplace=True)
    
    return df

# --- Execution ---

# Setup parameters
symbol = 'BTCUSDT'
interval = '1h' # '1d' = Daily. You can change this to '1h' (hourly) or '1m' (minute)
start_date = '2020-01-01' # Earliest available BTCUSDT data on Binance
end_date = '2025-01-01'

# Run the function
df_btc = get_binance_historical_data(symbol, interval, start_date, end_date)

# Save the resulting DataFrame to a CSV file
filename = f"{symbol}_{interval}.csv"
df_btc.to_csv(filename, index=False)

print(f"\nSuccessfully saved {len(df_btc)} rows of data to {filename}!")
print(df_btc.head())