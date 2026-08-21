import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import your custom modules
from pip_pattern_miner import PIPPatternMiner
from perceptually_important import find_pips

def main():
    print("Loading BTC dataset...")

    try:
        data = pd.read_csv('BTCUSDT_1h.csv')
    except FileNotFoundError:
        print('Error: Could not find path')
        return
    
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data['year'] = data.index.year
    
    # We must use log prices for the algorithm
    log_data = np.log(data)

    # --- 1. Filter for In-Sample Data (Before 2024) ---
    in_sample_data = log_data[log_data.index < '2024-01-01']
    arr = in_sample_data['close'].to_numpy()

    # Log returns for the period (Price_t+1 - Price_t)
    returns = in_sample_data['close'].diff().shift(-1).fillna(0).to_numpy()

    # --- 2. Train the Model ---
    n_pips = 5
    lookback = 24
    hold_period = 6

    print(f"Training PIP Miner on In-Sample data ({len(arr)} candles)...")
    pip_miner = PIPPatternMiner(n_pips=n_pips, lookback=lookback, hold_period=hold_period)
    pip_miner.train(arr, n_reps=-1) # Skip Monte Carlo for speed

    # --- 3. Run the Backtest Simulation ---
    print("Simulating Long-Only, Short-Only, and Combined portfolios...")
    
    pos_long = np.zeros(len(arr))
    pos_short = np.zeros(len(arr))
    pos_combined = np.zeros(len(arr))

    hold_long = 0
    hold_short = 0
    hold_combined = 0

    curr_long = 0
    curr_short = 0
    curr_combined = 0

    for i in range(lookback - 1, len(arr) - 1):
        window = arr[i - lookback + 1 : i + 1]
        _, pips_y = find_pips(window, n_pips, 3)
        signal = pip_miner.predict(pips_y)

        # --- COMBINED PORTFOLIO ---
        if hold_combined > 0:
            pos_combined[i] = curr_combined
            hold_combined -= 1
        else:
            if signal != 0.0:
                curr_combined = signal
                hold_combined = hold_period - 1
                pos_combined[i] = curr_combined
            else:
                curr_combined = 0

        # --- LONG ONLY PORTFOLIO ---
        if hold_long > 0:
            pos_long[i] = curr_long
            hold_long -= 1
        else:
            if signal == 1.0:
                curr_long = 1.0
                hold_long = hold_period - 1
                pos_long[i] = curr_long
            else:
                curr_long = 0

        # --- SHORT ONLY PORTFOLIO ---
        if hold_short > 0:
            pos_short[i] = curr_short
            hold_short -= 1
        else:
            if signal == -1.0:
                curr_short = -1.0
                hold_short = hold_period - 1
                pos_short[i] = curr_short
            else:
                curr_short = 0

    # --- 4. Calculate Equity Curves ---
    # Convert log returns back to absolute multipliers
    eq_combined = np.exp(np.cumsum(pos_combined * returns))
    eq_long = np.exp(np.cumsum(pos_long * returns))
    eq_short = np.exp(np.cumsum(pos_short * returns))
    eq_bh = np.exp(np.cumsum(returns)) # Buy and Hold Benchmark

    # --- 5. Calculate Sharpe Ratios ---
    def calc_sharpe(ret_array):
        mean_r = np.mean(ret_array)
        std_r = np.std(ret_array)
        return (mean_r / std_r) * np.sqrt(8760) if std_r > 0 else 0.0

    sharpe_combined = calc_sharpe(pos_combined * returns)
    sharpe_long = calc_sharpe(pos_long * returns)
    sharpe_short = calc_sharpe(pos_short * returns)
    sharpe_bh = calc_sharpe(returns)

    # Print summary statistics
    print("\n--- IN-SAMPLE Performance Summary (Pre-2024) ---")
    print(f"Combined Strategy Return: {(eq_combined[-1] - 1) * 100:>8.2f}% | Sharpe: {sharpe_combined:>5.2f}")
    print(f"Long-Only Return:         {(eq_long[-1] - 1) * 100:>8.2f}% | Sharpe: {sharpe_long:>5.2f}")
    print(f"Short-Only Return:        {(eq_short[-1] - 1) * 100:>8.2f}% | Sharpe: {sharpe_short:>5.2f}")
    print(f"Buy & Hold Return:        {(eq_bh[-1] - 1) * 100:>8.2f}% | Sharpe: {sharpe_bh:>5.2f}")

    # --- 5. Visualize Results ---
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 7))

    dates = in_sample_data.index

    plt.plot(dates, eq_bh, color='gray', alpha=0.5, label='Buy & Hold Benchmark')
    plt.plot(dates, eq_long, color='lime', linewidth=1.5, label='Long Only')
    plt.plot(dates, eq_short, color='red', linewidth=1.5, label='Short Only')
    plt.plot(dates, eq_combined, color='cyan', linewidth=2, label='Combined Strategy')

    plt.title("In-Sample Strategy Evaluation (Pre-2024)", fontsize=16, fontweight='bold', color='white')
    plt.ylabel("Cumulative Return Multiplier")
    plt.xlabel("Date")
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(color='white', alpha=0.1, linestyle='--')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()