import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import your custom modules
from pip_pattern_miner import PIPPatternMiner
from perceptually_important import find_pips

def calculate_drawdown(equity_curve):
    """Calculates the underwater drawdown curve."""
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return drawdown

def main():
    print("Loading BTC dataset...")
    try:
        data = pd.read_csv('BTCUSDT_1h.csv') 
    except FileNotFoundError:
        print("Error: Could not find your CSV file.")
        return

    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data['year'] = data.index.year
    
    log_data = np.log(data['close'])

    # Get the unique years in the dataset
    available_years = sorted(data['year'].unique())
    
    # Strategy parameters
    n_pips = 5
    lookback = 24
    hold_period = 6

    # Lists to store the concatenated Out-of-Sample results
    oos_dates = []
    oos_log_returns = []
    oos_positions = []
    
    # Trade tracking
    trades_won = 0
    total_trades = 0
    total_longs = 0
    total_shorts = 0

    print("\n" + "="*50)
    print("🚀 STARTING WALK-FORWARD OUT-OF-SAMPLE TEST 🚀")
    print("="*50)

    # Loop through the years to create our rolling windows
    for i in range(len(available_years) - 2):
        train_year_1 = available_years[i]
        train_year_2 = available_years[i + 1]
        test_year = available_years[i + 2]

        if test_year > 2025:
            break # Stop if we exceed 2025

        print(f"\n🔄 Window {i+1}: Train [{train_year_1}-{train_year_2}] -> Test [{test_year}]")

        # 1. Isolate Training Data (2 Years)
        train_mask = (data['year'] == train_year_1) | (data['year'] == train_year_2)
        train_arr = log_data[train_mask].to_numpy()

        # 2. Isolate Testing Data (1 Year)
        test_mask = (data['year'] == test_year)
        test_series = log_data[test_mask]
        test_arr = test_series.to_numpy()
        
        # Use LOG returns consistently (diff of log prices = log returns)
        test_log_returns = pd.Series(test_arr).diff().shift(-1).fillna(0).to_numpy()

        # 3. Train the Model on the 2-Year Window
        pip_miner = PIPPatternMiner(n_pips=n_pips, lookback=lookback, hold_period=hold_period)
        pip_miner.train(train_arr, n_reps=-1)

        # 4. Simulate the 1-Year Out-of-Sample Test
        pos_array = np.zeros(len(test_arr))
        hold_counter = 0
        curr_pos = 0
        trade_entry_idx = -1
        
        # Per-year trade counters
        year_long_trades = 0
        year_short_trades = 0
        year_trades_won = 0
        year_total_trades = 0

        for j in range(lookback - 1, len(test_arr) - 1):
            if hold_counter > 0:
                pos_array[j] = curr_pos
                hold_counter -= 1
                
                # Check win/loss when trade ends
                if hold_counter == 0:
                    trade_return = np.sum(test_log_returns[trade_entry_idx : j + 1])
                    if (curr_pos == 1.0 and trade_return > 0) or (curr_pos == -1.0 and trade_return < 0):
                        trades_won += 1
                        year_trades_won += 1
                    curr_pos = 0
            else:
                window = test_arr[j - lookback + 1 : j + 1]
                _, pips_y = find_pips(window, n_pips, 3)
                signal = pip_miner.predict(pips_y)

                if signal != 0.0:
                    curr_pos = signal
                    hold_counter = hold_period - 1
                    pos_array[j] = curr_pos
                    trade_entry_idx = j
                    total_trades += 1
                    year_total_trades += 1
                    if signal == 1.0:
                        year_long_trades += 1
                        total_longs += 1
                    else:
                        year_short_trades += 1
                        total_shorts += 1
                else:
                    curr_pos = 0

        # 5. Append this year's OOS results to our master list
        oos_dates.extend(test_series.index)
        oos_log_returns.extend(test_log_returns)
        oos_positions.extend(pos_array)
        
        strat_hourly_returns = pos_array * test_log_returns
        
        # Calculate annualized Sharpe Ratio for 1h data (8760 hours/year)
        mean_ret = np.mean(strat_hourly_returns)
        std_ret = np.std(strat_hourly_returns)
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(8760) if std_ret > 0 else 0.0
        year_win_rate = (year_trades_won / year_total_trades * 100) if year_total_trades > 0 else 0
        
        # Quick summary for the year
        year_strat_ret = np.exp(np.cumsum(pos_array * test_log_returns))[-1] - 1
        year_bh_ret = np.exp(np.cumsum(test_log_returns))[-1] - 1
        print(f"   -> OOS Strategy: {year_strat_ret*100:>7.2f}% | Benchmark: {year_bh_ret*100:>7.2f}% | Sharpe: {sharpe_ratio:>5.2f}")
        print(f"      Longs: {year_long_trades} | Shorts: {year_short_trades} | Win Rate: {year_win_rate:.1f}%")
    # --- Compile Final Master OOS Metrics ---
    oos_dates = pd.to_datetime(oos_dates)
    oos_log_returns = np.array(oos_log_returns)
    oos_positions = np.array(oos_positions)

    # Strategy vs Benchmark Equity
    strat_returns = oos_positions * oos_log_returns
    eq_strat = np.exp(np.cumsum(strat_returns))
    eq_bh = np.exp(np.cumsum(oos_log_returns))
    
    drawdown = calculate_drawdown(eq_strat)
    max_dd = np.min(drawdown) * 100
    win_rate = (trades_won / total_trades * 100) if total_trades > 0 else 0

    mean_strat = np.mean(strat_returns)
    std_strat = np.std(strat_returns)
    final_sharpe = (mean_strat / std_strat) * np.sqrt(8760) if std_strat > 0 else 0.0

    mean_bh = np.mean(oos_log_returns)
    std_bh = np.std(oos_log_returns)
    final_bh_sharpe = (mean_bh / std_bh) * np.sqrt(8760) if std_bh > 0 else 0.0

    print("\n" + "="*50)
    print("🏆 FINAL TRUE OUT-OF-SAMPLE METRICS 🏆")
    print("="*50)
    print(f"Total True Return:    {(eq_strat[-1] - 1) * 100:>8.2f}%")
    print(f"Buy & Hold Benchmark: {(eq_bh[-1] - 1) * 100:>8.2f}%")
    print(f"Strategy Sharpe:      {final_sharpe:>8.2f}")
    print(f"Benchmark Sharpe:     {final_bh_sharpe:>8.2f}")
    print(f"Maximum Drawdown:     {max_dd:>8.2f}%")
    print(f"Total Trades Taken:   {total_trades} (Long: {total_longs}, Short: {total_shorts})")
    print(f"True Win Rate:        {win_rate:>8.2f}%")
    print("="*50)

    # --- Visualizing the Master OOS Curve ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax1.plot(oos_dates, eq_bh, color='gray', alpha=0.5, label='Buy & Hold Benchmark')
    ax1.plot(oos_dates, eq_strat, color='cyan', linewidth=2, label='Walk-Forward Strategy PnL')
    ax1.set_title("True Out-of-Sample Walk-Forward Backtest", fontsize=16, fontweight='bold', color='white')
    ax1.set_ylabel("Cumulative Multiplier")
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(color='white', alpha=0.1, linestyle='--')

    ax2.fill_between(oos_dates, drawdown * 100, 0, color='red', alpha=0.4)
    ax2.plot(oos_dates, drawdown * 100, color='red', linewidth=1)
    ax2.set_title("Strategy Drawdown (%)", fontsize=14, color='white')
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(color='white', alpha=0.1, linestyle='--')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()