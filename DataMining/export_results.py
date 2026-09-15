"""
Export results from all pipeline scripts as images for the README.
Saves: pip_miner_results.png, IS_backtest_results.png, 
       long_short_results.png, OOS_backtest_results.png
"""
# Force pyclustering to use pure Python (C-core is x86_64, incompatible with ARM64 Mac)
import numpy as np
import warnings
if not hasattr(np, 'warnings'):
    np.warnings = warnings  # Shim for pyclustering compatibility with numpy >= 1.24
    
import pyclustering.core.wrapper as _pw
_pw.ccore_library.workable = staticmethod(lambda: False)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import sys
import os

from pip_pattern_miner import PIPPatternMiner
from perceptually_important import find_pips
from wf_pip_miner import WFPIPMiner

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def export_pip_pattern_miner():
    """Run pip_pattern_miner.py logic and save the MCPT histogram."""
    print("\n" + "="*60)
    print("📊 EXPORTING: pip_pattern_miner.py (Pattern Mining + MCPT)")
    print("="*60)
    
    data = pd.read_csv('BTCUSDT_1h.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = np.log(data)
    
    data = data[data.index < '01-01-2024']
    arr = data['close'].to_numpy()
    
    pip_miner = PIPPatternMiner(n_pips=5, lookback=24, hold_period=6)
    pip_miner.train(arr, n_reps=-1)
    
    actual_martin = pip_miner.get_fit_martin()
    
    # Read existing MCPT results if available (to avoid 1-hour re-run)
    perm_martins = []
    if os.path.exists('mcpt_test_results.txt'):
        with open('mcpt_test_results.txt', 'r') as f:
            content = f.read()
            # Parse the permutation martins from the file
            if 'Permutation Martins:' in content:
                martins_str = content.split('Permutation Martins:\n')[1]
                # Parse np.float64 values
                import re
                values = re.findall(r'np\.float64\(([\d.e+-]+)\)', martins_str)
                perm_martins = [float(v) for v in values]
    
    if len(perm_martins) > 0:
        p_val = np.sum(np.array(perm_martins) >= actual_martin) / len(perm_martins)
    else:
        p_val = 0
    
    print(f"   Actual Martin Ratio: {actual_martin:.4f}")
    print(f"   P-Value: {p_val:.4f}")
    print(f"   Number of Permutations: {len(perm_martins)}")
    
    # Generate MCPT histogram
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if len(perm_martins) > 0:
        ax.hist(perm_martins, bins=30, color='#4A90D9', edgecolor='#2C5F8A', alpha=0.85, label='Permutation Distribution')
    
    ax.axvline(actual_martin, color='#FF4444', linewidth=2.5, linestyle='--', label=f'Actual ({actual_martin:.2f})')
    ax.set_ylabel("# Of Permutations", fontsize=12)
    ax.set_xlabel("Martin Ratio", fontsize=12)
    ax.set_title(f"Monte Carlo Permutation Test — BTC/USDT (p-value: {p_val:.3f})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/pip_miner_mcpt.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"   ✅ Saved {RESULTS_DIR}/pip_miner_mcpt.png")
    
    return pip_miner, data


def export_cluster_grid(pip_miner, candle_data_log):
    """Generate a grid showing a sample of cluster archetypes."""
    print("\n" + "="*60)
    print("📊 EXPORTING: Cluster Pattern Grid")
    print("="*60)
    
    # Read raw candle data for plotting
    raw_data = pd.read_csv('BTCUSDT_1h.csv')
    raw_data['date'] = raw_data['date'].astype('datetime64[s]')
    raw_data = raw_data.set_index('date')
    raw_data = raw_data[raw_data.index < '01-01-2024']
    
    n_clusters = len(pip_miner._pip_clusters)
    # Show up to 8 clusters in a 2x4 grid, each with 1 example pattern
    n_show = min(n_clusters, 8)
    
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    flat_axs = axs.flatten()
    
    for ci in range(n_show):
        cluster = pip_miner._pip_clusters[ci]
        if len(cluster) == 0:
            continue
        
        # Pick the first member of each cluster
        pat_i = pip_miner._unique_pip_indices[cluster[0]]
        lookback = pip_miner._lookback
        data_slice = raw_data.iloc[pat_i - lookback + 1: pat_i + 1]
        idx = data_slice.index
        
        plot_pip_x, plot_pip_y = find_pips(data_slice['close'].to_numpy(), pip_miner._n_pips, 3)
        
        pip_lines = []
        colors = []
        for line_i in range(pip_miner._n_pips - 1):
            l0 = [(idx[plot_pip_x[line_i]], plot_pip_y[line_i]), 
                  (idx[plot_pip_x[line_i + 1]], plot_pip_y[line_i + 1])]
            pip_lines.append(l0)
            colors.append('w')
        
        mpf.plot(data_slice, type='candle', alines=dict(alines=pip_lines, colors=colors),
                 ax=flat_axs[ci], style='charles', update_width_config=dict(candle_linewidth=1.75))
        
        # Determine if this cluster is long/short/neutral
        label = "Neutral"
        color = "white"
        if ci in pip_miner._selected_long:
            label = "LONG ⬆"
            color = "#00FF88"
        elif ci in pip_miner._selected_short:
            label = "SHORT ⬇"
            color = "#FF4444"
        
        flat_axs[ci].set_title(f"Cluster {ci} ({label})", color=color, fontsize=11, fontweight='bold')
        flat_axs[ci].set_yticklabels([])
        flat_axs[ci].set_xticklabels([])
        flat_axs[ci].set_xticks([])
        flat_axs[ci].set_yticks([])
        flat_axs[ci].set_ylabel("")
    
    # Hide unused axes
    for ci in range(n_show, len(flat_axs)):
        flat_axs[ci].set_visible(False)
    
    fig.suptitle("Discovered Cluster Archetypes (K-Means on PIP Vectors)", fontsize=16, fontweight='bold', color='white')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{RESULTS_DIR}/cluster_archetypes.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"   ✅ Saved {RESULTS_DIR}/cluster_archetypes.png")


def export_is_backtest():
    """Run IS_backtest.py logic and save the equity curve plot."""
    print("\n" + "="*60)
    print("📊 EXPORTING: IS_backtest.py (In-Sample Backtest)")
    print("="*60)
    
    data = pd.read_csv('BTCUSDT_1h.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    
    log_data = np.log(data)
    in_sample_data = log_data[log_data.index < '2024-01-01']
    arr = in_sample_data['close'].to_numpy()
    returns = in_sample_data['close'].diff().shift(-1).fillna(0).to_numpy()
    
    n_pips = 5
    lookback = 24
    hold_period = 6
    
    pip_miner = PIPPatternMiner(n_pips=n_pips, lookback=lookback, hold_period=hold_period)
    pip_miner.train(arr, n_reps=-1)
    
    # Simulation
    pos_long = np.zeros(len(arr))
    pos_short = np.zeros(len(arr))
    pos_combined = np.zeros(len(arr))
    hold_long = hold_short = hold_combined = 0
    curr_long = curr_short = curr_combined = 0
    
    for i in range(lookback - 1, len(arr) - 1):
        window = arr[i - lookback + 1 : i + 1]
        _, pips_y = find_pips(window, n_pips, 3)
        signal = pip_miner.predict(pips_y)
        
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
    
    # Equity curves
    eq_combined = np.exp(np.cumsum(pos_combined * returns))
    eq_long = np.exp(np.cumsum(pos_long * returns))
    eq_short = np.exp(np.cumsum(pos_short * returns))
    eq_bh = np.exp(np.cumsum(returns))
    
    # Sharpe ratios
    def calc_sharpe(ret_array):
        mean_r = np.mean(ret_array)
        std_r = np.std(ret_array)
        return (mean_r / std_r) * np.sqrt(8760) if std_r > 0 else 0.0
    
    sharpe_combined = calc_sharpe(pos_combined * returns)
    sharpe_long = calc_sharpe(pos_long * returns)
    sharpe_short = calc_sharpe(pos_short * returns)
    sharpe_bh = calc_sharpe(returns)
    
    print(f"   Combined Return: {(eq_combined[-1] - 1) * 100:.2f}% | Sharpe: {sharpe_combined:.2f}")
    print(f"   Long-Only Return: {(eq_long[-1] - 1) * 100:.2f}% | Sharpe: {sharpe_long:.2f}")
    print(f"   Short-Only Return: {(eq_short[-1] - 1) * 100:.2f}% | Sharpe: {sharpe_short:.2f}")
    print(f"   Buy & Hold Return: {(eq_bh[-1] - 1) * 100:.2f}% | Sharpe: {sharpe_bh:.2f}")
    
    # Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    dates = in_sample_data.index
    
    ax.plot(dates, eq_bh, color='gray', alpha=0.5, label=f'Buy & Hold (Sharpe: {sharpe_bh:.2f})')
    ax.plot(dates, eq_long, color='#00FF88', linewidth=1.5, label=f'Long Only (Sharpe: {sharpe_long:.2f})')
    ax.plot(dates, eq_short, color='#FF4444', linewidth=1.5, label=f'Short Only (Sharpe: {sharpe_short:.2f})')
    ax.plot(dates, eq_combined, color='#00D4FF', linewidth=2, label=f'Combined Strategy (Sharpe: {sharpe_combined:.2f})')
    
    ax.set_title("In-Sample Strategy Evaluation (Pre-2024)", fontsize=16, fontweight='bold', color='white')
    ax.set_ylabel("Cumulative Return Multiplier", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(color='white', alpha=0.1, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/IS_backtest_results.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"   ✅ Saved {RESULTS_DIR}/IS_backtest_results.png")
    
    return {
        'combined_return': (eq_combined[-1] - 1) * 100,
        'long_return': (eq_long[-1] - 1) * 100,
        'short_return': (eq_short[-1] - 1) * 100,
        'bh_return': (eq_bh[-1] - 1) * 100,
        'sharpe_combined': sharpe_combined,
        'sharpe_long': sharpe_long,
        'sharpe_short': sharpe_short,
        'sharpe_bh': sharpe_bh,
    }


def export_long_short_examples():
    """Run long_short_example.py logic and save the grid plot."""
    print("\n" + "="*60)
    print("📊 EXPORTING: long_short_example.py (Long/Short Trigger Grid)")
    print("="*60)
    
    data = pd.read_csv('BTCUSDT_1h.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    plot_data = data.copy()
    
    data = np.log(data)
    arr = data['close'].to_numpy()
    
    N_PIPS = 5
    LOOKBACK = 24
    HOLD_PERIOD = 6
    NUM_EXAMPLES = 9
    
    wf_miner = WFPIPMiner(
        n_pips=N_PIPS, lookback=LOOKBACK, hold_period=HOLD_PERIOD,
        train_size=24 * 365 * 2, step_size=24 * 365 * 1
    )
    
    last_sig = 0.0
    long_slices = []
    short_slices = []
    
    print(f"   Scanning for {NUM_EXAMPLES} Long and {NUM_EXAMPLES} Short examples...")
    
    for i in range(len(arr)):
        curr_sig = wf_miner.update_signal(arr, i)
        
        if curr_sig == 1.0 and last_sig != 1.0:
            if len(long_slices) < NUM_EXAMPLES:
                long_slices.append(plot_data.iloc[i - LOOKBACK + 1: i + 1])
        elif curr_sig == -1.0 and last_sig != -1.0:
            if len(short_slices) < NUM_EXAMPLES:
                short_slices.append(plot_data.iloc[i - LOOKBACK + 1: i + 1])
        
        last_sig = curr_sig
        
        if len(long_slices) == NUM_EXAMPLES and len(short_slices) == NUM_EXAMPLES:
            print("   Found enough examples!")
            break
    
    # Create grid plot
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, NUM_EXAMPLES, figsize=(20, 8))
    fig.suptitle("Long vs. Short Pattern Triggers (Walk-Forward)", fontsize=18, fontweight='bold', color='white')
    
    for i, data_slice in enumerate(long_slices):
        plot_pip_x, plot_pip_y = find_pips(data_slice['close'].to_numpy(), N_PIPS, 3)
        idx = data_slice.index
        pip_lines = []
        colors = []
        for line_i in range(N_PIPS - 1):
            l0 = [(idx[plot_pip_x[line_i]], plot_pip_y[line_i]),
                  (idx[plot_pip_x[line_i + 1]], plot_pip_y[line_i + 1])]
            pip_lines.append(l0)
            colors.append('w')
        
        mpf.plot(data_slice, type='candle', alines=dict(alines=pip_lines, colors=colors),
                 ax=axs[0, i], style='charles', update_width_config=dict(candle_linewidth=1.75))
        
        end_date = data_slice.index[-1].strftime('%Y-%m-%d %H:00')
        axs[0, i].set_title(f"LONG ⬆\n{end_date}", color='#00FF88', fontsize=9, fontweight='bold')
        axs[0, i].set_yticklabels([])
        axs[0, i].set_xticklabels([])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[0, i].set_ylabel("")
    
    for i, data_slice in enumerate(short_slices):
        plot_pip_x, plot_pip_y = find_pips(data_slice['close'].to_numpy(), N_PIPS, 3)
        idx = data_slice.index
        pip_lines = []
        colors = []
        for line_i in range(N_PIPS - 1):
            l0 = [(idx[plot_pip_x[line_i]], plot_pip_y[line_i]),
                  (idx[plot_pip_x[line_i + 1]], plot_pip_y[line_i + 1])]
            pip_lines.append(l0)
            colors.append('w')
        
        mpf.plot(data_slice, type='candle', alines=dict(alines=pip_lines, colors=colors),
                 ax=axs[1, i], style='charles', update_width_config=dict(candle_linewidth=1.75))
        
        end_date = data_slice.index[-1].strftime('%Y-%m-%d %H:00')
        axs[1, i].set_title(f"SHORT ⬇\n{end_date}", color='#FF4444', fontsize=9, fontweight='bold')
        axs[1, i].set_yticklabels([])
        axs[1, i].set_xticklabels([])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        axs[1, i].set_ylabel("")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{RESULTS_DIR}/long_short_results.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"   ✅ Saved {RESULTS_DIR}/long_short_results.png")


def export_oos_backtest():
    """Run OOS_backtest.py logic and save the equity + drawdown plot."""
    print("\n" + "="*60)
    print("📊 EXPORTING: OOS_backtest.py (Walk-Forward Out-of-Sample)")
    print("="*60)
    
    data = pd.read_csv('BTCUSDT_1h.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data['year'] = data.index.year
    
    log_data = np.log(data['close'])
    available_years = sorted(data['year'].unique())
    
    n_pips = 5
    lookback = 24
    hold_period = 6
    
    oos_dates = []
    oos_log_returns = []
    oos_positions = []
    trades_won = 0
    total_trades = 0
    
    window_results = []
    
    for i in range(len(available_years) - 2):
        train_year_1 = available_years[i]
        train_year_2 = available_years[i + 1]
        test_year = available_years[i + 2]
        
        if test_year > 2025:
            break
        
        print(f"   Window {i+1}: Train [{train_year_1}-{train_year_2}] -> Test [{test_year}]")
        
        train_mask = (data['year'] == train_year_1) | (data['year'] == train_year_2)
        train_arr = log_data[train_mask].to_numpy()
        
        test_mask = (data['year'] == test_year)
        test_series = log_data[test_mask]
        test_arr = test_series.to_numpy()
        
        # Use LOG returns consistently (diff of log prices = log returns)
        test_log_returns = pd.Series(test_arr).diff().shift(-1).fillna(0).to_numpy()
        
        pip_miner = PIPPatternMiner(n_pips=n_pips, lookback=lookback, hold_period=hold_period)
        pip_miner.train(train_arr, n_reps=-1)
        
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
                    else:
                        year_short_trades += 1
                else:
                    curr_pos = 0
        
        oos_dates.extend(test_series.index)
        oos_log_returns.extend(test_log_returns)
        oos_positions.extend(pos_array)
        
        strat_hourly_returns = pos_array * test_log_returns
        mean_ret = np.mean(strat_hourly_returns)
        std_ret = np.std(strat_hourly_returns)
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(8760) if std_ret > 0 else 0.0
        
        year_strat_ret = np.exp(np.cumsum(pos_array * test_log_returns))[-1] - 1
        year_bh_ret = np.exp(np.cumsum(test_log_returns))[-1] - 1
        year_win_rate = (year_trades_won / year_total_trades * 100) if year_total_trades > 0 else 0
        
        window_results.append({
            'window': f"{train_year_1}-{train_year_2} → {test_year}",
            'test_year': test_year,
            'strat_return': year_strat_ret * 100,
            'bh_return': year_bh_ret * 100,
            'sharpe': sharpe_ratio,
            'long_trades': year_long_trades,
            'short_trades': year_short_trades,
            'total_trades': year_total_trades,
            'win_rate': year_win_rate,
        })
        
        print(f"      -> OOS Strategy: {year_strat_ret*100:>7.2f}% | Benchmark: {year_bh_ret*100:>7.2f}% | Sharpe: {sharpe_ratio:>5.2f}")
        print(f"         Longs: {year_long_trades} | Shorts: {year_short_trades} | Win Rate: {year_win_rate:.1f}%")
    
    # Final metrics
    oos_dates = pd.to_datetime(oos_dates)
    oos_log_returns = np.array(oos_log_returns)
    oos_positions = np.array(oos_positions)
    
    strat_returns = oos_positions * oos_log_returns
    eq_strat = np.exp(np.cumsum(strat_returns))
    eq_bh = np.exp(np.cumsum(oos_log_returns))
    
    running_max = np.maximum.accumulate(eq_strat)
    drawdown = (eq_strat - running_max) / running_max
    max_dd = np.min(drawdown) * 100
    win_rate = (trades_won / total_trades * 100) if total_trades > 0 else 0
    
    mean_strat = np.mean(strat_returns)
    std_strat = np.std(strat_returns)
    final_sharpe = (mean_strat / std_strat) * np.sqrt(8760) if std_strat > 0 else 0.0
    
    mean_bh = np.mean(oos_log_returns)
    std_bh = np.std(oos_log_returns)
    final_bh_sharpe = (mean_bh / std_bh) * np.sqrt(8760) if std_bh > 0 else 0.0
    
    # Total long/short counts
    total_longs = sum(w['long_trades'] for w in window_results)
    total_shorts = sum(w['short_trades'] for w in window_results)
    
    print(f"\n   FINAL OOS METRICS:")
    print(f"   Total Return:      {(eq_strat[-1] - 1) * 100:>8.2f}%")
    print(f"   Buy & Hold:        {(eq_bh[-1] - 1) * 100:>8.2f}%")
    print(f"   Strategy Sharpe:   {final_sharpe:>8.2f}")
    print(f"   Benchmark Sharpe:  {final_bh_sharpe:>8.2f}")
    print(f"   Maximum Drawdown:  {max_dd:>8.2f}%")
    print(f"   Total Trades:      {total_trades} (Long: {total_longs}, Short: {total_shorts})")
    print(f"   Win Rate:          {win_rate:>8.2f}%")
    
    # Plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(oos_dates, eq_bh, color='gray', alpha=0.5, label=f'Buy & Hold (Sharpe: {final_bh_sharpe:.2f})')
    ax1.plot(oos_dates, eq_strat, color='#00D4FF', linewidth=2, label=f'Walk-Forward Strategy (Sharpe: {final_sharpe:.2f})')
    ax1.set_title("True Out-of-Sample Walk-Forward Backtest", fontsize=16, fontweight='bold', color='white')
    ax1.set_ylabel("Cumulative Multiplier", fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(color='white', alpha=0.1, linestyle='--')
    
    ax2.fill_between(oos_dates, drawdown * 100, 0, color='red', alpha=0.4)
    ax2.plot(oos_dates, drawdown * 100, color='red', linewidth=1)
    ax2.set_title("Strategy Drawdown (%)", fontsize=14, color='white')
    ax2.set_ylabel("Drawdown %", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(color='white', alpha=0.1, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/OOS_backtest_results.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"   ✅ Saved {RESULTS_DIR}/OOS_backtest_results.png")
    
    return {
        'total_return': (eq_strat[-1] - 1) * 100,
        'bh_return': (eq_bh[-1] - 1) * 100,
        'sharpe': final_sharpe,
        'bh_sharpe': final_bh_sharpe,
        'max_dd': max_dd,
        'total_trades': total_trades,
        'total_longs': total_longs,
        'total_shorts': total_shorts,
        'win_rate': win_rate,
        'windows': window_results,
    }


if __name__ == '__main__':
    print("🚀 Exporting all pipeline results as images...")
    print("   Results will be saved to the 'results/' directory.\n")
    
    # 1. Pattern Miner + MCPT
    pip_miner, log_data = export_pip_pattern_miner()
    
    # 2. Cluster archetype grid
    export_cluster_grid(pip_miner, log_data)
    
    # 3. In-Sample Backtest
    is_metrics = export_is_backtest()
    
    # 4. Long/Short trigger examples
    export_long_short_examples()
    
    # 5. OOS Backtest
    oos_metrics = export_oos_backtest()
    
    print("\n" + "="*60)
    print("🎉 ALL RESULTS EXPORTED SUCCESSFULLY!")
    print("="*60)
    print(f"\nGenerated files in {RESULTS_DIR}/:")
    for f in os.listdir(RESULTS_DIR):
        print(f"   📄 {f}")
