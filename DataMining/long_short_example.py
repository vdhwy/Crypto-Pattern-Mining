import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

from wf_pip_miner import WFPIPMiner
from perceptually_important import find_pips

def plot_pattern_on_ax(data_slice, n_pips, ax, title):
    """Helper to plot a single PIP pattern on a specific matplotlib axis."""
    plot_pip_x, plot_pip_y = find_pips(data_slice['close'].to_numpy(), n_pips, 3)
    
    idx = data_slice.index
    pip_lines = []
    colors = []
    
    for line_i in range(n_pips - 1):
        l0 = [(idx[plot_pip_x[line_i]], plot_pip_y[line_i]), 
              (idx[plot_pip_x[line_i + 1]], plot_pip_y[line_i + 1])]
        pip_lines.append(l0)
        colors.append('w') # White lines

    mpf.plot(data_slice, type='candle', alines=dict(alines=pip_lines, colors=colors), 
             ax=ax, style='charles', update_width_config=dict(candle_linewidth=1.75))
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_title(title, color='white', fontsize=10)

def main():
    print("Loading BTC dataset...")
    try:
        data = pd.read_csv('BTCUSDT_1h.csv') 
    except FileNotFoundError:
        print("Error: Could not find 'BTCUSDT_1h.csv'.")
        return

    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    plot_data = data.copy()
    
    data = np.log(data)
    arr = data['close'].to_numpy()

    N_PIPS = 5
    LOOKBACK = 24
    HOLD_PERIOD = 6
    NUM_EXAMPLES = 9 # We will find 4 Longs and 4 Shorts for a 2x4 grid

    print("Initializing Walk-Forward Miner...")
    wf_miner = WFPIPMiner(
        n_pips=N_PIPS, 
        lookback=LOOKBACK, 
        hold_period=HOLD_PERIOD, 
        train_size=24 * 365 * 2, 
        step_size=24 * 365 * 1
    )
    
    last_sig = 0.0
    long_slices = []
    short_slices = []

    print(f"Scanning for {NUM_EXAMPLES} Long and {NUM_EXAMPLES} Short examples...")

    for i in range(len(arr)):
        curr_sig = wf_miner.update_signal(arr, i)
        
        # Capture New Long Signal
        if curr_sig == 1.0 and last_sig != 1.0:
            if len(long_slices) < NUM_EXAMPLES:
                long_slices.append(plot_data.iloc[i - LOOKBACK + 1: i + 1])
                
        # Capture New Short Signal
        elif curr_sig == -1.0 and last_sig != -1.0:
            if len(short_slices) < NUM_EXAMPLES:
                short_slices.append(plot_data.iloc[i - LOOKBACK + 1: i + 1])

        last_sig = curr_sig
        
        # Stop early once we have enough examples of both
        if len(long_slices) == NUM_EXAMPLES and len(short_slices) == NUM_EXAMPLES:
            print("Found enough examples! Generating grid plot...")
            break

    # --- Create the Grid Plot ---
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, NUM_EXAMPLES, figsize=(16, 8))
    fig.suptitle("Walk-Forward Strategy: Long vs. Short Triggers", fontsize=18, fontweight='bold', color='white')

    # Plot Longs on the Top Row (Row 0)
    for i, data_slice in enumerate(long_slices):
        end_date = data_slice.index[-1].strftime('%Y-%m-%d %H:%00')
        plot_pattern_on_ax(data_slice, N_PIPS, axs[0, i], f"LONG Trigger\n{end_date}")

    # Plot Shorts on the Bottom Row (Row 1)
    for i, data_slice in enumerate(short_slices):
        end_date = data_slice.index[-1].strftime('%Y-%m-%d %H:%00')
        plot_pattern_on_ax(data_slice, N_PIPS, axs[1, i], f"SHORT Trigger\n{end_date}")

    # Format and display
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit the main title
    plt.show()

if __name__ == '__main__':
    main()