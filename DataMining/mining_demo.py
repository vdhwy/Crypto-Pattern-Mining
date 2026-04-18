import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the PIPPatternMiner class from your original file.
from pip_pattern_miner import PIPPatternMiner

def main():
    print("Loading BTC dataset...")
    path = 'BTCUSDT_1h.csv'
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Could not find {path}. Please ensure it is in the same directory.")
        return

    # Format the date index
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    # Keep a raw copy of the data for mplfinance candlestick plotting
    plot_data = data.copy()
    plot_data = plot_data[plot_data.index < '01-01-2025']

    # Log-transform the data for the actual model training
    data = np.log(data)
    data = data[data.index < '01-01-2025']
    arr = data['close'].to_numpy()

    print("Initializing and training the PIP Pattern Miner...")
    pip_miner = PIPPatternMiner(n_pips=5, lookback=24, hold_period=6)
    pip_miner.train(arr, n_reps=-1)

    num_clusters = len(pip_miner._pip_clusters)
    print(f"\nTraining complete! Found {num_clusters} unique pattern clusters.")

    if num_clusters == 0:
        print("No clusters found to visualize.")
        return

    # --- NEW: Folder Creation Logic ---
    output_folder = "cluster_visualizations"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created output folder: '{output_folder}/'")

    # Save all clusters (or change to min(5, num_clusters) if you only want a few)
    for i in range(num_clusters):
        print(f"Saving plot for Cluster {i}...")
        
        # --- NEW: Intercept the plot and save it to a file ---
        # We temporarily replace matplotlib's show() function with our own save function
        original_show = plt.show 
        
        def save_and_close(*args, **kwargs):
            file_path = os.path.join(output_folder, f"cluster_{i:02d}.png")
            plt.savefig(file_path, bbox_inches='tight', dpi=150) # Save high-res PNG
            plt.close() # Close the figure to free up memory
            
        plt.show = save_and_close # Apply the override
        
        try:
            # grid_size=3 creates a 3x3 grid (up to 9 examples per cluster)
            pip_miner.plot_cluster_examples(plot_data, cluster_i=i, grid_size=3)
        finally:
            plt.show = original_show # Always restore the original show() function safely

    print(f"\nSuccess! All cluster plots have been saved inside the '{output_folder}' folder.")

if __name__ == '__main__':
    main()