import numpy as np
import pandas as pd

def block_bootstrap(data_array, block_size, num_samples, seed=None):
    rng = np.random.default_rng(seed)
        
    n = len(data_array)
    if n == 0:
        return np.array([])
        
    # Safely limit block_size to data array length
    block_size = min(block_size, n)
    num_blocks = int(np.ceil(num_samples / block_size))
    
    max_start_idx = max(0, n - block_size)
    start_indices = rng.integers(0, max_start_idx + 1, size=num_blocks)
    
    sampled = []
    for start_idx in start_indices:
        block = data_array[start_idx : start_idx + block_size]
        sampled.extend(block)
        
    return np.array(sampled[:num_samples])

def simulate_equity_paths(trades_df, num_simulations=10000, initial_capital=200, block_size=3, seed=None):
    if trades_df.empty:
        return np.array([[initial_capital] * num_simulations])
    
    pnls = trades_df['pnl'].values
    n_trades = len(pnls)
    
    rng = np.random.default_rng(seed)
        
    block_size = min(block_size, n_trades)
    max_start = max(0, n_trades - block_size)
    num_blocks = int(np.ceil(n_trades / block_size))
    
    # Vectorized path generation
    starts = rng.integers(0, max_start + 1, size=(num_simulations, num_blocks))
    
    sampled = np.zeros((num_simulations, n_trades))
    for b in range(num_blocks):
        start = starts[:, b]
        for j in range(block_size):
            col = b * block_size + j
            if col < n_trades:
                sampled[:, col] = pnls[start + j]
                
    paths = np.column_stack([np.full(num_simulations, initial_capital), initial_capital + np.cumsum(sampled, axis=1)])
    return paths.T

def calculate_risk_metrics(paths_array, ruin_level=0):
    num_sims = paths_array.shape[1]
    
    # Ruin probability
    min_capitals = np.min(paths_array, axis=0)
    ruined = np.sum(min_capitals <= ruin_level)
    prob_ruin = ruined / num_sims if num_sims > 0 else 0
    
    # Drawdowns
    # peak to trough
    running_max = np.maximum.accumulate(paths_array, axis=0)
    drawdowns = paths_array - running_max
    
    worst_drawdowns_per_sim = np.min(drawdowns, axis=0)
    expected_worst_dd = np.mean(worst_drawdowns_per_sim)
    
    return {
        "probability_of_ruin": prob_ruin,
        "expected_worst_drawdown": expected_worst_dd,
        "median_final_capital": np.median(paths_array[-1, :]),
        "mean_final_capital": np.mean(paths_array[-1, :]),
    }
