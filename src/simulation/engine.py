import numpy as np
import pandas as pd
from src.models.blockchain import Blockchain, Epoch

def run_simulation(params):
    """
    Run a blockchain simulation with the given parameters
    
    Args:
        params (dict): Dictionary containing simulation parameters:
            - tau: Adjustment rate parameter
            - gamma: Mining difficulty parameter  
            - upper_bound: Upper hashrate bound
            - lower_bound: Lower hashrate bound
            - initial_hashrate: Initial hashrate value
            - initial_block_reward: Initial block reward value
            - time_paths: Dictionary of time series data
            - time_steps: Number of time steps to simulate
            
    Returns:
        dict: Simulation results
    """
    # Extract parameters
    tau = params.get('tau', 0.1)
    gamma = params.get('gamma', 0.1)
    upper_bound = params.get('upper_bound', 1e12)
    lower_bound = params.get('lower_bound', 1e11)
    initial_hashrate = params.get('initial_hashrate', 1e11)
    initial_block_reward = params.get('initial_block_reward', 6.25)
    time_paths = params.get('time_paths', {})
    time_steps = params.get('time_steps', 100)
    
    # Initialize blockchain
    blockchain = Blockchain(
        tau=tau,
        gamma=gamma,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        initial_difficulty=1e12
    )
    
    # Initialize results arrays
    N = [float(initial_hashrate)]
    P = [float(initial_block_reward)]
    
    # Get time paths
    e_path = time_paths.get('exchange_rate', [50000] * time_steps)
    efficiency_path = time_paths.get('efficiency', [0.001] * time_steps)
    electricity_cost_path = time_paths.get('electricity_cost', [0.05] * time_steps)
    
    # Run simulation
    for t in range(1, time_steps):
        current_N = N[-1]
        current_P = P[-1]
        
        # Adjust reward within the epoch
        adjusted_reward = blockchain.adjust_reward(current_P, len(blockchain.epochs) - 1)
        P.append(adjusted_reward)
        
        # Calculate new hashrate
        new_N = blockchain.adjust_hashrate(
            current_N,
            len(blockchain.epochs) - 1,
            e_path[t] if t < len(e_path) else e_path[-1],
            current_P,
            efficiency_path[t] if t < len(efficiency_path) else efficiency_path[-1],
            electricity_cost_path[t] if t < len(electricity_cost_path) else electricity_cost_path[-1]
        )
        N.append(new_N)
        blockchain.DT = new_N
        
        # Check for epoch end
        if t % 14 == 0:  # Assuming 14 blocks per epoch
            blockchain.end_of_epoch()
    
    # Calculate statistics
    hashrate_volatility = np.std(N) / np.mean(N) * 100
    reward_volatility = np.std(P) / np.mean(P) * 100
    time_in_bounds = sum(1 for n in N if lower_bound <= n <= upper_bound)
    bound_adherence = (time_in_bounds / len(N)) * 100
    
    return {
        "time": list(range(time_steps)),
        "hashrate": N,
        "block_reward": P,
        "exchange_rate": e_path[:time_steps],
        "efficiency": efficiency_path[:time_steps],
        "electricity_cost": electricity_cost_path[:time_steps],
        "epochs": len(blockchain.epochs),
        "stats": {
            "hashrate_volatility": hashrate_volatility,
            "reward_volatility": reward_volatility,
            "bound_adherence": bound_adherence
        }
    }