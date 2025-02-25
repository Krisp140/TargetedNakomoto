from flask import Flask, request, jsonify, send_file
import numpy as np
import matplotlib
from flask_cors import CORS
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import requests
import pandas as pd
app = Flask(__name__)
CORS(app)
class Epoch:
    def __init__(self):
        self.rewards = []  # To store the rewards of each block within an epoch
        self.hashrates = [] # To store the hashrates of each block within an epoch
        self.ceil = None
        self.floor = None
        self.difficulty = None
        self.elapsed_time = None # T_n

    def median_block_reward(self):
        sorted_rewards = sorted(self.rewards)
        return np.median(sorted_rewards) 

    def average_hashrate(self):
        sorted_hashrates = sorted(self.hashrates) 
        return np.mean(sorted_hashrates)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_hashrate(self, n):
        self.hashrates.append(n)


class Blockchain:
    def __init__(self, tau, gamma, upper_bound, lower_bound, initial_difficulty = 1e12):
        self.tau = tau
        self.gamma = gamma
        self.epochs = [Epoch()]
        self.epochs[-1].difficulty = initial_difficulty
        self.DT = 646325930.8894218  # Puzzle difficulty/blockchain growth rate
        self.DT_N_UB = upper_bound
        self.DT_N_LB = lower_bound

    def adjust_reward(self, reward, epoch_idx):
        epoch = self.epochs[epoch_idx]
        if epoch.ceil is not None:
            reward = min(reward, epoch.ceil)
        if epoch.floor is not None:
            reward = max(reward, epoch.floor)
        epoch.add_reward(reward)
        return reward
    
    def adjust_hashrate(self, hashrate, epoch_idx, e_current, P_current, efficiency_current, electricity_cost_current):
        alpha1 = 10.9076
        alpha6 = 0.0130
        alpha7 = 0.6412
        alpha8 = -3.9930

        epoch = self.epochs[epoch_idx]
        log_eP_current = np.log(1+ (e_current * P_current))  # avoid log(0)

        log_eff_current = np.log(1+efficiency_current)
        log_electricity_cost_current = np.log(1+electricity_cost_current)
        # Put it all together:
        # bN_{t+1} = α1 + α2*log(eP_t) + α3*log(1+∆T_t) - α4*log(c_t) + α5*log(eP_{t-1})
        predicted_log_hashrate = (
            alpha1 
            + alpha6 * log_eP_current
            + alpha7 * log_eff_current
            + alpha8 * log_electricity_cost_current
        )

        # Exponentiate to get N_{t+1}
        new_hashrate = np.exp(predicted_log_hashrate)      
        epoch.add_hashrate(new_hashrate)
        print(f"Updateing Hashrate from {hashrate} to: {new_hashrate}")
        return new_hashrate
    

    def adjust_hashrate_with_changes(self, hashrate, epoch_idx, e_current, e_past, e_past_2, P_current, P_past, P_past_2):
        alpha1 = 0.0233
        alpha2 = 0.1277
        alpha5 = 0.0753

        epoch = self.epochs[epoch_idx]
        log_EP_current = np.log(e_current * P_current + 1e12)
        log_EP_past = np.log(e_past * P_past + 1e12)
        log_EP_past2 = np.log(e_past_2 * P_past_2 + 1e12)

        predicted_log_hashrate_change = (
            alpha1
            + alpha2 * (log_EP_current - log_EP_past)
            + alpha5 * (log_EP_past - log_EP_past2)
        )
        new_hashrate_change = np.exp(predicted_log_hashrate_change)
        new_hashrate = hashrate + new_hashrate_change
        epoch.add_hashrate(new_hashrate)
        return new_hashrate
    
    def end_of_epoch(self):
        last_epoch = self.epochs[-1]
        P_B_median = last_epoch.median_block_reward()
        N_n = last_epoch.average_hashrate()
        new_epoch = Epoch()

        BLOCKS_PER_EPOCH = 14
        T_STAR = BLOCKS_PER_EPOCH * 10 * 60 # 1,209,600 for 2016 blocks
        D_n = last_epoch.difficulty
    

        T_n = (BLOCKS_PER_EPOCH * D_n * 2**32) / N_n
        last_epoch.elapsed_time = T_n

        T_hat = max(T_STAR /4, min(T_n, 4 *T_STAR))
        D_next = D_n * (T_STAR / T_hat)
        new_epoch.difficulty = D_next
        print("\nNEW EPOCH\n")
        print(f"Updating difficulty from {D_n} => {D_next}")
        print("Median Block Reward: " + str(P_B_median) + '\n')
        #print("Current DT: " + str(self.DT))

         # Case 1: Hashrate too high - need to decrease rewards
        if self.DT > self.DT_N_UB:
            print("Case 1: Hashrate above upper bound")
            new_epoch.ceil = (self.tau) * P_B_median
            new_epoch.floor = None  # Clear any existing floor

       
        # Case 2 & 4: Hashrate within bounds - gradual adjustment
        elif self.DT_N_LB < self.DT < self.DT_N_UB:
            
            # Case 2: Previous epoch had a ceiling
            if last_epoch.ceil is not None:
                if last_epoch.ceil >= self.DT:
                    print("Case 2: Hashrate in bounds and ceiling is being relaxed")
                    # Gradually relax the ceiling
                    new_epoch.ceil =  (1+self.gamma) * last_epoch.ceil
                    print("Old ceiling: " + str(last_epoch.ceil))
                    print("Ceiling relaxed: " + str(new_epoch.ceil))
                else:
                    # Remove ceiling if it's no longer needed
                    new_epoch.ceil = None
            
            # Case 4: Previous epoch had a floor
            elif last_epoch.floor is not None:
                if last_epoch.floor <= self.DT:
                    print("Case 4: Hashrate in bound and floor is being relaxed")
                    # Gradually relax the floor
                    new_epoch.ceil = (self.gamma) * last_epoch.floor
                    print("Old floor: " + str(last_epoch.floor))
                    print("Floor relaxed: " + str(new_epoch.floor))
                else:
                    # Remove floor if it's no longer needed
                    new_epoch.floor = None
            else:
                new_epoch.floor = last_epoch.floor
                new_epoch.ceil = last_epoch.ceil

         # Case 3: Hashrate too low - need to increase rewards
        elif self.DT < self.DT_N_LB:
            print("Case 3: Hashrate below lower bound")
            new_epoch.floor =  (1+(1-self.tau)) * P_B_median
            new_epoch.ceil = None  # Clear any existing ceiling


        self.epochs.append(new_epoch)

def simulate_blockchain(initial_conditions, target_bounds, tau, gamma, time_paths):
    blockchain = Blockchain(
        tau=tau,
        gamma = gamma,
        upper_bound=target_bounds['upper_bound'],
        lower_bound=target_bounds['lower_bound']
    )

    N = [float(initial_conditions['hashrate'])]
    P = [float(initial_conditions['block_reward'])]
    e_path = time_paths['exchange_rate']
    efficiency_path = time_paths['efficiency']
    electricity_cost_path = time_paths['electricity_cost']

    for t in range(1, len(e_path)):
        current_N = N[-1]
        current_P = P[-1]
        #blockchain.DT = (e_path[t] * current_P) / (c_path[t])

        # Adjust reward within the epoch
        adjusted_reward = blockchain.adjust_reward(current_P, len(blockchain.epochs) - 1)
        #print("adjusted reward: " + str(adjusted_reward))
        P.append(adjusted_reward)
        new_N = blockchain.adjust_hashrate(current_N, len(blockchain.epochs)-1, e_path[t], current_P, efficiency_path[t], electricity_cost_path[t])
        N.append(new_N)
        blockchain.DT = new_N

        # End epoch and adjust policy
        if t % 14 == 0:  # Assuming 14 blocks per epoch for simplicity, going to change this to actual once real data is available
            blockchain.end_of_epoch()

    return {
        "time": list(range(len(e_path))),
        "hashrate": N,
        "block_rewards": P,
        "epochs": len(blockchain.epochs),
    }

@app.route('/')
def home():
    return """
    <h1>Welcome to the Hashrate Control API</h1>
    <p>Use the <code>/simulate</code> endpoint to run simulations.</p>
    <p>Use the <code>/visualize/&lt;metric&gt;</code> endpoint to generate plots for metrics like <code>hashrate</code>.</p>
    """
# API endpoint
@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # Parse input JSON
        data = request.json
        initial_conditions = data['initial_conditions']
        target_bounds = data['target_bounds']
        tau = data['control_params']['tau']
        time_paths = data['time_paths']
        
        # Run simulation
        results = simulate_blockchain(
            initial_conditions=initial_conditions,
            target_bounds=target_bounds,
            tau=tau,
            time_paths=time_paths
        )
        
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    

# Visualization endpoint
@app.route('/visualize/<metric>', methods=['POST'])
def visualize(metric):
    try:
        # Parse input JSON
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data received"}), 400
        
        # Validate required fields
        required_fields = ['initial_conditions', 'target_bounds', 'control_params', 'time_paths']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        initial_conditions = data['initial_conditions']
        target_bounds = data['target_bounds']
        tau = data['control_params']['tau']
        gamma = data['control_params']['gamma']
        time_paths = data['time_paths']

        # Load actual hashrate data from the CSV
        df = pd.read_csv('data/bitcoin_mining_data_daily.csv')
        actual_hashrate = df['hashrate'].tolist()

        # Run simulation
        results = simulate_blockchain(
            initial_conditions=initial_conditions,
            target_bounds=target_bounds,
            tau=tau,
            gamma=gamma,
            time_paths=time_paths
        )
        
        # Modify this section to handle different metrics
        if metric == 'hashrate':
            metric_data = results.get('hashrate')
        elif metric == 'mining_cost':
            metric_data = data['time_paths']['mining_cost']
        else:
            return jsonify({"success": False, "error": f"Invalid metric: {metric}"}), 400
            
        if metric_data is None:
            return jsonify({"success": False, "error": f"No data available for metric: {metric}"}), 400
        
        # Generate plot
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 6))

        # Define the slice you want to show (e.g., blocks 1000 to 2000)
        start_block = 0  # Adjust these values as needed
        end_block = 400    # Adjust these values as needed
        # Plot sliced data
        plt.plot(results["time"], 
                metric_data, 
                label=metric.capitalize(), 
                marker='o')
        # Plot actual hashrate (sliced)
        actual_slice = min(len(actual_hashrate), end_block)
        plt.plot(results["time"], 
                actual_hashrate, 
                label='Actual Hashrate',
                marker='x',
                alpha=0.6,
                linestyle='--',
                color='red')
        # Add target bounds as horizontal lines
        if metric == 'hashrate':  # Only show bounds for hashrate metric
            plt.axhline(y=target_bounds['upper_bound'], color='r', linestyle='--', label='Upper Bound')
            plt.axhline(y=target_bounds['lower_bound'], color='g', linestyle='--', label='Lower Bound')
        
        # Add block numbers to x-axis
        plt.xlabel("Day Number")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} Over Last Year (Day {start_block} to {end_block})")
        plt.legend()
        plt.grid(True)

        # Save plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return send_file(img, mimetype='image/png')
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')