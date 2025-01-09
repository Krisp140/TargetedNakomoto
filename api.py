from flask import Flask, request, jsonify, send_file
import numpy as np
import matplotlib
from flask_cors import CORS
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
CORS(app)
class Epoch:
    def __init__(self):
        self.rewards = []  # To store the rewards of each block within an epoch
        self.ceil = None
        self.floor = None

    def median_block_reward(self):
        sorted_rewards = sorted(self.rewards)
        return np.median(sorted_rewards)  

    def add_reward(self, reward):
        self.rewards.append(reward)


class Blockchain:
    def __init__(self, tau, upper_bound, lower_bound):
        self.tau = tau
        self.epochs = [Epoch()]
        self.DT = None  # Puzzle difficulty/blockchain growth rate
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

    def end_of_epoch(self):
        last_epoch = self.epochs[-1]
        P_B_median = last_epoch.median_block_reward()
        new_epoch = Epoch()

        if self.DT > self.DT_N_UB:
            new_epoch.ceil = self.tau * P_B_median
        elif self.DT < self.DT_N_LB:
            new_epoch.floor = (1 + self.tau) * P_B_median
        else:
            if last_epoch.ceil and last_epoch.ceil <= P_B_median:
                new_epoch.ceil = (1 - self.tau) * last_epoch.ceil
            elif last_epoch.floor and last_epoch.floor >= P_B_median:
                new_epoch.floor = (1 + self.tau) * last_epoch.floor

        self.epochs.append(new_epoch)

def simulate_blockchain(initial_conditions, target_bounds, tau, time_paths):
    blockchain = Blockchain(
        tau=tau,
        upper_bound=target_bounds['upper_bound'],
        lower_bound=target_bounds['lower_bound']
    )

    N = [float(initial_conditions['hashrate'])]
    P = [float(initial_conditions['block_reward'])]
    e_path = time_paths['exchange_rate']
    c_path = time_paths['mining_cost']

    for t in range(1, len(e_path)):
        current_N = N[-1]
        current_P = P[-1]
        blockchain.DT = (e_path[t] * current_P) / (c_path[t])

        # Adjust reward within the epoch
        adjusted_reward = blockchain.adjust_reward(current_P, len(blockchain.epochs) - 1)
        P.append(adjusted_reward)

        # Update hashrate
        new_N = (e_path[t] * adjusted_reward) / (c_path[t])
        N.append(new_N)

        # End epoch and adjust policy
        if t % 10 == 0:  # Assuming 10 blocks per epoch for simplicity, going to change this to actual once real data is available
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
        time_paths = data['time_paths']

        # Run simulation
        results = simulate_blockchain(
            initial_conditions=initial_conditions,
            target_bounds=target_bounds,
            tau=tau,
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
        plt.plot(results["time"], metric_data, label=metric.capitalize(), marker='o')
        plt.xlabel("Time")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} Over Time")
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