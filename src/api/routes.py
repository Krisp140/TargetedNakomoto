from flask import Flask, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import pandas as pd
from flask_cors import CORS
from src.models.blockchain import Blockchain, Epoch

app = Flask(__name__)
CORS(app)

def simulate_blockchain(initial_conditions, target_bounds, tau, gamma, time_paths):
    blockchain = Blockchain(
        tau=tau,
        gamma=gamma,
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

        # Adjust reward within the epoch
        adjusted_reward = blockchain.adjust_reward(current_P, len(blockchain.epochs) - 1)
        P.append(adjusted_reward)
        new_N = blockchain.adjust_hashrate(current_N, len(blockchain.epochs)-1, e_path[t], current_P, efficiency_path[t], electricity_cost_path[t])
        N.append(new_N)
        blockchain.DT = new_N

        # End epoch and adjust policy
        if t % 14 == 0:  # Assuming 14 blocks per epoch for simplicity
            blockchain.end_of_epoch()

    return {
        "time": list(range(len(e_path))),
        "hashrate": N,
        "block_reward": P,
        "exchange_rate": e_path,
        "epochs": len(blockchain.epochs)
    }

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Nakamoto Blockchain Simulation API"})

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        result = simulate_blockchain(
            data.get('initial_conditions', {}),
            data.get('target_bounds', {}),
            data.get('control_params', {}).get('tau', 0.5),
            data.get('control_params', {}).get('gamma', 0.1),
            data.get('time_paths', {})
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visualize/<metric>', methods=['POST'])
def visualize(metric):
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        result = simulate_blockchain(
            data.get('initial_conditions', {}),
            data.get('target_bounds', {}),
            data.get('control_params', {}).get('tau', 0.5),
            data.get('control_params', {}).get('gamma', 0.1),
            data.get('time_paths', {})
        )
        
        # Create the visualization
        plt.figure(figsize=(12, 6))
        
        if metric == 'hashrate':
            plt.plot(result['time'], result['hashrate'], label='Hashrate', color='blue')
            plt.axhline(y=data['target_bounds']['upper_bound'], color='r', linestyle='--', label='Upper Bound')
            plt.axhline(y=data['target_bounds']['lower_bound'], color='g', linestyle='--', label='Lower Bound')
            plt.title('Hashrate over Time')
            plt.ylabel('Hashrate (H/s)')
            plt.yscale('log')
        elif metric == 'block_reward':
            plt.plot(result['time'], result['block_reward'], label='Block Reward', color='orange')
            plt.title('Block Reward over Time')
            plt.ylabel('Block Reward (BTC)')
            plt.yscale('log')
        elif metric == 'exchange_rate':
            plt.plot(result['time'], result['exchange_rate'], label='Exchange Rate', color='green')
            plt.title('Exchange Rate over Time')
            plt.ylabel('Price (USD)')
        else:
            return jsonify({"error": "Invalid metric"}), 400
        
        plt.xlabel('Time Steps')
        plt.grid(True)
        plt.legend()
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Return the image as a response
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')