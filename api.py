from flask import Flask, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

def simulate_hashrate_control(initial_conditions, target_bounds, control_params, time_paths):
    # Unpack inputs
    N = [initial_conditions['hashrate']]
    ceiling = [initial_conditions['block_reward']]
    floor = [initial_conditions['block_reward']]
    P = [initial_conditions['block_reward']]
    e_path = time_paths['exchange_rate']
    c_path = time_paths['mining_cost']
    N_UB, N_LB = target_bounds['upper_bound'], target_bounds['lower_bound']
    tau, gamma = control_params['tau'], control_params['gamma']
    
    # Run simulation
    for t in range(1, len(e_path)):
        current_N = N[-1]
        current_P = P[-1]
        current_e = e_path[t]
        current_c = c_path[t]

        # Apply HCA rules
        if current_N > N_UB:
            ceiling.append(ceiling[-1] * (1 - tau))
            floor.append(floor[-1])
            P.append(min(current_P, ceiling[-1]))
        elif current_N < N_LB:
            floor.append(floor[-1] * (1 + tau))
            ceiling.append(ceiling[-1])
            P.append(max(current_P, floor[-1]))
        else:
            # Within bounds, relax ceilings/floors
            if ceiling[-1] > current_P:
                ceiling.append((1 - tau) * ceiling[-1])
            else:
                ceiling.append(ceiling[-1])
            if floor[-1] < current_P:
                floor.append((1 + tau) * floor[-1])
            else:
                floor.append(floor[-1])
            P.append(current_P)

        # Update hashrate
        new_N = (current_e * P[-1]) / (current_c + gamma)
        N.append(new_N)

    # Package results
    return {
        "time": list(range(len(e_path))),
        "hashrate": N,
        "block_reward_ceiling": ceiling,
        "block_reward_floor": floor,
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
        control_params = data['control_params']
        time_paths = data['time_paths']
        
        # Run simulation
        results = simulate_hashrate_control(
            initial_conditions,
            target_bounds,
            control_params,
            time_paths
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
        
        # Run simulation
        results = simulate_hashrate_control(
            data['initial_conditions'],
            data['target_bounds'],
            data['control_params'],
            data['time_paths']
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