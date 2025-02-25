import streamlit as st
import numpy as np
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from api import Blockchain, Epoch  # Reusing your existing classes
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Hashrate Control Simulation", layout="wide",page_icon="💻")

# Create columns for title and logo
title_col, logo_col = st.columns([3, 1])

with title_col:
    st.title("Hashrate Control Simulation")

with logo_col:
    # Open and crop the image
    image = Image.open('btc2.png')
    # Get current dimensions
    width, height = image.size
    # Crop from the bottom (adjust the 0.8 factor to crop more or less)
    cropped_image = image.crop((0, int(height * 0.2), width, int(height * 0.8)))
    st.image(cropped_image, width=200)

# Sample data for simulation
data = pd.read_csv('data/bitcoin_mining_data_daily.csv')
hashrate = data['hashrate'].tolist()
e_path = data['price'].tolist()  # Exchange rate path
time_steps = len(e_path)
efficiency_path = data['mining_cost'].tolist()     # Efficiency path
electricity_cost_path = data['electricity_price'].tolist()  # Electricity cost path
block_reward_path = data['block_reward'].tolist()
fee_path = data['fees'].tolist()

# Sidebar for parameters
st.sidebar.header("Control Parameters")
tau = st.sidebar.number_input("Tau", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
gamma = st.sidebar.number_input("Gamma", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

st.sidebar.header("Target Bounds")
upper_bound_percent = st.sidebar.slider("Upper Bound (%)", 
    min_value=0.0, 
    max_value=200.0, 
    value=120.0, 
    step=1.0,
    help="Upper bound as a percentage of the mean hashrate"
)

lower_bound_percent = st.sidebar.slider("Lower Bound (%)", 
    min_value=0.0, 
    max_value=200.0, 
    value=80.0, 
    step=1.0,
    help="Lower bound as a percentage of the mean hashrate"
)

# Convert percentages to actual bounds based on initial block reward
mean_hashrate = float(data['hashrate'].mean())
upper_bound = (upper_bound_percent / 100.0) * mean_hashrate
lower_bound = (lower_bound_percent / 100.0) * mean_hashrate

# Display the actual values
st.sidebar.text(f"Actual Upper Bound: {upper_bound:.2e}")
st.sidebar.text(f"Actual Lower Bound: {lower_bound:.2e}")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Simulation Results")
    if st.button("Run Simulation"):
        # Initialize blockchain with parameters
        blockchain = Blockchain(
            tau=tau,
            gamma=gamma,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            initial_difficulty=1e12
        )
    
            
        time_paths = {
            'exchange_rate': e_path,
            'efficiency': efficiency_path,
            'electricity_cost': electricity_cost_path
        }
        # Run simulation
        initial_conditions = {
            'hashrate': hashrate[0],
            'block_reward': block_reward_path[0]
        }
        
        # Run simulation
        N = [float(initial_conditions['hashrate'])]
        P = [float(initial_conditions['block_reward'])]
        
        for t in range(time_steps):

            current_N = N[-1]
            current_P = P[-1]

            # Adjust reward within the epoch
            adjusted_reward = blockchain.adjust_reward(current_P, len(blockchain.epochs) - 1)
            P.append(adjusted_reward)
            
            # Calculate new hashrate
            new_N = blockchain.adjust_hashrate(
                current_N,
                len(blockchain.epochs) - 1,
                e_path[t],
                current_P,
                efficiency_path[t],
                electricity_cost_path[t]
            )
            N.append(new_N)
            blockchain.DT = new_N
            current_N = new_N
            current_P = adjusted_reward
            
            # Check for epoch end (every 14 blocks in your case)
            if t > 0 and t % 14 == 0:
                blockchain.end_of_epoch()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot hashrate
        ax1.plot(N, label='Simulated Hashrate', color='blue')
        ax1.plot(hashrate[:time_steps], 
                label='Actual Hashrate',
                marker='x',
                alpha=0.6,
                linestyle='--',
                color='red')
        ax1.set_title('Hashrate over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Hashrate')
        ax1.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
        ax1.axhline(y=lower_bound, color='g', linestyle='--', label='Lower Bound')
        ax1.grid(True)
        ax1.set_yscale('log')
        ax1.legend()
        
        # Combine P and fee_path vertically
        # Add fees to block rewards for each timestep
        #P = [p + f for p, f in zip(P, fee_path[:len(P)])]
        # Plot block rewards
        ax2.plot(P, label='Block Reward', color='orange')
        ax2.plot(block_reward_path[:time_steps], 
                label='Actual Block Reward',
                marker='x',
                alpha=0.6,
                linestyle='--',
                color='blue')
        ax2.set_title('Miners Block Reward over Time')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Block Reward (log scale)')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display statistics
        with col2:
            st.header("Statistics")
            
            # Current Values
            st.subheader("Current Values")
            st.metric("Final Hashrate", f"{N[-1]:.2e} H/s")
            st.metric("Final Block Reward", f"{P[-1]:.2f}")
            st.metric("Number of Epochs", len(blockchain.epochs))
            
            # Volatility Metrics
            st.subheader("Volatility Metrics")
            hashrate_volatility = np.std(N) / np.mean(N) * 100
            reward_volatility = np.std(P) / np.mean(P) * 100
            st.metric("Hashrate Volatility: (std / mean)", f"{hashrate_volatility:.2f}%")
            #st.metric("Reward Volatility", f"{reward_volatility:.2f}%")
            
            # Performance Metrics
            st.subheader("Performance Metrics")
            time_in_bounds = sum(1 for n in N if lower_bound <= n <= upper_bound)
            bound_adherence = (time_in_bounds / len(N)) * 100
            st.metric("Time in Bounds", f"{bound_adherence:.1f}%")
            

            # Comparison with Actual Data
            #st.subheader("Model vs Actual")
            actual_data = np.array(hashrate[:len(N)])  # Convert to numpy array
            sim_data = np.array(N[1:])  # Convert to numpy array
            
            # Calculate correlation
            correlation = np.corrcoef(sim_data[:-1], actual_data[:-1])[0, 1]
            #st.metric("Correlation", f"{correlation:.3f}")
            
            # Calculate Mean Absolute Percentage Error
            mape = np.mean(np.abs((actual_data - sim_data) / actual_data)) * 100
            #st.metric("MAPE", f"{mape:.2f}%")
            
            # Calculate Root Mean Square Error
            rmse = np.sqrt(np.mean((actual_data - sim_data) ** 2))
            #st.metric("RMSE", f"{rmse:.2e}")
            
            # Directional Accuracy
            sim_changes = np.diff(sim_data) > 0
            actual_changes = np.diff(actual_data) > 0
            directional_accuracy = np.mean(sim_changes == actual_changes) * 100
            #st.metric("Directional Accuracy", f"{directional_accuracy:.1f}%")

            # Min/Max Values
            st.subheader("Extremes")
            st.metric("Max Hashrate", f"{max(N):.2e}")
            st.metric("Min Hashrate", f"{min(N):.2e}")
            st.metric("Hashrate Range", f"{(max(N) - min(N))/max(N)*100:.1f}%")


# Add new button for parameter sweep analysis
if st.button("Run Parameter Sweep Analysis"):
    st.write("Running parameter sweep analysis... This may take a few minutes.")
    
    # Define the range of parameters to test
    tau_range = np.linspace(0.1, 1.0, 10)
    gamma_range = np.linspace(0.1, 1.0, 10)
    
    # Create meshgrid for 3D plot
    TAU, GAMMA = np.meshgrid(tau_range, gamma_range)
    TIME_IN_BOUNDS = np.zeros_like(TAU)
    
    # Progress bar
    progress_bar = st.progress(0)
    total_iterations = len(tau_range) * len(gamma_range)
    current_iteration = 0
    
    # Evaluate each combination
    for i, tau_val in enumerate(tau_range):
        for j, gamma_val in enumerate(gamma_range):
            # Initialize blockchain with current parameters
            blockchain = Blockchain(
                tau=tau_val,
                gamma=gamma_val,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                initial_difficulty=1e12
            )
            
            # Run simulation
            N = [float(hashrate[0])]
            P = [float(block_reward_path[0])]
            
            for t in range(time_steps):
                current_N = N[-1]
                current_P = P[-1]
                
                # Adjust reward within the epoch
                adjusted_reward = blockchain.adjust_reward(current_P, len(blockchain.epochs) - 1)
                
                # Get current values from time paths
                e_current = e_path[t]
                efficiency_current = efficiency_path[t]
                electricity_cost_current = electricity_cost_path[t]
                
                # Update hashrate
                new_N = blockchain.adjust_hashrate(
                    current_N,
                    len(blockchain.epochs) - 1,
                    e_current,
                    current_P,
                    efficiency_current,
                    electricity_cost_current
                )
                N.append(new_N)
                P.append(adjusted_reward)
                blockchain.DT = new_N
                current_N = new_N
                current_P = adjusted_reward
                
                # Check for epoch end (every 14 blocks in your case)
                if t > 0 and t % 14 == 0:
                    blockchain.end_of_epoch()
            # Calculate time in bounds metric
            in_bounds = sum(1 for n in N if lower_bound <= n <= upper_bound)
            time_in_bounds_percent = (in_bounds / len(N)) * 100
            TIME_IN_BOUNDS[j, i] = time_in_bounds_percent
            
            # Update progress
            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)
    
    # Create 3D plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(TAU, GAMMA, TIME_IN_BOUNDS, cmap='viridis')
    
    # Customize plot
    ax.set_xlabel('Tau')
    ax.set_ylabel('Gamma')
    #ax.set_zlabel('Time in Bounds (%)')
    ax.set_title('Parameter Sweep Analysis')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, label='Time in Bounds (%)')
    
    # Display plot in Streamlit
    st.pyplot(fig)
    
    # Find optimal parameters
    max_idx = np.unravel_index(np.argmax(TIME_IN_BOUNDS), TIME_IN_BOUNDS.shape)
    optimal_tau = tau_range[max_idx[1]]
    optimal_gamma = gamma_range[max_idx[0]]
    max_time_in_bounds = TIME_IN_BOUNDS[max_idx]
    
    st.write(f"Optimal Parameters:")
    st.write(f"Tau: {optimal_tau:.2f}")
    st.write(f"Gamma: {optimal_gamma:.2f}")
    st.write(f"Maximum Time in Bounds: {max_time_in_bounds:.2f}%")