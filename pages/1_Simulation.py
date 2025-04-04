import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from src.models.blockchain import Blockchain
from src.simulation.engine import run_simulation

# Initialize session state variables if they don't exist
if 'run_optimal_sim' not in st.session_state:
    st.session_state.run_optimal_sim = False
if 'optimal_tau' not in st.session_state:
    st.session_state.optimal_tau = 0.0
if 'optimal_gamma' not in st.session_state:
    st.session_state.optimal_gamma = 0.0

# Function to set session state when button is clicked
def run_with_optimal_params():
    st.session_state.run_optimal_sim = True
    st.session_state.optimal_tau = optimal_tau
    st.session_state.optimal_gamma = optimal_gamma

st.set_page_config(page_title="Targeted Nakamoto Bitcoin Hashrate Control Simulator", layout="wide", page_icon="💻")

# Apply custom button styling throughout the app
st.markdown(
    """
    <style>
    div.stButton > button {
        font-size: 20px;
        font-weight: bold;
        height: auto;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 16px;
        padding-right: 16px;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        width: 100%;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        border-color: #45a049;
    }
    
    /* Make sidebar toggle button larger (especially on mobile) */
    button[kind="header"] {
        transform: scale(1.5) !important;
        margin: 10px !important;
    }
    
    /* Additional scaling for mobile devices */
    @media (max-width: 640px) {
        button[kind="header"] {
            transform: scale(2.0) !important;
            margin: 12px !important;
        }
    }

    section[data-testid="stSidebar"] .stNumberInput label, 
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] h1, 
    /* ... */
    {
        font-size: 1.2rem !important;
        font-weight: bold !important;
    }

    section[data-testid="stSidebar"] div.stNumberInput,
    section[data-testid="stSidebar"] div.stSlider {
        margin-bottom: 2rem !important;
        padding: 0.5rem !important;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Create columns for title and logo
title_col, logo_col = st.columns([3, 1])

# Title and subtitle
with title_col:
    st.title("Targeted Nakamoto Bitcoin Hashrate Control Simulator")
    st.markdown("*Created by Kristian Praizner & Daniel Aronoff*")
    st.markdown("A control theory approach to Bitcoin hashrate targeting")

with logo_col:
    # Open and crop the image
    image = Image.open('static/mit-logo.jpg')
    # Get current dimensions
    width, height = image.size
    # Crop from the bottom (adjust the 0.8 factor to crop more or less)
    #cropped_image = image.crop((0, int(height * 0.5), width, int(height * 0.5)))
    st.image(image)

# Sample data for simulation
data = pd.read_csv('data/merged_data.csv')
hashrate = data['HashRate'].tolist()
e_path = data['BTCPrice'].tolist()  # Exchange rate path
time_steps = len(e_path)
efficiency_path = data['Efficiency'].tolist()     # Efficiency path
electricity_cost_path = data['ElectricityPrice'].tolist()  # Electricity cost path
block_reward_path = data['BTC'].tolist()
#fee_path = data['fees'].tolist()

# Sidebar for parameters
st.sidebar.header("Control Parameters")
tau = st.sidebar.number_input("Tau", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
gamma = st.sidebar.number_input("Gamma", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

st.sidebar.header("Target Bounds")
# Calculate mean hashrate for default value
mean_hashrate = float(data['HashRate'].mean())

# Extract significant digits and exponent
sig_figs, exponent = "{:.3e}".format(mean_hashrate).split('e')
sig_figs = float(sig_figs)
exponent = int(exponent)

# Create input for significant figures only
sig_fig_input = st.sidebar.number_input(
    f"Target Hashrate (×10^{exponent})", 
    min_value=0.1, 
    max_value=9.99,
    value=sig_figs,
    format="%.3f",
    help="Enter the significant figures of the target hashrate"
)

# Convert back to full hashrate
target_hashrate = sig_fig_input * (10 ** exponent)

# Input field for range percentage with 10% as default
range_percent = st.sidebar.number_input(
    "Range (% of target hashrate)", 
    min_value=1.0, 
    max_value=100.0, 
    value=10.0, 
    step=1.0,
    help="Range as a percentage of the target hashrate"
)

# Calculate upper and lower bounds based on target and range
upper_bound = target_hashrate * (1 + range_percent / 100.0)
lower_bound = target_hashrate * (1 - range_percent / 100.0)

# Display the actual values
st.sidebar.text(f"Upper Bound: {upper_bound:.2e} H/s")
st.sidebar.text(f"Lower Bound: {lower_bound:.2e} H/s")

# Calculate estimated cost per block at current hashrate and efficiency
# Bitcoin block time is approximately 10 minutes
block_time_hours = 10/60  # 10 minutes in hours
current_hashrate = hashrate[-1]
current_efficiency = efficiency_path[-1]
current_electricity_cost = electricity_cost_path[-1]

# Formula: (Hashrate / Efficiency) * Electricity Cost * Block Time
# Calculate for actual hashrate
actual_energy_per_block = (current_hashrate/1e18 / current_efficiency) * block_time_hours
actual_cost_per_block = actual_energy_per_block * current_electricity_cost

st.sidebar.header("Electricity Cost Estimate")
st.sidebar.metric("Current Actual Cost per Block (Texas)", f"${actual_cost_per_block:.2f}")
st.sidebar.text("Based on current hashrate, efficiency, and electricity prices")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Simulation Results")
    if st.button("Run Simulation", key="run_standard_simulation"):
        # Prepare simulation parameters
        sim_params = {
            'tau': tau,
            'gamma': gamma,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'initial_hashrate': hashrate[0],
            'initial_block_reward': block_reward_path[0],
            'time_steps': time_steps,
            'time_paths': {
                'exchange_rate': e_path,
                'efficiency': efficiency_path,
                'electricity_cost': electricity_cost_path
            }
        }
        
        # Run simulation
        results = run_simulation(sim_params)
        
        # Calculate cost per block for final predicted hashrate
        block_time_hours = 10/60  # 10 minutes in hours
        final_predicted_hashrate = results['hashrate'][-1]
        final_actual_hashrate = hashrate[-1] if len(hashrate) > 0 else hashrate[0]
        final_efficiency = efficiency_path[-1] if len(efficiency_path) > 0 else efficiency_path[0]
        final_electricity_cost = electricity_cost_path[-1] if len(electricity_cost_path) > 0 else electricity_cost_path[0]
        
        # Formula: (Hashrate / Efficiency) * Electricity Cost * Block Time
        final_predicted_energy_per_block = (final_predicted_hashrate/1e18 / final_efficiency) * block_time_hours
        final_predicted_cost_per_block = final_predicted_energy_per_block * final_electricity_cost
        
        # Also calculate for actual hashrate
        final_actual_energy_per_block = (final_actual_hashrate/1e18 / final_efficiency) * block_time_hours
        final_actual_cost_per_block = final_actual_energy_per_block * final_electricity_cost
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot hashrate
        ax1.plot(results['hashrate'], label='Simulated Hashrate', color='blue')
        ax1.plot(hashrate[:time_steps], 
                label='Actual Hashrate',
                marker='x',
                alpha=0.6,
                linestyle='--',
                color='red')
        ax1.set_title('Actual vs Predicted Hashrate (1-1-2022 to 3-8-2025)')
        ax1.set_xlabel('Time Steps (days)')
        ax1.set_ylabel('Hashrate')
        ax1.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
        ax1.axhline(y=lower_bound, color='g', linestyle='--', label='Lower Bound')
        ax1.grid(True)
        ax1.set_yscale('log')
        ax1.legend()
        
        # Plot block rewards
        ax2.plot(results['block_reward'], label='Block Reward', color='orange')
        ax2.plot(block_reward_path[:time_steps], 
                label='Actual Block Reward',
                marker='x',
                alpha=0.6,
                linestyle='--',
                color='blue')
        ax2.set_title('Miners Block Reward over Time (1-1-2022 to 3-8-2025)')
        ax2.set_xlabel('Time Steps (days)')
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
            st.metric("Final Hashrate", f"{results['hashrate'][-1]:.2e} H/s")
            st.metric("Final Block Reward (per day)", f"{results['block_reward'][-1]:.2f} BTC")
            st.metric("Number of Epochs", results['epochs'])
            
            # Electricity Cost section
            st.subheader("Electricity Costs")
            st.metric("Predicted Cost per Block", f"${final_predicted_cost_per_block:.2f}")
            st.metric("Actual Cost per Block", f"${final_actual_cost_per_block:.2f}")
            
            # Volatility Metrics
            st.subheader("Volatility Metrics")
            st.metric("Hashrate Volatility: (std / mean)", f"{results['stats']['hashrate_volatility']:.2f}%")
            
            # Performance Metrics
            st.subheader("Performance Metrics")
            st.metric("Time in Bounds", f"{results['stats']['bound_adherence']:.1f}%")

# Add parameter sweep analysis functionality
if st.button("Run Parameter Sweep Analysis", key="run_parameter_sweep"):
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
    
    # Create a container for optimal simulation results
    optimal_sim_container = st.container()
    
    # Add button to run simulation with optimal parameters using session state
    def run_with_optimal_params():
        st.session_state.run_optimal_sim = True
        st.session_state.optimal_tau = optimal_tau
        st.session_state.optimal_gamma = optimal_gamma
    
    st.button("Run Simulation with Optimal Parameters", key="run_optimal_params", on_click=run_with_optimal_params)

# Check if we should run the simulation with optimal parameters
if st.session_state.get('run_optimal_sim', False):
    st.header("Simulation with Optimal Parameters")
    st.write(f"Running simulation with optimal parameters: Tau={st.session_state.optimal_tau:.2f}, Gamma={st.session_state.optimal_gamma:.2f}")
    
    # Create columns for plots and statistics, matching the original layout
    opt_col1, opt_col2 = st.columns([3, 1])
    
    # Prepare simulation parameters with optimal values
    sim_params = {
        'tau': st.session_state.optimal_tau,
        'gamma': st.session_state.optimal_gamma,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'initial_hashrate': hashrate[0],
        'initial_block_reward': block_reward_path[0],
        'time_steps': time_steps,
        'time_paths': {
            'exchange_rate': e_path,
            'efficiency': efficiency_path,
            'electricity_cost': electricity_cost_path
        }
    }
    
    # Run simulation
    results = run_simulation(sim_params)
    
    # Calculate cost per block for final predicted hashrate
    block_time_hours = 10/60  # 10 minutes in hours
    final_predicted_hashrate = results['hashrate'][-1]
    final_actual_hashrate = hashrate[-1] if len(hashrate) > 0 else hashrate[0]
    final_efficiency = efficiency_path[-1] if len(efficiency_path) > 0 else efficiency_path[0]
    final_electricity_cost = electricity_cost_path[-1] if len(electricity_cost_path) > 0 else electricity_cost_path[0]
    
    # Formula: (Hashrate / Efficiency) * Electricity Cost * Block Time
    final_predicted_energy_per_block = (final_predicted_hashrate/1e18 / final_efficiency) * block_time_hours
    final_predicted_cost_per_block = final_predicted_energy_per_block * final_electricity_cost
    
    # Also calculate for actual hashrate
    final_actual_energy_per_block = (final_actual_hashrate/1e18 / final_efficiency) * block_time_hours
    final_actual_cost_per_block = final_actual_energy_per_block * final_electricity_cost
    
    # Display plots in the first column
    with opt_col1:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot hashrate
        ax1.plot(results['hashrate'], label='Simulated Hashrate', color='blue')
        ax1.plot(hashrate[:time_steps], 
                label='Actual Hashrate',
                marker='x',
                alpha=0.6,
                linestyle='--',
                color='red')
        ax1.set_title('Actual vs Predicted Hashrate with Optimal Parameters')
        ax1.set_xlabel('Time Steps (days)')
        ax1.set_ylabel('Hashrate')
        ax1.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
        ax1.axhline(y=lower_bound, color='g', linestyle='--', label='Lower Bound')
        ax1.grid(True)
        ax1.set_yscale('log')
        ax1.legend()
        
        # Plot block rewards
        ax2.plot(results['block_reward'], label='Block Reward', color='orange')
        ax2.plot(block_reward_path[:time_steps], 
                label='Actual Block Reward',
                marker='x',
                alpha=0.6,
                linestyle='--',
                color='blue')
        ax2.set_title('Miners Block Reward over Time with Optimal Parameters')
        ax2.set_xlabel('Time Steps (days)')
        ax2.set_ylabel('Block Reward (log scale)')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Display statistics in the second column
    with opt_col2:
        st.header("Statistics")
        
        # Current Values
        st.subheader("Current Values")
        st.metric("Final Hashrate", f"{results['hashrate'][-1]:.2e} H/s")
        st.metric("Final Block Reward (per day)", f"{results['block_reward'][-1]:.2f}")
        st.metric("Number of Epochs", results['epochs'])
        
        # Electricity Cost section
        st.subheader("Electricity Costs")
        st.metric("Predicted Cost per Block", f"${final_predicted_cost_per_block:.2f}")
        st.metric("Actual Cost per Block", f"${final_actual_cost_per_block:.2f}")
        
        # Volatility Metrics
        st.subheader("Volatility Metrics")
        st.metric("Hashrate Volatility: (std / mean)", f"{results['stats']['hashrate_volatility']:.2f}%")
        
        # Performance Metrics
        st.subheader("Performance Metrics")
        st.metric("Time in Bounds", f"{results['stats']['bound_adherence']:.1f}%")
    