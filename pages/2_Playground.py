import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
from PIL import Image
from src.models.blockchain import Blockchain
from src.simulation.engine import run_simulation
import math

# Load default time path data
try:
    default_data = pd.read_csv('data/merged_data.csv')
    default_exchange_rate_path = default_data['BTCPrice'].tolist()
    default_efficiency_path = default_data['Efficiency'].tolist()
    default_electricity_cost_path = default_data['ElectricityPrice'].tolist()
    hashrate_path = default_data['HashRate'].tolist()
    block_reward_path = default_data['BTC'].tolist()
    default_time_steps = len(default_exchange_rate_path)
except Exception as e:
    st.warning(f"Could not load default data: {e}. Using synthetic defaults.")
    # Fallback defaults if data can't be loaded
    default_time_steps = 50
    default_exchange_rate_path = [50000.0] * default_time_steps
    default_efficiency_path = [0.001] * default_time_steps
    default_electricity_cost_path = [0.05] * default_time_steps
    hashrate_path = [0.0] * default_time_steps
    default_block_reward_path = [6.25] * default_time_steps

st.set_page_config(page_title="Custom Simulation Playground", layout="wide", page_icon="🎮")

# Page title
st.title("Blockchain Simulation Playground")
st.markdown("""
This playground allows you to create custom time paths for different parameters and run simulations to see 
how the blockchain system would behave under those conditions.
""")

# Sidebar for parameters
st.sidebar.header("Control Parameters")
tau = st.sidebar.number_input("Tau", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
gamma = st.sidebar.number_input("Gamma", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

st.sidebar.header("Target Bounds")
# Add toggle for setting bounds by electricity cost instead of hashrate
use_electricity_cost_bounds = st.sidebar.checkbox("Set bounds by electricity cost instead of hashrate", value=False)

# Calculate mean hashrate for default value
mean_hashrate = float(default_data['HashRate'].mean())

# Extract significant digits and exponent
sig_figs, exponent = "{:.3e}".format(mean_hashrate).split('e')
sig_figs = float(sig_figs)
exponent = int(exponent)

# Calculate mean electricity cost equivalent
block_time_seconds = 10 * 60  # 10 minutes in seconds
current_efficiency = default_efficiency_path[-1]  # J/TH
current_electricity_cost = default_electricity_cost_path[-1]  # $/kWh
joules_to_kwh = 2.78e-7
hashrate_in_th = mean_hashrate / 1e12

# Calculate equivalent electricity cost for mean hashrate
mean_energy_per_block = hashrate_in_th * block_time_seconds * current_efficiency * joules_to_kwh
mean_cost_per_block = mean_energy_per_block * current_electricity_cost

if not use_electricity_cost_bounds:
    # Create input for significant figures only (hashrate mode)
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
else:
    # Display inputs for electricity cost instead
    target_cost_in_millions = st.sidebar.number_input(
        "Target Electricity Cost per Block ($ millions)", 
        min_value=0.000001,
        max_value=1000.0,
        value=float(f"{mean_cost_per_block / 1000000:.6f}"),
        format="%.6f",
        help="Enter the target electricity cost per block in millions of USD"
    )
    
    # Convert from millions to actual dollars for calculations
    target_cost = target_cost_in_millions * 1000000

    # Input field for range percentage with 10% as default
    range_percent = st.sidebar.number_input(
        "Range (% of target cost)", 
        min_value=1.0, 
        max_value=100.0, 
        value=10.0, 
        step=1.0,
        help="Range as a percentage of the target electricity cost"
    )

    # Calculate upper and lower bounds for electricity cost
    upper_cost_bound = target_cost * (1 + range_percent / 100.0)
    lower_cost_bound = target_cost * (1 - range_percent / 100.0)

    # Display the cost bounds (in millions)
    st.sidebar.text(f"Upper Cost Bound: ${upper_cost_bound/1000000:.6f}M")
    st.sidebar.text(f"Lower Cost Bound: ${lower_cost_bound/1000000:.6f}M")
    
    # Convert cost bounds back to hashrate for simulation
    # Formula: hashrate (TH/s) = cost / (seconds * efficiency * joules_to_kwh * electricity_cost)
    # Then convert TH/s to H/s by multiplying by 1e12
    upper_hashrate_th = upper_cost_bound / (block_time_seconds * current_efficiency * joules_to_kwh * current_electricity_cost)
    lower_hashrate_th = lower_cost_bound / (block_time_seconds * current_efficiency * joules_to_kwh * current_electricity_cost)
    
    # Convert to H/s for simulation
    upper_bound = upper_hashrate_th * 1e12
    lower_bound = lower_hashrate_th * 1e12
    
    # Also set target_hashrate for consistency with the rest of the code
    target_hashrate = (upper_bound + lower_bound) / 2
    
    # Show equivalent hashrate bounds
    st.sidebar.text(f"Equivalent Upper Hashrate: {upper_bound:.2e} H/s")
    st.sidebar.text(f"Equivalent Lower Hashrate: {lower_bound:.2e} H/s")

# Parameter selection
st.header("Select Time Paths to Customize")
modify_exchange_rate = st.checkbox("Modify Exchange Rate Path", value=True)  # Default selected
modify_efficiency = st.checkbox("Modify Efficiency Path", value=False)
modify_electricity_cost = st.checkbox("Modify Electricity Cost Path", value=False)

# Simulation length - replace the two sliders with a single range slider
st.subheader("Simulation Window")

# Get the maximum possible time step based on data length
max_time_step = 1163

# Use a range slider for selecting both start and end points
start_point, end_point = st.select_slider(
    "Select time step range",
    options=range(0, max_time_step + 1),
    value=(0, min(500, max_time_step))
)

# Calculate time steps based on the window
time_steps = end_point - start_point

# Show selected range information
st.info(f"Selected range: {start_point} to {end_point} (Total: {time_steps} time steps)")

# Method selection tabs
tab1, tab2 = st.tabs(["Manual Pattern Creation", "Upload CSV Data"])

model = False if modify_electricity_cost else True

with tab1:
    any_parameter_selected = modify_exchange_rate or modify_efficiency or modify_electricity_cost

    if any_parameter_selected:
        st.subheader("Create Patterns Manually")
        
        # Initialize paths with default values if not modified
        if not modify_exchange_rate:
            exchange_rate_path = default_exchange_rate_path[:time_steps] if time_steps <= len(default_exchange_rate_path) else default_exchange_rate_path + [default_exchange_rate_path[-1]] * (time_steps - len(default_exchange_rate_path))
        
        if not modify_efficiency:
            efficiency_path = default_efficiency_path[:time_steps] if time_steps <= len(default_efficiency_path) else default_efficiency_path + [default_efficiency_path[-1]] * (time_steps - len(default_efficiency_path))
        
        if not modify_electricity_cost:
            electricity_cost_path = default_electricity_cost_path[:time_steps] if time_steps <= len(default_electricity_cost_path) else default_electricity_cost_path + [default_electricity_cost_path[-1]] * (time_steps - len(default_electricity_cost_path))
        
        # Exchange Rate Pattern
        if modify_exchange_rate:
            st.markdown("### Exchange Rate Pattern")
            exchange_pattern_type = st.selectbox(
                "Select Exchange Rate Pattern",
                ["Steady", "Linear Increase", "Linear Decrease", "Cyclical", "Random Walk", "Step Function"],
                key="exchange_pattern"
            )
            
            if exchange_pattern_type == "Steady":
                base_value_exchange = st.number_input("Exchange Rate Base Value (USD)", min_value=1.0, value=50000.0)
                exchange_rate_path = [base_value_exchange] * time_steps
            
            elif exchange_pattern_type == "Linear Increase":
                start_exchange = st.number_input("Exchange Rate Start Value (USD)", min_value=1.0, value=30000.0)
                end_exchange = st.number_input("Exchange Rate End Value (USD)", min_value=1.0, value=70000.0)
                exchange_rate_path = np.linspace(start_exchange, end_exchange, time_steps).tolist()
            
            elif exchange_pattern_type == "Linear Decrease":
                start_exchange = st.number_input("Exchange Rate Start Value (USD)", min_value=1.0, value=70000.0)
                end_exchange = st.number_input("Exchange Rate End Value (USD)", min_value=1.0, value=30000.0)
                exchange_rate_path = np.linspace(start_exchange, end_exchange, time_steps).tolist()
            
            elif exchange_pattern_type == "Cyclical":
                base_exchange = st.number_input("Exchange Rate Base Value (USD)", min_value=1.0, value=50000.0)
                amplitude_exchange = st.number_input("Exchange Rate Amplitude (USD)", min_value=0.0, value=20000.0)
                cycles_exchange = st.number_input("Number of Complete Cycles", min_value=0.1, value=3.0)
                exchange_rate_path = [base_exchange + amplitude_exchange * math.sin(2 * math.pi * cycles_exchange * t / time_steps) for t in range(time_steps)]
            
            elif exchange_pattern_type == "Random Walk":
                start_exchange = st.number_input("Exchange Rate Start Value (USD)", min_value=1.0, value=50000.0)
                volatility_exchange = st.number_input("Exchange Rate Volatility (%)", min_value=0.1, value=2.0)
                np.random.seed(42)  # For reproducibility
                
                # Generate random walk
                percent_changes = np.random.normal(0, volatility_exchange / 100, time_steps)
                exchange_rate_path = [start_exchange]
                for i in range(1, time_steps):
                    next_value = exchange_rate_path[-1] * (1 + percent_changes[i])
                    exchange_rate_path.append(max(1.0, next_value))  # Ensure minimum value of 1.0
            
            elif exchange_pattern_type == "Step Function":
                initial_exchange = st.number_input("Initial Exchange Rate (USD)", min_value=1.0, value=30000.0)
                final_exchange = st.number_input("Final Exchange Rate (USD)", min_value=1.0, value=70000.0)
                step_point = st.slider("Step Point (% of time steps)", min_value=0, max_value=100, value=50)
                step_index = int(time_steps * step_point / 100)
                exchange_rate_path = [initial_exchange] * step_index + [final_exchange] * (time_steps - step_index)
            
            st.markdown("---")  # Divider between features
        
        # Efficiency Pattern
        if modify_efficiency:
            st.markdown("### Efficiency Pattern")
            efficiency_pattern_type = st.selectbox(
                "Select Efficiency Pattern",
                ["Steady", "Linear Increase", "Linear Decrease", "Cyclical", "Random Walk", "Step Function"],
                key="efficiency_pattern"
            )
            
            if efficiency_pattern_type == "Steady":
                base_value_efficiency = st.number_input("Efficiency Base Value", min_value=0.0001, value=33.)
                efficiency_path = [base_value_efficiency] * time_steps
            
            elif efficiency_pattern_type == "Linear Increase":
                start_efficiency = st.number_input("Efficiency Start Value", min_value=0.0001, value=10.)
                end_efficiency = st.number_input("Efficiency End Value", min_value=0.0001, value=30.)
                efficiency_path = np.linspace(start_efficiency, end_efficiency, time_steps).tolist()
            
            elif efficiency_pattern_type == "Linear Decrease":
                start_efficiency = st.number_input("Efficiency Start Value", min_value=0.0001, value=40.)
                end_efficiency = st.number_input("Efficiency End Value", min_value=0.0001, value=10.)
                efficiency_path = np.linspace(start_efficiency, end_efficiency, time_steps).tolist()

            elif efficiency_pattern_type == "Cyclical":
                base_efficiency = st.number_input("Efficiency Base Value", min_value=0.0001, value=33.)
                amplitude_efficiency = st.number_input("Efficiency Amplitude", min_value=0.0, value=0.0002)
                cycles_efficiency = st.number_input("Number of Complete Cycles (Efficiency)", min_value=0.1, value=3.0)
                efficiency_path = [base_efficiency + amplitude_efficiency * math.sin(2 * math.pi * cycles_efficiency * t / time_steps) for t in range(time_steps)]
            
            elif efficiency_pattern_type == "Random Walk":
                start_efficiency = st.number_input("Efficiency Start Value", min_value=0.0001, value=25.)
                volatility_efficiency = st.number_input("Efficiency Volatility (%)", min_value=0.1, value=2.0)
                np.random.seed(43)  # Different seed from exchange rate
                
                # Generate random walk
                percent_changes = np.random.normal(0, volatility_efficiency / 100, time_steps)
                efficiency_path = [start_efficiency]
                for i in range(1, time_steps):
                    next_value = efficiency_path[-1] * (1 + percent_changes[i])
                    efficiency_path.append(max(0.0001, next_value))  # Ensure minimum value
            
            elif efficiency_pattern_type == "Step Function":
                initial_efficiency = st.number_input("Initial Efficiency", min_value=0.0001, value=10.)
                final_efficiency = st.number_input("Final Efficiency", min_value=0.0001, value=30.)
                step_point_eff = st.slider("Step Point (% of time steps) - Efficiency", min_value=0, max_value=100, value=50)
                step_index_eff = int(time_steps * step_point_eff / 100)
                efficiency_path = [initial_efficiency] * step_index_eff + [final_efficiency] * (time_steps - step_index_eff)
            
            st.markdown("---")  # Divider between features
        
        # Electricity Cost Pattern
        if modify_electricity_cost:
            st.markdown("### Electricity Cost Pattern")
            electricity_pattern_type = st.selectbox(
                "Select Electricity Cost Pattern",
                ["Steady", "Linear Increase", "Linear Decrease", "Cyclical", "Random Walk", "Step Function"],
                key="electricity_pattern"
            )
            
            if electricity_pattern_type == "Steady":
                base_value_electricity = st.number_input("Electricity Cost Base Value (USD/kWh)", min_value=0.01, value=0.15)
                electricity_cost_path = [base_value_electricity] * time_steps
            
            elif electricity_pattern_type == "Linear Increase":
                start_electricity = st.number_input("Electricity Cost Start Value (USD/kWh)", min_value=0.01, value=0.03)
                end_electricity = st.number_input("Electricity Cost End Value (USD/kWh)", min_value=0.01, value=0.07)
                electricity_cost_path = np.linspace(start_electricity, end_electricity, time_steps).tolist()
            
            elif electricity_pattern_type == "Linear Decrease":
                start_electricity = st.number_input("Electricity Cost Start Value (USD/kWh)", min_value=0.01, value=0.07)
                end_electricity = st.number_input("Electricity Cost End Value (USD/kWh)", min_value=0.01, value=0.03)
                electricity_cost_path = np.linspace(start_electricity, end_electricity, time_steps).tolist()
            
            elif electricity_pattern_type == "Cyclical":
                base_electricity = st.number_input("Electricity Cost Base Value (USD/kWh)", min_value=0.01, value=0.05)
                amplitude_electricity = st.number_input("Electricity Cost Amplitude (USD/kWh)", min_value=0.0, value=0.02)
                cycles_electricity = st.number_input("Number of Complete Cycles (Electricity)", min_value=0.1, value=3.0)
                electricity_cost_path = [base_electricity + amplitude_electricity * math.sin(2 * math.pi * cycles_electricity * t / time_steps) for t in range(time_steps)]
            
            elif electricity_pattern_type == "Random Walk":
                start_electricity = st.number_input("Electricity Cost Start Value (USD/kWh)", min_value=0.01, value=0.05)
                volatility_electricity = st.number_input("Electricity Cost Volatility (%)", min_value=0.1, value=2.0)
                np.random.seed(44)  # Different seed from others
                
                # Generate random walk
                percent_changes = np.random.normal(0, volatility_electricity / 100, time_steps)
                electricity_cost_path = [start_electricity]
                for i in range(1, time_steps):
                    next_value = electricity_cost_path[-1] * (1 + percent_changes[i])
                    electricity_cost_path.append(max(0.01, next_value))  # Ensure minimum value
            
            elif electricity_pattern_type == "Step Function":
                initial_electricity = st.number_input("Initial Electricity Cost (USD/kWh)", min_value=0.01, value=0.03)
                final_electricity = st.number_input("Final Electricity Cost (USD/kWh)", min_value=0.01, value=0.07)
                step_point_elec = st.slider("Step Point (% of time steps) - Electricity", min_value=0, max_value=100, value=50)
                step_index_elec = int(time_steps * step_point_elec / 100)
                electricity_cost_path = [initial_electricity] * step_index_elec + [final_electricity] * (time_steps - step_index_elec)

with tab2:
    st.subheader("Upload CSV Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(data.head())
            
            # Column selection
            exchange_col = st.selectbox("Select Exchange Rate column", data.columns)
            efficiency_col = st.selectbox("Select Efficiency column", data.columns)
            electricity_col = st.selectbox("Select Electricity Cost column", data.columns)
            
            # Extract time paths from the selected columns
            exchange_rate_path = data[exchange_col].tolist()
            efficiency_path = data[efficiency_col].tolist()
            electricity_cost_path = data[electricity_col].tolist()
            
            # Update time steps based on the data length
            time_steps = len(exchange_rate_path)
            
            st.success(f"Successfully loaded {time_steps} data points")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            # Set default values
            exchange_rate_path = [50000.0] * time_steps
            efficiency_path = [0.001] * time_steps
            electricity_cost_path = [0.05] * time_steps

# Combine custom and default paths
# For exchange rate
if modify_exchange_rate:
    # Keep the custom path that was generated
    exchange_rate_path_display = exchange_rate_path
    path_type_exchange = "(Custom)"
else:
    # Use default data within the selected window
    if end_point <= len(default_exchange_rate_path):
        exchange_rate_path = default_exchange_rate_path[start_point:end_point]
    else:
        # Handle case where end_point exceeds data length
        available_data = default_exchange_rate_path[start_point:]
        extension = [default_exchange_rate_path[-1]] * (end_point - len(default_exchange_rate_path))
        exchange_rate_path = available_data + extension
    exchange_rate_path_display = exchange_rate_path
    path_type_exchange = "(Default Historical Data)"

# For efficiency
if modify_efficiency:
    # Keep the custom path that was generated
    efficiency_path_display = efficiency_path
    path_type_efficiency = "(Custom)"
else:
    # Use default data within the selected window
    if end_point <= len(default_efficiency_path):
        efficiency_path = default_efficiency_path[start_point:end_point]
    else:
        # Handle case where end_point exceeds data length
        available_data = default_efficiency_path[start_point:]
        extension = [default_efficiency_path[-1]] * (end_point - len(default_efficiency_path))
        efficiency_path = available_data + extension
    efficiency_path_display = efficiency_path
    path_type_efficiency = "(Default Historical Data)"

# For electricity cost
if modify_electricity_cost:
    # Keep the custom path that was generated
    electricity_cost_path_display = electricity_cost_path
    path_type_electricity = "(Custom)"
else:
    # Use default data within the selected window
    if end_point <= len(default_electricity_cost_path):
        electricity_cost_path = default_electricity_cost_path[start_point:end_point]
    else:
        # Handle case where end_point exceeds data length
        available_data = default_electricity_cost_path[start_point:]
        extension = [default_electricity_cost_path[-1]] * (end_point - len(default_electricity_cost_path))
        electricity_cost_path = available_data + extension
    electricity_cost_path_display = electricity_cost_path
    path_type_electricity = "(Default Historical Data)"

# Initial conditions - use the start of the window
initial_hashrate = hashrate_path[start_point] if start_point < len(hashrate_path) else hashrate_path[0]
initial_block_reward = block_reward_path[start_point] if start_point < len(block_reward_path) else block_reward_path[0]

# Adjust hashrate path for plotting comparison
hashrate_for_comparison = hashrate_path[start_point:end_point] if end_point <= len(hashrate_path) else hashrate_path[start_point:]
if len(hashrate_for_comparison) < time_steps:
    hashrate_for_comparison = hashrate_for_comparison + [hashrate_for_comparison[-1]] * (time_steps - len(hashrate_for_comparison))

# Adjust block reward path for plotting comparison
block_reward_for_comparison = block_reward_path[start_point:end_point] if end_point <= len(block_reward_path) else block_reward_path[start_point:]
if len(block_reward_for_comparison) < time_steps:
    block_reward_for_comparison = block_reward_for_comparison + [block_reward_for_comparison[-1]] * (time_steps - len(block_reward_for_comparison))

# Visualize the time paths
st.header("Preview Time Paths")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# Create x-axis range based on the window
x_range = list(range(start_point, end_point))

# Plot with proper time step values
ax1.plot(x_range, exchange_rate_path_display, color='blue')
ax1.set_title(f'Exchange Rate Path {path_type_exchange}')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Exchange Rate (USD)')
ax1.grid(True)

ax2.plot(x_range, efficiency_path_display, color='green')
ax2.set_title(f'Efficiency Path {path_type_efficiency}')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Efficiency')
ax2.grid(True)

ax3.plot(x_range, electricity_cost_path_display, color='red')
ax3.set_title(f'Electricity Cost Path {path_type_electricity}')
ax3.set_xlabel('Time Steps')
ax3.set_ylabel('Electricity Cost (USD/kWh)')
ax3.grid(True)

plt.tight_layout()
st.pyplot(fig)

# Run simulation button
if st.button("Run Simulation with Custom Data"):
    # Initialize blockchain with parameters
    blockchain = Blockchain(
        tau=tau,
        gamma=gamma,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        initial_difficulty=1e12
    )

    sim_params = {
            'model': model,
            'tau': tau,
            'gamma': gamma,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'initial_hashrate': initial_hashrate,
            'initial_block_reward': initial_block_reward,
            'time_steps': time_steps,
            'time_paths': {
                'exchange_rate': exchange_rate_path,
                'efficiency': efficiency_path,
                'electricity_cost': electricity_cost_path
            }
        }
    
    # Run simulation
    results = run_simulation(sim_params)
    
    # Create columns for plots and statistics
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Simulation Results")
        # Create x-axis range based on the window
        x_range = list(range(start_point, end_point))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot hashrate with proper time step values
        ax1.plot(x_range, results['hashrate'], label='Simulated Hashrate', color='blue')
        ax1.plot(x_range, hashrate_for_comparison, 
                label='Actual Hashrate',
                marker='x',
                alpha=0.6,
                linestyle='--',
                color='red')
        ax1.set_title('Hashrate over Time')
        ax1.set_xlabel('Time Steps (days)')
        ax1.set_ylabel('Hashrate')
        ax1.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
        ax1.axhline(y=lower_bound, color='g', linestyle='--', label='Lower Bound')
        ax1.grid(True)
        ax1.set_yscale('log')
        ax1.legend()
        
        # Plot block rewards with proper time step values
        ax2.plot(x_range, results['block_reward'], label='Block Reward', color='orange')
        ax2.plot(x_range, block_reward_for_comparison, 
                label='Actual Block Reward',
                marker='x',
                alpha=0.6,
                linestyle='--',
                color='blue')
        ax2.set_title('Miners Block Reward over Time')
        ax2.set_xlabel('Time Steps (days)')
        ax2.set_ylabel('Block Reward (log scale)')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.header("Statistics")
        
        # Calculate cost per block for final predicted hashrate
        block_time_seconds = 10 * 60  # 10 minutes in seconds
        final_predicted_hashrate = results['hashrate'][-1]
        final_actual_hashrate = hashrate_for_comparison[-1] if len(hashrate_for_comparison) > 0 else hashrate_for_comparison[0]
        final_efficiency = efficiency_path[-1] if len(efficiency_path) > 0 else efficiency_path[0]
        final_electricity_cost = electricity_cost_path[-1] if len(electricity_cost_path) > 0 else electricity_cost_path[0]
        
        # Convert hashrates from H/s to TH/s
        pred_hashrate_in_th = final_predicted_hashrate / 1e12
        actual_hashrate_in_th = final_actual_hashrate / 1e12
        
        # Convert J/TH to kWh/TH - 1 J = 2.78e-7 kWh
        joules_to_kwh = 2.78e-7
        
        # Calculate energy and cost for predicted hashrate
        final_predicted_energy_per_block = pred_hashrate_in_th * block_time_seconds * final_efficiency * joules_to_kwh
        final_predicted_cost_per_block = final_predicted_energy_per_block * final_electricity_cost
        
        # Calculate for actual hashrate
        final_actual_energy_per_block = actual_hashrate_in_th * block_time_seconds * final_efficiency * joules_to_kwh
        final_actual_cost_per_block = final_actual_energy_per_block * final_electricity_cost
        
        # Current Values
        st.subheader("Current Values")
        st.metric("Final Adjusted Hashrate", f"{results['hashrate'][-1]:.2e} H/s")
        st.metric("Final Baseline Hashrate", f"{hashrate_for_comparison[-1]:.2e} H/s")
        st.metric("Final Adjusted Block Reward (per block)", f"${results['block_reward'][-1]/144*exchange_rate_path[-1]:,.2f}")
        st.metric("Final Baseline Block Reward (per block)", f"${block_reward_for_comparison[-1]/144*exchange_rate_path[-1]:,.2f}")
        st.metric("Number of Epochs", results['epochs'])
        
        # Electricity Cost section
        st.subheader("Electricity Costs")
        st.metric("Adjusted Cost per Block", f"${final_predicted_cost_per_block:,.2f}")
        st.metric("Baseline Cost per Block", f"${final_actual_cost_per_block:,.2f}")
        
        # Volatility Metrics
        st.subheader("Volatility Metrics")
        st.metric("Hashrate Volatility: (std / mean)", f"{results['stats']['hashrate_volatility']:.2f}%")
        
        # Performance Metrics
        st.subheader("Performance Metrics")
        st.metric("Time in Bounds", f"{results['stats']['bound_adherence']:.1f}%")
    
    # Export results option
    st.subheader("Export Results")
    N = len(results['hashrate'])
    results_df = pd.DataFrame({
        'Time_Step': list(range(N)),
        'Hashrate': results['hashrate'],
        'Block_Reward': results['block_reward'],
        'Exchange_Rate': exchange_rate_path,
        'Efficiency': efficiency_path,
        'Electricity_Cost': electricity_cost_path
    })
    
    # Convert to CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="blockchain_simulation_results.csv",
        mime="text/csv",
    )