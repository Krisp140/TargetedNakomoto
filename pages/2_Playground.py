import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
from PIL import Image
from src.models.blockchain import Blockchain
from src.simulation.engine import run_simulation

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

# Target bounds
st.sidebar.header("Target Bounds")
mean_hashrate = float(default_data['HashRate'].mean())
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

# Calculate actual bounds
upper_bound = (upper_bound_percent / 100.0) * mean_hashrate
lower_bound = (lower_bound_percent / 100.0) * mean_hashrate

# Display the actual values
st.sidebar.text(f"Actual Upper Bound: {upper_bound:.2e}")
st.sidebar.text(f"Actual Lower Bound: {lower_bound:.2e}")

# Parameter selection
st.header("Select Time Paths to Customize")
modify_exchange_rate = st.checkbox("Modify Exchange Rate Path", value=True)  # Default selected
modify_efficiency = st.checkbox("Modify Efficiency Path", value=False)
modify_electricity_cost = st.checkbox("Modify Electricity Cost Path", value=False)

# Simulation length
time_steps = st.slider("Simulation Length (time steps)", min_value=10, max_value=1163, value=500)

# Method selection tabs
tab1, tab2 = st.tabs(["Manual Pattern Creation", "Upload CSV Data"])

model = False if modify_electricity_cost else True

with tab1:
    any_parameter_selected = modify_exchange_rate or modify_efficiency or modify_electricity_cost

    if any_parameter_selected:
        st.subheader("Create Patterns Manually")
        pattern_type = st.selectbox(
            "Select Pattern Type",
            ["Steady", "Linear Increase", "Linear Decrease", "Cyclical", "Random Walk", "Step Function"]
        )
        
        # Initialize paths with default values if not modified
        if not modify_exchange_rate:
            exchange_rate_path = default_exchange_rate_path[:time_steps] if time_steps <= len(default_exchange_rate_path) else default_exchange_rate_path + [default_exchange_rate_path[-1]] * (time_steps - len(default_exchange_rate_path))
        
        if not modify_efficiency:
            efficiency_path = default_efficiency_path[:time_steps] if time_steps <= len(default_efficiency_path) else default_efficiency_path + [default_efficiency_path[-1]] * (time_steps - len(default_efficiency_path))
        
        if not modify_electricity_cost:
            electricity_cost_path = default_electricity_cost_path[:time_steps] if time_steps <= len(default_electricity_cost_path) else default_electricity_cost_path + [default_electricity_cost_path[-1]] * (time_steps - len(default_electricity_cost_path))
    

        # Parameters for pattern generation
        if pattern_type == "Steady":
            if modify_exchange_rate:
                base_value_exchange = st.number_input("Exchange Rate Base Value (USD)", min_value=1.0, value=50000.0)
                exchange_rate_path = [base_value_exchange] * time_steps
            
            if modify_efficiency:
                base_value_efficiency = st.number_input("Efficiency Base Value", min_value=0.0001, value=0.001)
                efficiency_path = [base_value_efficiency] * time_steps
            
            if modify_electricity_cost:
                base_value_electricity = st.number_input("Electricity Cost Base Value (USD/kWh)", min_value=0.01, value=0.15)
                electricity_cost_path = [base_value_electricity] * time_steps
    
            
        elif pattern_type == "Linear Increase":
            if modify_exchange_rate:
                start_exchange = st.number_input("Exchange Rate Start Value (USD)", min_value=1.0, value=30000.0)
                end_exchange = st.number_input("Exchange Rate End Value (USD)", min_value=1.0, value=70000.0)
                exchange_rate_path = np.linspace(start_exchange, end_exchange, time_steps).tolist()
            
            if modify_efficiency:
                start_efficiency = st.number_input("Efficiency Start Value", min_value=0.0001, value=0.0008)
                end_efficiency = st.number_input("Efficiency End Value", min_value=0.0001, value=0.0012)
                efficiency_path = np.linspace(start_efficiency, end_efficiency, time_steps).tolist()
            
            if modify_electricity_cost:
                start_electricity = st.number_input("Electricity Cost Start Value (USD/kWh)", min_value=0.01, value=0.03)
                end_electricity = st.number_input("Electricity Cost End Value (USD/kWh)", min_value=0.01, value=0.07)
                electricity_cost_path = np.linspace(start_electricity, end_electricity, time_steps).tolist()
                 
            
        elif pattern_type == "Linear Decrease":
            if modify_exchange_rate:
                start_exchange = st.number_input("Exchange Rate Start Value (USD)", min_value=1.0, value=70000.0)
                end_exchange = st.number_input("Exchange Rate End Value (USD)", min_value=1.0, value=30000.0)
                exchange_rate_path = np.linspace(start_exchange, end_exchange, time_steps).tolist()
            
            if modify_efficiency:
                start_efficiency = st.number_input("Efficiency Start Value", min_value=0.0001, value=0.0012)
                end_efficiency = st.number_input("Efficiency End Value", min_value=0.0001, value=0.0008)
                efficiency_path = np.linspace(start_efficiency, end_efficiency, time_steps).tolist()
            
            if modify_electricity_cost:
                start_electricity = st.number_input("Electricity Cost Start Value (USD/kWh)", min_value=0.01, value=0.07)
                end_electricity = st.number_input("Electricity Cost End Value (USD/kWh)", min_value=0.01, value=0.03)
                electricity_cost_path = np.linspace(start_electricity, end_electricity, time_steps).tolist()
            
        elif pattern_type == "Cyclical":
            if modify_exchange_rate:
                base_exchange = st.number_input("Exchange Rate Base Value (USD)", min_value=1.0, value=50000.0)
                amplitude_exchange = st.number_input("Exchange Rate Amplitude (USD)", min_value=0.0, value=20000.0)
                periods_exchange = st.number_input("Number of Exchange Rate Cycles", min_value=0.5, value=2.0, step=0.5)
                t = np.linspace(0, 2*np.pi*periods_exchange, time_steps)
                exchange_rate_path = (base_exchange + amplitude_exchange * np.sin(t)).tolist()
            
            if modify_efficiency:
                base_efficiency = st.number_input("Efficiency Base Value", min_value=0.0001, value=0.001)
                amplitude_efficiency = st.number_input("Efficiency Amplitude", min_value=0.0, value=0.0002)
                periods_efficiency = st.number_input("Number of Efficiency Cycles", min_value=0.5, value=1.5, step=0.5)
                t = np.linspace(0, 2*np.pi*periods_efficiency, time_steps)
                efficiency_path = (base_efficiency + amplitude_efficiency * np.sin(t)).tolist()
            
            if modify_electricity_cost:
                base_electricity = st.number_input("Electricity Cost Base Value (USD/kWh)", min_value=0.01, value=0.05)
                amplitude_electricity = st.number_input("Electricity Cost Amplitude (USD/kWh)", min_value=0.0, value=0.02)
                periods_electricity = st.number_input("Number of Electricity Cost Cycles", min_value=0.5, value=1.0, step=0.5)
                t = np.linspace(0, 2*np.pi*periods_electricity, time_steps)
                electricity_cost_path = (base_electricity + amplitude_electricity * np.sin(t)).tolist()
            
        elif pattern_type == "Random Walk":
            seed = st.number_input("Random Seed", min_value=0, value=42)
            np.random.seed(seed)  # Set the seed for reproducibility
            
            if modify_exchange_rate:
                start_exchange = st.number_input("Exchange Rate Start Value (USD)", min_value=1.0, value=50000.0)
                volatility_exchange = st.number_input("Exchange Rate Volatility (%)", min_value=0.1, value=5.0)
                
                # Generate random walk path for exchange rate
                exchange_rate_path = [start_exchange]
                for _ in range(time_steps-1):
                    change = exchange_rate_path[-1] * (volatility_exchange/100) * np.random.randn()
                    exchange_rate_path.append(max(1.0, exchange_rate_path[-1] + change))
            
            if modify_efficiency:
                start_efficiency = st.number_input("Efficiency Start Value", min_value=0.0001, value=0.001)
                volatility_efficiency = st.number_input("Efficiency Volatility (%)", min_value=0.1, value=3.0)
                
                # Generate random walk path for efficiency
                efficiency_path = [start_efficiency]
                for _ in range(time_steps-1):
                    change = efficiency_path[-1] * (volatility_efficiency/100) * np.random.randn()
                    efficiency_path.append(max(0.0001, efficiency_path[-1] + change))
            
            if modify_electricity_cost:
                start_electricity = st.number_input("Electricity Cost Start Value (USD/kWh)", min_value=0.01, value=0.05)
                volatility_electricity = st.number_input("Electricity Cost Volatility (%)", min_value=0.1, value=2.0)
                
                # Generate random walk path for electricity cost
                electricity_cost_path = [start_electricity]
                for _ in range(time_steps-1):
                    change = electricity_cost_path[-1] * (volatility_electricity/100) * np.random.randn()
                    electricity_cost_path.append(max(0.01, electricity_cost_path[-1] + change))
        
        elif pattern_type == "Step Function":
            if modify_exchange_rate:
                base_exchange = st.number_input("Exchange Rate Base Value (USD)", min_value=1.0, value=50000.0)
                step_size_exchange = st.number_input("Exchange Rate Step Size (USD)", min_value=0.0, value=10000.0)
                step_points_exchange = st.multiselect(
                    "Exchange Rate Step Points (time steps)",
                    options=list(range(time_steps)),
                    default=[int(time_steps/3), int(2*time_steps/3)]
                )
                exchange_rate_path = [base_exchange] * time_steps
                for step in step_points_exchange:
                    for i in range(step, time_steps):
                        exchange_rate_path[i] += step_size_exchange
            
            if modify_efficiency:
                base_efficiency = st.number_input("Efficiency Base Value", min_value=0.0001, value=0.001)
                step_size_efficiency = st.number_input("Efficiency Step Size", min_value=0.0, value=0.0002)
                step_points_efficiency = st.multiselect(
                    "Efficiency Step Points (time steps)",
                    options=list(range(time_steps)),
                    default=[int(time_steps/4), int(3*time_steps/4)]
                )
                efficiency_path = [base_efficiency] * time_steps
                for step in step_points_efficiency:
                    for i in range(step, time_steps):
                        efficiency_path[i] += step_size_efficiency
            
            if modify_electricity_cost:
                base_electricity = st.number_input("Electricity Cost Base Value (USD/kWh)", min_value=0.01, value=0.05)
                step_size_electricity = st.number_input("Electricity Cost Step Size (USD/kWh)", min_value=0.0, value=0.02)
                step_points_electricity = st.multiselect(
                    "Electricity Cost Step Points (time steps)",
                    options=list(range(time_steps)),
                    default=[int(time_steps/2)]
                )
                electricity_cost_path = [base_electricity] * time_steps
                for step in step_points_electricity:
                    for i in range(step, time_steps):
                        electricity_cost_path[i] += step_size_electricity

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

# Replace the visualization section (around line 257-280) with this code:

# Combine custom and default paths
# For exchange rate
if modify_exchange_rate:
    # Keep the custom path that was generated
    exchange_rate_path_display = exchange_rate_path
    path_type_exchange = "(Custom)"
else:
    # Use default data, trimmed or extended to match time_steps
    if time_steps <= len(default_exchange_rate_path):
        exchange_rate_path = default_exchange_rate_path[:time_steps]
    else:
        # Extend with the last value if needed
        exchange_rate_path = default_exchange_rate_path + [default_exchange_rate_path[-1]] * (time_steps - len(default_exchange_rate_path))
    exchange_rate_path_display = exchange_rate_path
    path_type_exchange = "(Default Historical Data)"

# For efficiency
if modify_efficiency:
    # Keep the custom path that was generated
    efficiency_path_display = efficiency_path
    path_type_efficiency = "(Custom)"
else:
    # Use default data, trimmed or extended to match time_steps
    if time_steps <= len(default_efficiency_path):
        efficiency_path = default_efficiency_path[:time_steps]
    else:
        efficiency_path = default_efficiency_path + [default_efficiency_path[-1]] * (time_steps - len(default_efficiency_path))
    efficiency_path_display = efficiency_path
    path_type_efficiency = "(Default Historical Data)"

# For electricity cost
if modify_electricity_cost:
    # Keep the custom path that was generated
    electricity_cost_path_display = electricity_cost_path
    path_type_electricity = "(Custom)"
else:
    # Use default data, trimmed or extended to match time_steps
    if time_steps <= len(default_electricity_cost_path):
        electricity_cost_path = default_electricity_cost_path[:time_steps]
    else:
        electricity_cost_path = default_electricity_cost_path + [default_electricity_cost_path[-1]] * (time_steps - len(default_electricity_cost_path))
    electricity_cost_path_display = electricity_cost_path
    path_type_electricity = "(Default Historical Data)"

# Visualize the time paths
st.header("Preview Time Paths")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

ax1.plot(exchange_rate_path_display, color='blue')
ax1.set_title(f'Exchange Rate Path {path_type_exchange}')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Exchange Rate (USD)')
ax1.grid(True)

ax2.plot(efficiency_path_display, color='green')
ax2.set_title(f'Efficiency Path {path_type_efficiency}')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Efficiency')
ax2.grid(True)

ax3.plot(electricity_cost_path_display, color='red')
ax3.set_title(f'Electricity Cost Path {path_type_electricity}')
ax3.set_xlabel('Time Steps')
ax3.set_ylabel('Electricity Cost (USD/kWh)')
ax3.grid(True)

plt.tight_layout()
st.pyplot(fig)

#Initial conditions
initial_hashrate = hashrate_path[0]
initial_block_reward = block_reward_path[0]
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
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot hashrate
    ax1.plot(results['hashrate'], label='Simulated Hashrate', color='blue')
    ax1.plot(hashrate_path[:time_steps], 
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
    
    # Plot block rewards
    ax2.plot(results['block_reward'], label='Block Reward', color='orange')
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
    st.header("Statistics")
    
    # Current Values
    st.subheader("Current Values")
    st.metric("Final Hashrate", f"{results['hashrate'][-1]:.2e} H/s")
    st.metric("Final Block Reward", f"{results['block_reward'][-1]:.2f}")
    st.metric("Number of Epochs", results['epochs'])
    
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