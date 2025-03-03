import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
from PIL import Image
from src.models.blockchain import Blockchain

# Load default time path data
try:
    default_data = pd.read_csv('data/bitcoin_mining_data_daily.csv')
    default_exchange_rate_path = default_data['price'].tolist()
    default_efficiency_path = default_data['mining_cost'].tolist()
    default_electricity_cost_path = default_data['electricity_price'].tolist()
    default_time_steps = len(default_exchange_rate_path)
except Exception as e:
    st.warning(f"Could not load default data: {e}. Using synthetic defaults.")
    # Fallback defaults if data can't be loaded
    default_time_steps = 50
    default_exchange_rate_path = [50000.0] * default_time_steps
    default_efficiency_path = [0.001] * default_time_steps
    default_electricity_cost_path = [0.05] * default_time_steps


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
mean_hashrate = 6.53e+08 # Default value
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
time_steps = st.slider("Simulation Length (time steps)", min_value=10, max_value=365, value=200)

# Method selection tabs
tab1, tab2 = st.tabs(["Manual Pattern Creation", "Upload CSV Data"])

with tab1:
    any_parameter_selected = modify_exchange_rate or modify_efficiency or modify_electricity_cost

    if any_parameter_selected:
        st.subheader("Create Patterns Manually")
        pattern_type = st.selectbox(
            "Select Pattern Type",
            ["Steady", "Linear Increase", "Linear Decrease", "Cyclical", "Random Walk", "Step Function"]
    )
    

        # Parameters for pattern generation
        if pattern_type == "Steady":
            if modify_exchange_rate:
                base_value_exchange = st.number_input("Exchange Rate Base Value (USD)", min_value=1.0, value=50000.0)
                exchange_rate_path = [base_value_exchange] * time_steps
            if modify_efficiency:
                base_value_efficiency = st.number_input("Efficiency Base Value", min_value=0.0001, value=3e6)
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
            start_exchange = st.number_input("Exchange Rate Start Value (USD)", min_value=1.0, value=70000.0)
            end_exchange = st.number_input("Exchange Rate End Value (USD)", min_value=1.0, value=30000.0)
            
            start_efficiency = st.number_input("Efficiency Start Value", min_value=0.0001, value=0.0012)
            end_efficiency = st.number_input("Efficiency End Value", min_value=0.0001, value=0.0008)
            
            start_electricity = st.number_input("Electricity Cost Start Value (USD/kWh)", min_value=0.01, value=0.07)
            end_electricity = st.number_input("Electricity Cost End Value (USD/kWh)", min_value=0.01, value=0.03)
            
            # Generate linearly decreasing paths
            exchange_rate_path = np.linspace(start_exchange, end_exchange, time_steps).tolist()
            efficiency_path = np.linspace(start_efficiency, end_efficiency, time_steps).tolist()
            electricity_cost_path = np.linspace(start_electricity, end_electricity, time_steps).tolist()
            
        elif pattern_type == "Cyclical":
            base_exchange = st.number_input("Exchange Rate Base Value (USD)", min_value=1.0, value=50000.0)
            amplitude_exchange = st.number_input("Exchange Rate Amplitude (USD)", min_value=0.0, value=20000.0)
            periods_exchange = st.number_input("Number of Exchange Rate Cycles", min_value=0.5, value=2.0, step=0.5)
            
            base_efficiency = st.number_input("Efficiency Base Value", min_value=0.0001, value=0.001)
            amplitude_efficiency = st.number_input("Efficiency Amplitude", min_value=0.0, value=0.0002)
            periods_efficiency = st.number_input("Number of Efficiency Cycles", min_value=0.5, value=1.5, step=0.5)
            
            base_electricity = st.number_input("Electricity Cost Base Value (USD/kWh)", min_value=0.01, value=0.05)
            amplitude_electricity = st.number_input("Electricity Cost Amplitude (USD/kWh)", min_value=0.0, value=0.02)
            periods_electricity = st.number_input("Number of Electricity Cost Cycles", min_value=0.5, value=1.0, step=0.5)
            
            # Generate cyclical paths
            t = np.linspace(0, 2*np.pi*periods_exchange, time_steps)
            exchange_rate_path = (base_exchange + amplitude_exchange * np.sin(t)).tolist()
            
            t = np.linspace(0, 2*np.pi*periods_efficiency, time_steps)
            efficiency_path = (base_efficiency + amplitude_efficiency * np.sin(t)).tolist()
            
            t = np.linspace(0, 2*np.pi*periods_electricity, time_steps)
            electricity_cost_path = (base_electricity + amplitude_electricity * np.sin(t)).tolist()
            
        elif pattern_type == "Random Walk":
            seed = st.number_input("Random Seed", min_value=0, value=42)
            start_exchange = st.number_input("Exchange Rate Start Value (USD)", min_value=1.0, value=50000.0)
            volatility_exchange = st.number_input("Exchange Rate Volatility (%)", min_value=0.1, value=5.0)
            
            start_efficiency = st.number_input("Efficiency Start Value", min_value=0.0001, value=0.001)
            volatility_efficiency = st.number_input("Efficiency Volatility (%)", min_value=0.1, value=3.0)
            
            start_electricity = st.number_input("Electricity Cost Start Value (USD/kWh)", min_value=0.01, value=0.05)
            volatility_electricity = st.number_input("Electricity Cost Volatility (%)", min_value=0.1, value=2.0)
            
            # Set the seed for reproducibility
            np.random.seed(seed)
            
            # Generate random walk paths
            exchange_rate_path = [start_exchange]
            for _ in range(time_steps-1):
                change = exchange_rate_path[-1] * (volatility_exchange/100) * np.random.randn()
                exchange_rate_path.append(max(1.0, exchange_rate_path[-1] + change))
            
            efficiency_path = [start_efficiency]
            for _ in range(time_steps-1):
                change = efficiency_path[-1] * (volatility_efficiency/100) * np.random.randn()
                efficiency_path.append(max(0.0001, efficiency_path[-1] + change))
            
            electricity_cost_path = [start_electricity]
            for _ in range(time_steps-1):
                change = electricity_cost_path[-1] * (volatility_electricity/100) * np.random.randn()
                electricity_cost_path.append(max(0.01, electricity_cost_path[-1] + change))
        
        elif pattern_type == "Step Function":
            base_exchange = st.number_input("Exchange Rate Base Value (USD)", min_value=1.0, value=50000.0)
            step_size_exchange = st.number_input("Exchange Rate Step Size (USD)", min_value=0.0, value=10000.0)
            step_points_exchange = st.multiselect(
                "Exchange Rate Step Points (time steps)",
                options=list(range(time_steps)),
                default=[int(time_steps/3), int(2*time_steps/3)]
            )
            
            base_efficiency = st.number_input("Efficiency Base Value", min_value=0.0001, value=0.001)
            step_size_efficiency = st.number_input("Efficiency Step Size", min_value=0.0, value=0.0002)
            step_points_efficiency = st.multiselect(
                "Efficiency Step Points (time steps)",
                options=list(range(time_steps)),
                default=[int(time_steps/4), int(3*time_steps/4)]
            )
            
            base_electricity = st.number_input("Electricity Cost Base Value (USD/kWh)", min_value=0.01, value=0.05)
            step_size_electricity = st.number_input("Electricity Cost Step Size (USD/kWh)", min_value=0.0, value=0.02)
            step_points_electricity = st.multiselect(
                "Electricity Cost Step Points (time steps)",
                options=list(range(time_steps)),
                default=[int(time_steps/2)]
            )
            
            # Generate step function paths
            exchange_rate_path = [base_exchange] * time_steps
            for step in step_points_exchange:
                for i in range(step, time_steps):
                    exchange_rate_path[i] += step_size_exchange
            
            efficiency_path = [base_efficiency] * time_steps
            for step in step_points_efficiency:
                for i in range(step, time_steps):
                    efficiency_path[i] += step_size_efficiency
            
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

# Initial conditions section
st.header("Initial Conditions")
col1, col2 = st.columns(2)
with col1:
    initial_hashrate = st.number_input("Initial Hashrate", min_value=1e6, value=600517281.7255559, format="%e")
with col2:
    initial_block_reward = st.number_input("Initial Block Reward", min_value=0.1, value=6.25)

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
    
    # Run simulation
    N = [float(initial_hashrate)]
    P = [float(initial_block_reward)]
    
    for t in range(time_steps):
        current_N = N[-1]
        current_P = P[-1]

        # Adjust reward within the epoch
        adjusted_reward = blockchain.adjust_reward(current_P, len(blockchain.epochs) - 1)
        
        # Get time path values for this step (with bounds checking)
        e_current = exchange_rate_path[min(t, len(exchange_rate_path)-1)]
        efficiency_current = efficiency_path[min(t, len(efficiency_path)-1)]
        electricity_cost_current = electricity_cost_path[min(t, len(electricity_cost_path)-1)]
        
        # Calculate new hashrate
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
        
        # Check for epoch end
        if t > 0 and t % 14 == 0:
            blockchain.end_of_epoch()
    
    # Create visualization
    st.header("Simulation Results")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot hashrate
    ax1.plot(N, label='Simulated Hashrate', color='blue')
    ax1.set_title('Hashrate over Time')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Hashrate')
    ax1.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
    ax1.axhline(y=lower_bound, color='g', linestyle='--', label='Lower Bound')
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.legend()
    
    # Plot block rewards
    ax2.plot(P, label='Block Reward', color='orange')
    ax2.set_title('Miners Block Reward over Time')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Block Reward')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display statistics
    st.header("Simulation Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Hashrate", f"{N[-1]:.2e} H/s")
        st.metric("Initial Hashrate", f"{N[0]:.2e} H/s")
        st.metric("Hashrate Change", f"{(N[-1]/N[0] - 1)*100:.2f}%")
    
    with col2:
        st.metric("Final Block Reward", f"{P[-1]:.2f}")
        st.metric("Initial Block Reward", f"{P[0]:.2f}")
        st.metric("Reward Change", f"{(P[-1]/P[0] - 1)*100:.2f}%")
    
    with col3:
        st.metric("Number of Epochs", len(blockchain.epochs))
        time_in_bounds = sum(1 for n in N if lower_bound <= n <= upper_bound)
        bound_adherence = (time_in_bounds / len(N)) * 100
        st.metric("Time in Bounds", f"{bound_adherence:.1f}%")
        hashrate_volatility = np.std(N) / np.mean(N) * 100
        st.metric("Hashrate Volatility", f"{hashrate_volatility:.2f}%")
    
    # Export results option
    st.subheader("Export Results")
    results_df = pd.DataFrame({
        'Time_Step': list(range(len(N))),
        'Hashrate': N,
        'Block_Reward': P,
        'Exchange_Rate': exchange_rate_path + [exchange_rate_path[-1]] if len(exchange_rate_path) < len(N) else exchange_rate_path[:len(N)],
        'Efficiency': efficiency_path + [efficiency_path[-1]] if len(efficiency_path) < len(N) else efficiency_path[:len(N)],
        'Electricity_Cost': electricity_cost_path + [electricity_cost_path[-1]] if len(electricity_cost_path) < len(N) else electricity_cost_path[:len(N)]
    })
    
    # Convert to CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="blockchain_simulation_results.csv",
        mime="text/csv",
    )