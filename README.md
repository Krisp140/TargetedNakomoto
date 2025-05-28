# Targeted Nakamoto: Bitcoin Hashrate Control Simulator

A simulation platform for analyzing Bitcoin blockchain behavior and implementing hashrate control mechanisms using control theory principles.

## Project Overview

The Targeted Nakamoto project is a research initiative that implements and tests a novel approach to Bitcoin hashrate adjustment using control theory. The model implements a feedback control system that aims to maintain the network hashrate within specified bounds while adapting to changing market conditions such as Bitcoin price, mining efficiency, and electricity costs.

Built as a web application using Streamlit, the simulator allows users to experiment with different control parameters (tau, gamma) and observe how they affect the blockchain's hashrate stability and block rewards over time.

## Project Structure

```
TargetedNakamoto/
├── src/                  # Core code modules
│   ├── models/           # Core blockchain models
│   ├── simulation/       # Simulation engine
│   └── api/              # Flask API endpoints (not currently used)
├── pages/                # Streamlit pages (main application UI)
│   ├── 0_About.py        # Project overview and introduction
│   ├── 1_Simulation.py   # Main simulation interface
│   ├── 2_Playground.py   # Interactive parameter testing
│   └── 3_DataSources.py  # Data source documentation
├── data/                 # CSV and JSON data files
├── static/               # Images and static assets
├── tests/                # Test files
└── requirements.txt      # Project dependencies
```

## Components

### Core Modules

#### Blockchain Model (`src/models/blockchain.py`)

The blockchain model implements the core logic for the simulation:

- **Epoch Class**: Represents a blockchain epoch (a period of time with consistent mining difficulty)
  - Stores rewards and hashrates for blocks within an epoch
  - Calculates median block rewards and average hashrates

- **Blockchain Class**: Implements the blockchain simulation
  - Adjusts rewards based on control parameters (tau, gamma)
  - Calculates new hashrates based on economic factors
  - Handles epoch transitions and difficulty adjustments

#### Simulation Engine (`src/simulation/engine.py`)

The simulation engine runs the blockchain simulation with given parameters:

- **run_simulation function**: Executes the simulation over multiple time steps
  - Takes parameters for tau, gamma, upper/lower bounds, and time paths
  - Returns time series data for hashrate, block rewards, etc.
  - Calculates performance statistics (volatility, bound adherence)

### User Interface

The application has a multi-page Streamlit interface:

- **About (`pages/0_About.py`)**: The main landing page
  - Provides overview information and project description
  - Includes key formulas and explanations of the model
  - Displays navigation buttons to other pages

- **Simulation (`pages/1_Simulation.py`)**: The main simulation interface
  - Allows users to set control parameters (tau, gamma)
  - Defines target hashrate bounds or electricity cost bounds
  - Runs simulations and visualizes results
  - Provides electricity cost estimates and statistical analysis

- **Playground (`pages/2_Playground.py`)**: Interactive interface for experimentation
  - Allows more detailed parameter adjustments
  - Supports custom time paths for exchange rate, efficiency, and electricity cost
  - Enables comparison of different simulation scenarios
  - Includes time window selection and pattern creation tools

- **Data Sources (`pages/3_DataSources.py`)**: Documents the data sources
  - Displays historical Bitcoin network data
  - Provides information about data collection methodology
  - Shows mining company information and electricity costs

### Data Files

The `data/` directory contains various datasets used by the simulation:

- Historical blockchain data (hashrate, price, block speed)
- Mining efficiency data
- Electricity price data
- Predicted hashrate data from different models

## Setup and Running

### Prerequisites

- Python 3.8+ installed
- Git (to clone the repository)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TargetedNakamoto.git
cd TargetedNakamoto
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# OR
.\.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the Streamlit interface:
```bash
streamlit run pages/app.py
```

The application should open automatically in your default web browser, typically at `http://localhost:8501`.

## Using the Simulator

1. **Setting Control Parameters**:
   - Adjust tau (responsiveness outside bounds) and gamma (responsiveness within bounds)
   - Set target hashrate or electricity cost bounds

2. **Running Simulations**:
   - Click "Run Simulation" to see the effects of your parameters
   - Review statistical results and performance metrics

3. **Parameter Sweep Analysis**:
   - Run parameter sweep to find optimal tau and gamma values
   - Visualize the results in 3D surface plots

4. **Creating Custom Scenarios**:
   - Use the Playground to create custom time paths
   - Test how the system responds to different economic conditions

## Development

When making changes to the codebase:

1. Ensure all tests pass:
```bash
python -m pytest tests/
```

2. Follow the existing code style and conventions

3. Update documentation when adding new features

## Credits

Developed by ANON at ANON.

## License

MIT