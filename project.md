# Nakamoto Blockchain Simulation Project

This document outlines the structure and components of the Nakamoto Blockchain Simulation project, a platform for analyzing blockchain behavior and hashrate control mechanisms using control theory principles.

## Project Overview

The Nakamoto Blockchain Simulation is a research project focused on implementing and testing a novel approach to Bitcoin hashrate adjustment using control theory. The model implements a feedback control system that aims to maintain the network hashrate within specified bounds while adapting to changing market conditions.

The simulation is built as a web application using Streamlit for the frontend. It allows users to experiment with different control parameters and see how they affect the blockchain's hashrate and block rewards over time.

## Project Structure

```
Nakamoto/
├── src/                  # Core code modules
│   ├── models/           # Core blockchain models
│   ├── api/              # Flask API endpoints (unused)
│   └── simulation/       # Simulation engine
├── tests/                # Test files
├── data/                 # CSV and JSON data files
├── static/               # Images and static assets
├── pages/                # Streamlit pages
├── app.py                # Main Streamlit application (duplicate, consider removing)
├── 0_About.py            # About page (duplicate, consider removing)
└── requirements.txt      # Project dependencies
```

## Components

### Core Modules (src/)

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

#### API Routes (`src/api/routes.py`)

A Flask-based REST API implementation for running simulations, but currently unused in the application:

- `/simulate` endpoint: Runs a simulation with provided parameters
- `/visualize/<metric>` endpoint: Generates visualizations for different metrics
- **Status**: Unused component, Streamlit pages call simulation functions directly

### User Interface

#### Main App (`app.py` and `0_About.py`)

These files are duplicates of `pages/0_About.py` and should be consolidated:

- Sets up the page configuration
- Provides an overview of the hashrate control simulation
- Includes key formulas and explanations of the model
- Displays regression results and statistical tests
- **Status**: Redundant components, standardize on `pages/0_About.py`

#### Streamlit Pages (`pages/`)

- **About (`pages/0_About.py`)**: The main About page
  - Contains overview information and project description
  - Includes navigation buttons to other pages
  - Most complete implementation of the About page

- **Simulation (`pages/1_Simulation.py`)**: The main simulation interface
  - Allows users to set control parameters (tau, gamma)
  - Defines target hashrate bounds
  - Runs simulations and visualizes results
  - Provides electricity cost estimates

- **Playground (`pages/2_Playground.py`)**: Interactive interface for experimenting
  - Allows more detailed parameter adjustments
  - Supports comparing different simulation scenarios

- **Data Sources (`pages/3_DataSources.py`)**: Documents the data sources used
  - Displays historical Bitcoin network data
  - Provides information about data collection methodology

### Data Handling

#### Data Files (`data/`)

Contains various data files used by the simulation:

- Historical blockchain data (hashrate, price, block speed)
- Mining efficiency data
- Electricity price data
- Predicted hashrate data from different models
- Data processing scripts (regression_analysis.py, data_merger.py)
- Visualization assets (PNG files for charts)

#### Data Collection Scripts

- `data_collection.py`: Script for gathering blockchain data
- `data_collection_2.py`: Alternative data collection script
- `data_merger.py`: Script for merging different data sources
- `regression_analysis.py`: Script for analyzing relationships between variables

### Testing

- `tests/deployment_checks.py`: Script to verify deployment readiness
  - Checks environment, requirements, and critical files
  - Verifies Streamlit configuration

## Unused or Redundant Components

Based on the codebase analysis, the following components appear to be redundant or no longer needed:

1. **0_About.py** (root directory): Duplicate of `pages/0_About.py`
2. **app.py**: Duplicate of `pages/0_About.py` with slight differences
3. **src/api/**: The Flask API is implemented but not used in the application
4. **tests/api.py**: Referenced in deployment checks but not found in the codebase
5. **Some data collection scripts may be redundant** as they perform similar functions

## Development Status

The project appears to be under active development, as evidenced by the TODO notes in notes.txt:

- Implementing electricity cost and consumption tracking
- Adding sliding window functionality to the playground
- Adjusting cost calculations per block
- Implementing fee handling
- Creating bounds for electricity cost and consumption along with hashrate

## Dependencies

The main dependencies for this project include:

- Streamlit: For the web interface
- NumPy/Pandas: For data handling and analysis
- Matplotlib: For visualization
- PIL: For image processing

## Setup and Running

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the Streamlit interface:
```bash
streamlit run pages/0_About.py
```

## Recommendations for Improvement

1. **Remove redundant About files**: Delete the root `0_About.py` and `app.py`, keeping only `pages/0_About.py`
   - The `pages/0_About.py` version is the most complete with navigation buttons and proper styling
   - Update `.streamlit/config.toml` to use `pages/0_About.py` as the entry point

2. **Consider removing or integrating the Flask API**: Either remove the unused Flask API or integrate it with the Streamlit frontend

3. **Fix the referenced but missing api.py file** in deployment_checks.py

4. **Consolidate data collection scripts** to avoid duplication

5. **Complete the TODOs** listed in notes.txt

6. **Improve test coverage** with more comprehensive tests

7. **Fix cost calculation per block** as mentioned in notes.txt

8. **Standardize code structure** across different modules for better maintainability 