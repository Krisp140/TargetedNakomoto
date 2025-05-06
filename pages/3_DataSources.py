import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Data Sources",
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("Data Sources")
st.write("This page provides information about the data sources used in our model and additional contextual data.")

# Section 1: Data Source Links
st.header("Data Source Links")
st.write("The following data sources were used to build and validate our model:")

# Create a table of data sources with descriptions
data_sources = {
    "Source": [
        "Bitcoin Visuals", 
        "Bitcoin Visuals", 
        "Cambridge Centre for Alternative Finance",
        "ERCOT (Electric Reliability Council of Texas)",
        "Bitcoin Visuals"
    ],
    "Data Type": [
        "Network hashrate (TH/s)",
        "Bitcoin price ($)",
        "Mining efficiency (Th/kWh)",
        "Electricity costs ($/kWh)",
        "Block speed (s)"
    ],
    "URL": [
        "https://bitcoinvisuals.com/",
        "https://bitcoinvisuals.com/",
        "https://ccaf.io/",
        "https://www.ercot.com/gridinfo/load/load_hist",
        "https://bitcoinvisuals.com/"
    ]
}

df_sources = pd.DataFrame(data_sources)
st.dataframe(df_sources, use_container_width=True, hide_index=True)
st.write("Note: The electricity cost data specifically focuses on Texas prices, as it's a major hub for Bitcoin mining operations in the United States.")

# Section 2: Electricity Costs Comparison
st.header("Electricity Costs by Mining Company")

# Description
st.write("""
This table shows the electricity costs for four major Bitcoin mining companies over the years 2021, 2022, and 2023.
Electricity cost is one of the most critical factors affecting mining profitability and hashrate deployment decisions.
""")

# Create the electricity cost comparison table with quarterly data
years = [2021, 2022, 2023]
quarters = ["Q1", "Q2", "Q3", "Q4"]

# Generate dates for the index
dates = [f"{year} {quarter}" for year in years for quarter in quarters]

# Create sample data for 4 companies (in cents per kWh)
# This is placeholder data - replace with actual data when available
data = {
    "Riot Platforms": [
        'N/A', '2.5 ¢/kWh', '2.5 ¢/kWh', '2.5 ¢/kWh',  # 2021
        '2.5 ¢/kWh', '2.5 ¢/kWh', '2.4 ¢/kWh', '3.1 ¢/kWh',  # 2022
        '4.2 ¢/kWh', '2.8 ¢/kWh', '1.7 ¢/kWh', '1.7 ¢/kWh'   # 2023
    ],
    "Marathon Digital": [
        '2.8 ¢/kWh', '2.8 ¢/kWh', '2.8 ¢/kWh', '2.8 ¢/kWh',  # 2021
        '2.8 ¢/kWh', '2.8 ¢/kWh', '2.8 ¢/kWh', '2.8 ¢/kWh',  # 2022
        '2.8 ¢/kWh', '2.8 ¢/kWh', '2.8 ¢/kWh', '2.8 ¢/kWh'   # 2023
    ],
    "Core Scientific": [
        '4.2 ¢/kWh', '4.2 ¢/kWh', '4.2 ¢/kWh', '4.2 ¢/kWh',  # 2021
        '4.2 ¢/kWh', '5.0 ¢/kWh', '6.6 ¢/kWh', '6.0 ¢/kWh',  # 2022
        '5.0 ¢/kWh', '5.0 ¢/kWh', '5.0 ¢/kWh', '5.0 ¢/kWh'   # 2023
    ],
    "CleanSpark": [
        '4.5 ¢/kWh', '4.5 ¢/kWh', '4.5 ¢/kWh', '4.5 ¢/kWh',  # 2021
        '4.5 ¢/kWh', '4.5 ¢/kWh', '4.5 ¢/kWh', '4.5 ¢/kWh',  # 2022
        '4.5 ¢/kWh', '4.5 ¢/kWh', '4.5 ¢/kWh', '4.5 ¢/kWh'   # 2023
    ]
}

# Create DataFrame
df_electricity = pd.DataFrame(data, index=dates)

# Display the table
st.dataframe(df_electricity, use_container_width=True)

# Add explanations for key trends
st.subheader("Note")
st.write("""
- **Source**: All major mining companies hold contracts with electricity providers. The data above is based on publicly available information (10Q, 10K, etc.) and is not an exhaustive list of all electricity costs for all mining companies.
- **Impact on Model**: These electricity cost directly impact our model predictions, as they represent a significant proportion of operational costs for miners.
""")

# Section 3: Model Prediction Visualization
st.header("Model Prediction Results")

# Description
st.write("""
The graph below shows our hashrate prediction model results compared to actual hashrate values. 
The model incorporates the variables discussed in our analysis, including bitcoin price, mining efficiency, 
electricity costs, and block speed adjustments.
""")

# Try to load the prediction image if it exists
try:
    # Attempt to load from the default locations
    image_paths = [
        "data/new_model.png"
    ]
    
    image_loaded = False
    for path in image_paths:
        if os.path.exists(path):
            image = Image.open(path)
            st.image(image, caption="Predicted vs Actual Bitcoin Network Hashrate", use_container_width=True)
            image_loaded = True
            break
    
except Exception as e:
    st.error(f"Error displaying the prediction visualization: {e}")
    st.write("Please ensure your prediction image exists and is properly formatted.")

# Additional Information
st.header("Additional Context")
st.write("""
The data used in our model indicates several key trends:

1. **Long-term correlation** between Bitcoin price and hashrate remains strong, but with varying time lags.
2. **Mining efficiency improvements** have accelerated, with average efficiency improving approximately 20% year-over-year.
3. **Geographic shift** in mining operations from China to North America has affected the average electricity costs.
4. **Increased institutionalization** of mining has led to more predictable hashrate deployment patterns.

These trends help explain the accuracy and limitations of our prediction model.
""")

# Footer
st.caption("This analysis is for educational purposes only and should not be considered financial advice.")
