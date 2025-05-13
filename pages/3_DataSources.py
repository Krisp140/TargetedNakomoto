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

# Create sources data
sources = [
    {
        "name": "Bitcoin Visuals",
        "data_type": "Network hashrate (H/s)",
        "description": "Derived in section 4.1 of paper https://arxiv.org/abs/2405.15089.",
        "url": "https://bitcoinvisuals.com/chain-hash-rate"
    },
    {
        "name": "CoinMarketCap",
        "data_type": "Average Daily Bitcoin Price ($)",
        "description": "Weighted average across several exchanges.",
        "url": "https://support.coinmarketcap.com/hc/en-us/articles/360015968632-How-are-prices-calculated-on-CoinMarketCap?utm_source=chatgpt.com"
    },
    {
        "name": "Cambridge Centre for Alternative Finance",
        "data_type": "Mining efficiency (Th/kWh)",
        "description": "Bottom of page. Received direct data from inquiry. Estimate derived from specifications of over a 100 different device types.",
        "url": "https://ccaf.io/cbnsi/cbeci"
    },
    {
        "name": "Bitcoin Visuals",
        "data_type": "Block reward ($)",
        "description": "Run by Bitcoin Visuals node.",
        "url": "https://bitcoinvisuals.com/chain-block-reward-day"
    },
    {
        "name": "Bitcoin Visuals",
        "data_type": "Block speed (s)",
        "description": "Daily median time between bitcoin blocks. Run by Bitcoin Visuals node.",
        "url": "https://bitcoinvisuals.com/chain-speed"
    }
]

# Display sources with quantity first, then source and explanation
for i, source in enumerate(sources):
    st.markdown(f"**{i+1}. {source['data_type']}** - [{source['name']}]({source['url']}). {source['description']}")

st.write("Note: We initially used electricity cost data specifically focuses on Texas prices, but decided to use the SEC EDGAR data as it is more indicative of the average electricity cost for all mining companies due to private contracts.")

# Section 2: Electricity Costs Comparison
st.header("Electricity Costs by Mining Company")

# Description
st.write("""
This table shows the electricity costs for four major Bitcoin mining companies over the years 2021, 2022, and 2023.
Electricity cost is one of the most critical factors affecting mining profitability and hashrate deployment decisions.
""")

# Company SEC EDGAR URLs
company_urls = {
    "Riot Platforms": "https://www.sec.gov/edgar/browse/?CIK=1167419",
    "Marathon Digital": "https://www.sec.gov/edgar/browse/?CIK=1507605", 
    "Core Scientific": "https://www.sec.gov/edgar/browse/?CIK=1839341",
    "CleanSpark": "https://www.sec.gov/edgar/browse/?CIK=827876"
}

# Create a static HTML table with clickable company names
html_table = """
<style>
.electricity-table {{
    width: 100%;
    border-collapse: collapse;
    background-color: #23272b;
}}
.electricity-table th, .electricity-table td {{
    border: 1px solid #444;
    padding: 8px;
    text-align: left;
    color: #f1f1f1;
}}
.electricity-table th {{
    background-color: #343a40;
    font-weight: bold;
    color: #fff;
}}
.electricity-table tr:nth-child(even) {{
    background-color: #2c3035;
}}
.company-link {{
    color: #4da3ff;
    text-decoration: underline;
}}
</style>

<table class="electricity-table">
    <tr>
        <th>Company</th>
        <th>2021 Q1</th>
        <th>2021 Q2</th>
        <th>2021 Q3</th>
        <th>2021 Q4</th>
        <th>2022 Q1</th>
        <th>2022 Q2</th>
        <th>2022 Q3</th>
        <th>2022 Q4</th>
        <th>2023 Q1</th>
        <th>2023 Q2</th>
        <th>2023 Q3</th>
        <th>2023 Q4</th>
    </tr>
    <tr>
        <td><a href="{0}" class="company-link" target="_blank">Riot Platforms</a></td>
        <td>N/A</td>
        <td>2.5 ¢/kWh</td>
        <td>2.5 ¢/kWh</td>
        <td>2.5 ¢/kWh</td>
        <td>2.5 ¢/kWh</td>
        <td>2.5 ¢/kWh</td>
        <td>2.4 ¢/kWh</td>
        <td>3.1 ¢/kWh</td>
        <td>4.2 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>1.7 ¢/kWh</td>
        <td>1.7 ¢/kWh</td>
    </tr>
    <tr>
        <td><a href="{1}" class="company-link" target="_blank">Marathon Digital</a></td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
        <td>2.8 ¢/kWh</td>
    </tr>
    <tr>
        <td><a href="{2}" class="company-link" target="_blank">Core Scientific</a></td>
        <td>4.2 ¢/kWh</td>
        <td>4.2 ¢/kWh</td>
        <td>4.2 ¢/kWh</td>
        <td>4.2 ¢/kWh</td>
        <td>4.2 ¢/kWh</td>
        <td>5.0 ¢/kWh</td>
        <td>6.6 ¢/kWh</td>
        <td>6.0 ¢/kWh</td>
        <td>5.0 ¢/kWh</td>
        <td>5.0 ¢/kWh</td>
        <td>5.0 ¢/kWh</td>
        <td>5.0 ¢/kWh</td>
    </tr>
    <tr>
        <td><a href="{3}" class="company-link" target="_blank">CleanSpark</a></td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
        <td>4.5 ¢/kWh</td>
    </tr>
    <tr style="background-color: #1e3a5f; font-weight: bold;">
        <td>Average</td>
        <td>3.8 ¢/kWh</td>
        <td>3.5 ¢/kWh</td>
        <td>3.5 ¢/kWh</td>
        <td>3.5 ¢/kWh</td>
        <td>3.5 ¢/kWh</td>
        <td>3.7 ¢/kWh</td>
        <td>4.1 ¢/kWh</td>
        <td>4.1 ¢/kWh</td>
        <td>4.1 ¢/kWh</td>
        <td>3.8 ¢/kWh</td>
        <td>3.5 ¢/kWh</td>
        <td>3.5 ¢/kWh</td>
    </tr>
</table>
"""

formatted_table = html_table.format(
    company_urls["Riot Platforms"],
    company_urls["Marathon Digital"],
    company_urls["Core Scientific"],
    company_urls["CleanSpark"]
)

# Display the table using st.markdown with unsafe_allow_html=True
st.markdown(formatted_table, unsafe_allow_html=True)

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
The graph below shows our hashrate prediction model results compared to estimated hashrate values. 
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
            st.image(image, caption="Actual vs Model Predicted Hashrate")
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

These trends help explain the accuracy and limitations of our prediction model.
""")

# Footer