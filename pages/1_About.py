import streamlit as st
from PIL import Image
import os
st.set_page_config(page_title="About - Hashrate Control Simulation", layout="wide", page_icon="💻")

st.title("About the Hashrate Control Simulation")

st.header("Overview")
st.write("""
This simulation demonstrates a novel approach to Bitcoin hashrate adjustment using control theory principles.
The model implements a feedback control system that aims to maintain the network hashrate within specified bounds
while adapting to changing market conditions.
""")
st.header("Key Formulas")

st.subheader("Original Hashrate Adjustment (with logs)")
st.latex(r"""
\widehat{N_{t+1}} = \alpha_1  +  \alpha_2 \widehat{e P_{t+1}} +  \alpha_3 \widehat{\eta} +  \alpha_4 \widehat{c} +  \alpha_5 \widehat{\Delta \text{T+1}}
""")
st.write("Where:")
st.markdown("""
- $N_{t+1}$ is the predicted hashrate
- $e$ is the btc/usd exchange rate
- $\eta$ is the mining efficiency
- $c$ is the electricity cost
- $\Delta {T+1}$ is the change in block speed
""")

st.write("Resultant Prediction:")
image = Image.open(os.path.join("data", "old_model.png"))
st.image(image, caption="Original Hashrate Prediction")

st.write("However this model has a condition number of 235, indicating potential issues with multicollinearity. Therefore we run a lasso regression and find this:")
image = Image.open(os.path.join("data", "reg_path.png"))
st.image(image, caption="Lasso Regression")

st.write("Based on the Lasso results, we remove the electricity and block speed variables from the model.")
st.write("Resultant Prediction:")
image = Image.open(os.path.join("data", "new_model.png"))
st.image(image, caption="New Hashrate Prediction")

st.subheader("New Hashrate Adjustment (with logs)")
st.latex(r"""
\widehat{N_{t+1}} = \alpha_1  +  \alpha_2 \widehat{e P_{t+1}} +  \alpha_3 \widehat{\eta}
""")
st.write("Where:")
st.markdown("""
- $N_{t+1}$ is the predicted hashrate
- $e$ is the btc/usd exchange rate
- $\eta$ is the mining efficiency
""")
st.write("And the alphas are:")
st.latex(r"""
\alpha_1 = 45.6767\\
\alpha_2 = 0.4108\\
\alpha_3 = -1.5100
""")
st.write("And the condition number changes: 235.42 → 60.02")

st.header("Key Components")

st.subheader("1. Control Parameters")
st.markdown("""
- **Tau (τ)**: The control parameter that determines the responsiveness of the system. 
  - Range: 0 to 1
  - Higher values make the system more responsive but potentially less stable
  - Lower values make the system more stable but slower to respond

- **Gamma (γ)**: The control parameter that influences the strength of the correction.
  - Range: 0 to 1
  - Lower values result in stronger corrections
  - Higher values result in gentler adjustments
""")

st.subheader("2. Target Bounds")
st.markdown("""
The simulation maintains the hashrate within specified bounds:
- **Upper Bound**: Maximum acceptable hashrate (as % of mean actual hashrate)
- **Lower Bound**: Minimum acceptable hashrate (as % of mean actual hashrate)
""")



st.header("Performance Metrics")
st.markdown("""
The simulation evaluates performance using several metrics:
1. **Hashrate Volatility**: Standard deviation of hashrate divided by mean
2. **Time in Bounds**: Percentage of time the hashrate stays within target bounds
""")

st.header("Data Sources")
st.markdown("""
The simulation uses historical Bitcoin network data including:
- Network hashrate (TH/s) (from Blockchain.info)
- Bitcoin price ($) (from CoinGecko)
- Mining efficiency (Th/kWh) (from ccaf.io)
- Electricity costs ($/kWh) (from ERCOT (Texas))
""")