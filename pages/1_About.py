import streamlit as st
from PIL import Image
import os
import pandas as pd
st.set_page_config(page_title="About - Hashrate Control Simulation", layout="wide", page_icon="💻")

st.title("About the Hashrate Control Simulation")

st.header("Overview")
st.write("""
This simulation demonstrates a novel approach to Bitcoin hashrate adjustment using control theory principles.
The model implements a feedback control system that aims to maintain the network hashrate within specified bounds
while adapting to changing market conditions. The estimation equation is based on equation 4 from https://arxiv.org/abs/2405.15089
""")
st.header("Key Formulas")

st.subheader("Original Hashrate Adjustment (with logs)(1)(2)")
st.latex(r"""
\widehat{N_{t}} = \alpha_1  +  \alpha_2 \widehat{e P_{t}} +  \alpha_3 \widehat{\eta} +  \alpha_4 \widehat{c} +  \alpha_5 \widehat{\Delta \text{T}}
""")
st.write("Where:")
st.markdown("""
- $N_{t}$ is the predicted hashrate
- $e$ is the btc/usd exchange rate
- $\eta$ is the mining efficiency
- $c$ is the electricity cost
- $\Delta {T}$ is the deviation in block speed from target block time (10 minutes)
- $t$ is epoch (2016 blocks)
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

st.subheader("New Hashrate Adjustment (with logs)(3)")
st.latex(r"""
\widehat{N_{t}} = \alpha_1  +  \alpha_2 \widehat{e P_{t}} +  \alpha_3 \widehat{\eta}
""")
st.write("Where:")
st.markdown("""
- $N_{t}$ is the predicted hashrate
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

st.subheader("1. Target Bounds")
st.markdown("""
The simulation maintains the hashrate within specified bounds: The mean hashrate can be chosen by the user
- **Target Hashrate**: The mean hashrate can be chosen by the user
- **Range**: The % range of the hashrate serves as the upper and lower bounds
""")
st.subheader("2. Control Parameters")
st.markdown("""
- **Tau (τ)**: The control parameter that determines the responsiveness of the system outside the bounds. 
  - Range: 0 to 1, symmetric to floor/ceiling
  - Higher values make the system more responsive but potentially less stable
  - Lower values make the system more stable but slower to respond

- **Gamma (γ)**: The control parameter that influences the strength of the correction inside the bounds.
  - Range: 0 to 1, symmetric to floor/ceiling
  - Lower values result in stronger corrections
  - Higher values result in gentler adjustments
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

- Network hashrate (TH/s) (https://bitcoinvisuals.com/)
- Bitcoin price ($) (https://bitcoinvisuals.com/)
- Mining efficiency (Th/kWh) (https://ccaf.io/)
- Electricity costs ($/kWh) (https://www.ercot.com/gridinfo/load/load_hist (Texas))
- Block speed (s) (https://bitcoinvisuals.com/)
""")

st.header("Footnotes")
st.markdown("""
(1) We tested several different lag structures for the hashrate adjustment model and found that using the current period was the most robust.
""")

st.markdown("""
**(2) OLS Regression Summary:**  
- Dependent Variable: log_hashrate_forward
- R-squared: 0.970
- Adj. R-squared: 0.970
- F-statistic: 7261
- No. Observations: 891
""")

regression_coef = pd.DataFrame({
    'Variable': ['constant', 'log_eP_forward', 'log_efficiency_forward', 
                'log_electricity_cost_forward', 'log_block_speed_change_forward'],
    'Coefficient': [44.9623, 0.4098, -1.4882, 0.2499, 0.0346],
    'Std Error': [0.404, 0.010, 0.015, 0.095, 0.003],
    't-value': [111.379, 42.317, -100.872, 2.639, 10.881],
    'P>|t|': [0.000, 0.000, 0.000, 0.008, 0.000],
    '95% CI Lower': [44.170, 0.391, -1.517, 0.064, 0.028],
    '95% CI Upper': [45.755, 0.429, -1.459, 0.436, 0.041]
})

st.dataframe(regression_coef)

st.markdown("""
**Statistical Tests:**
- Durbin-Watson: 0.254 (indicates positive autocorrelation)
- Condition Number: 2.67e+03 (might indicate multicollinearity)
""")

st.markdown("""
**(3) Regression Results with Selected Features:**
""")

st.markdown("""
**OLS Regression Summary:**  
- Dependent Variable: log_hashrate_forward
- R-squared: 0.966
- Adj. R-squared: 0.966
- F-statistic: 1.255e+04
- No. Observations: 892
""")

selected_regression_coef = pd.DataFrame({
    'Variable': ['constant', 'log_eP_forward', 'log_efficiency_forward'],
    'Coefficient': [45.6767, 0.4108, -1.5100],
    'Std Error': [0.163, 0.009, 0.011],
    't-value': [279.711, 47.316, -135.745],
    'P>|t|': [0.000, 0.000, 0.000],
    '95% CI Lower': [45.356, 0.394, -1.532],
    '95% CI Upper': [45.997, 0.428, -1.488]
})

st.dataframe(selected_regression_coef)

st.markdown("""
**Statistical Tests:**
- Durbin-Watson: 0.516 (improved, but still indicates autocorrelation)
- Condition Number: 974 (significantly improved from the original model)
""")