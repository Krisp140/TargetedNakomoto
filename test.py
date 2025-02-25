import requests
import json
from PIL import Image
from io import BytesIO
import pandas as pd

base_url = 'http://localhost:5000'

df = pd.read_csv("bitcoin_mining_data_daily.csv")
print(float(df['hashrate'].mean()))
payload = {
    "initial_conditions": {
        "hashrate": float(df['hashrate'].iloc[0]),  # Initial hashrate from data
        "block_reward": float(df['block_reward'].iloc[0])  # Initial block reward from data
    },
    "target_bounds": {
        "upper_bound": float(df['hashrate'].mean() * 1.2),  # 10% above mean hashrate
        "lower_bound": float(df['hashrate'].mean() * 0.8)   # 10% below mean hashrate
    },
    "control_params": {
        "tau": 0.5,    # Adjustment rate
        "gamma": 0.01  # Mining difficulty parameter
    },
    "time_paths": {
        "exchange_rate": df['price'].tolist(),  # Real exchange rate path
        "efficiency": df['mining_cost'].tolist(),
        "electricity_cost": df['electricity_price'].tolist()       # Real mining cost path
    }
}

"""payload = {
    "initial_conditions": {
        "hashrate": 739000000,  # Initial hashrate
        "block_reward": 3.125  # Initial block reward
    },
    "target_bounds": {
        "upper_bound": 500000000,  # Upper hashrate bound
        "lower_bound": 339000000    # Lower hashrate bound
    },
    "control_params": {
        "tau": 0.5,    # Adjustment rate
        "gamma": 0.01   # Mining difficulty parameter
    },
    "time_paths": {
        "exchange_rate": [90000, 90000, 90000,90000,100000,90000,90000,90000,90000,90000],  # Example BTC/USD prices
        "mining_cost": [.0012, .0012, .0012,.0012,.0012,.0012,.0012,.0012,.0012,.0012]  # Example mining costs
    }
}"""

# Send the POST request
choice = input("Do you want to simulate or visualize? (s/v): ")
if choice == "s":
    url = 'http://localhost:5000/simulate'
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    try:
        print("JSON Response:", json.dumps(response.json(), indent=4))
        
    except requests.exceptions.JSONDecodeError as e:
        print("Failed to decode JSON response:", e)
else:
    metric = input("Do you want to visualize hashrate or mining cost? (h/m): ")
    if metric == "h":
        url = 'http://localhost:5000/visualize/hashrate'
    else:
        url = 'http://localhost:5000/visualize/mining_cost'
    response = requests.post(url, json=payload)
    img = Image.open(BytesIO(response.content))
    img.show()
    print("Status Code:", response.status_code)