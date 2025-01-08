import requests
import json
from PIL import Image
from io import BytesIO

base_url = 'http://localhost:5000'

payload = {
    "initial_conditions": {
        "hashrate": 10000,  # Initial hashrate
        "block_reward": 6.25  # Initial block reward
    },
    "target_bounds": {
        "upper_bound": 6.25,  # Upper hashrate bound
        "lower_bound": 6.25    # Lower hashrate bound
    },
    "control_params": {
        "tau": 0.1,    # Adjustment rate
        "gamma": 0.01   # Mining difficulty parameter
    },
    "time_paths": {
        "exchange_rate": [20000, 21000, 22000, 21500, 20800],  # Example BTC/USD prices
        "mining_cost": [10, 10, 11, 11, 10]  # Example mining costs
    }
}

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