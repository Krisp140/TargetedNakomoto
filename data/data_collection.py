import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import json

class DataCollector:
    def __init__(self):
        # Glassnode API key - you'll need to sign up at https://glassnode.com
        self.GLASSNODE_API_KEY = 'YOUR_API_KEY'
        
        # CoinGecko API doesn't require a key
        self.COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'

    def get_hashrate_data(self):
        """
        Fetch Bitcoin hashrate data from Blockchain.info and reformat into 2-week intervals
        Returns hashrate in TH/s aligned with Bitcoin's difficulty adjustment periods
        """
        url = "https://api.blockchain.info/charts/hash-rate"
        params = {
            "timespan": "1year",
            "format": "json",
            "interval": "24h"  # Get daily data first, then we'll aggregate
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df_hashrate = pd.DataFrame(data['values'])
            df_hashrate.columns = ['timestamp', 'hashrate']
            df_hashrate['date'] = pd.to_datetime(df_hashrate['timestamp'], unit='s')
            
            print("\nHashrate Statistics:")
            print(df_hashrate['hashrate'].describe())
            
            return df_hashrate[['date', 'hashrate']]
        else:
            raise Exception(f"Failed to fetch hashrate data: {response.status_code}")
   

    def get_price_data(self):
        """
        Fetch Bitcoin price history from CoinGecko
        """
        url = f'{self.COINGECKO_BASE_URL}/coins/bitcoin/market_chart'
        params = {
            'vs_currency': 'usd',
            'days': '365',  # Last year of data
            'interval': 'daily'  # Bi-weekly interval to match block times
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df_price['date'] = pd.to_datetime(df_price['timestamp'], unit='ms')
            # Set date as index for resampling
            df_price = df_price.set_index('date')
            
            # Resample to 2-week intervals using mean price
            df_price_2w = df_price['price'].resample('2W').mean().reset_index()
            print(df_price.keys())
            print("\nPrice Statistics:")
            print(df_price['price'].describe())
            
            return df_price.reset_index().rename(columns={'index': 'date'})
        else:
            raise Exception(f"Failed to fetch price data: {response.status_code}")

    def get_block_reward_data(self):
        """
        Calculate historical block rewards based on halving events
        """
        # Create date range
        dates = pd.date_range(start='2024-01-21', end='2025-01-21', freq='D')
        
        with open('transaction-fees.json', 'r') as f:
            fees_data = json.load(f)
        
        fees_df = pd.DataFrame(fees_data['transaction-fees'])
        fees_df['date'] = pd.to_datetime(fees_df['x'], unit='ms')
        fees_df['fees'] = fees_df['y']
        
        # Initial block reward
        rewards = []
        fees = []
        for date in dates:
            # Bitcoin halving occurs every 210,000 blocks (approximately every 4 years)
            # Last halving was on May 11, 2020 - reward went from 12.5 to 6.25 BTC
            if date < pd.Timestamp('2024-04-19'):
                reward = 6.25
            else:
                reward = 3.125

            date_fees = fees_df[fees_df['date'].dt.date == date.date()]['fees'].values
            fee = float(date_fees[0]) if len(date_fees) > 0 else 0
            fees.append(fee)
            rewards.append(reward+fee)

        return pd.DataFrame({
            'date': dates,
            'block_reward': rewards,
            'fees': fees
        })

    def get_mining_cost_data(self):
        """
        Estimate mining costs based on hardcoded electricity prices and mining efficiency
        """
        # Hardcoded monthly commercial electricity prices (cents per kWh)
        monthly_prices = {
            '2023-12': 12.24,
            '2024-01': 12.59,
            '2024-02': 12.75,
            '2024-03': 12.73,
            '2024-04': 12.63,
            '2024-05': 12.48,
            '2024-06': 13.07,
            '2024-07': 13.58,
            '2024-08': 13.39,
            '2024-09': 13.47,
            '2024-10': 13.20,
            '2024-11': 12.22,
            '2024-12': 12.64
        }
        
        # Get date range for our analysis
        dates = pd.date_range(start='2024-01-21', end='2025-01-21', freq='D')
        
        # Create daily prices DataFrame - modify this part
        daily_prices = pd.DataFrame({
            'date': dates,
            'electricity_price': None
        })
        
        # Fill in daily prices based on month
        for date in dates:
            month_key = date.strftime('%Y-%m')
            if month_key in monthly_prices:
                daily_prices.loc[daily_prices['date'] == date, 'electricity_price'] = monthly_prices[month_key] / 100
            else:
                daily_prices.loc[daily_prices['date'] == date, 'electricity_price'] = monthly_prices['2024-12'] / 100

        # Read mining efficiency data
        efficiency_data = pd.read_csv('efficiency_data.txt', header=None)
        efficiency_data.columns = ['date', 'lower_efficiency', 'estimated_efficiency', 'upper_efficiency']
        efficiency_data['date'] = pd.to_datetime(efficiency_data['date'])
        
        # Convert J/Th to kWh/Th (1 kWh = 3,600,000 Joules)
        efficiency_data['power_usage_kwh'] = 1/(efficiency_data['estimated_efficiency'] / 3600000)
        
        # Merge efficiency data with daily prices - modified merge
        merged_data = pd.merge(daily_prices, 
                              efficiency_data[['date', 'power_usage_kwh']], 
                              on='date',  # Changed to merge on 'date' column
                              how='left')
        
        # Calculate mining cost per TH/s using commercial electricity prices
        #merged_data['mining_cost'] = merged_data['electricity_price'] * merged_data['power_usage_kwh'] * 24  # Daily cost
        merged_data['mining_cost'] = merged_data['power_usage_kwh'] * 24  # Daily cost

        print(merged_data)
        df_costs = pd.DataFrame({
            'date': merged_data['date'],
            'mining_cost': merged_data['mining_cost']
        })
        
        print("\nMining Cost Statistics:")
        print(df_costs['mining_cost'].describe())
        
        return df_costs
    def get_electricity_cost_data(self):
        """
        Estimate mining costs based on hardcoded electricity prices and mining efficiency
        """
        # Hardcoded monthly commercial electricity prices (cents per kWh)
        monthly_prices = {
            '2023-12': 12.24,
            '2024-01': 12.59,
            '2024-02': 12.75,
            '2024-03': 12.73,
            '2024-04': 12.63,
            '2024-05': 12.48,
            '2024-06': 13.07,
            '2024-07': 13.58,
            '2024-08': 13.39,
            '2024-09': 13.47,
            '2024-10': 13.20,
            '2024-11': 12.22,
            '2024-12': 12.64
        }
        # Get date range for our analysis
        dates = pd.date_range(start='2024-01-21', end='2025-01-21', freq='D')
        
        # Create daily prices DataFrame - modify this part
        daily_prices = pd.DataFrame({
            'date': dates,
            'electricity_price': None
        })
                # Fill in daily prices based on month
        for date in dates:
            month_key = date.strftime('%Y-%m')
            if month_key in monthly_prices:
                daily_prices.loc[daily_prices['date'] == date, 'electricity_price'] = monthly_prices[month_key] / 100
            else:
                daily_prices.loc[daily_prices['date'] == date, 'electricity_price'] = monthly_prices['2024-12'] / 100
        return daily_prices



def prepare_dataset():
    collector = DataCollector()
    
    print("Fetching hashrate data...")
    df_hashrate = collector.get_hashrate_data()
    print("\nFetching price data...")
    df_price = collector.get_price_data()
    print("\nCalculating block rewards...")
    df_rewards = collector.get_block_reward_data()
    print("\nEstimating mining costs...")
    df_costs = collector.get_mining_cost_data()
    print("\nEstimating electricity costs...")
    df_electricity_costs = collector.get_electricity_cost_data()
    
    # Ensure date columns are datetime type
    df_hashrate['date'] = pd.to_datetime(df_hashrate['date'])
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_rewards['date'] = pd.to_datetime(df_rewards['date'])
    df_costs['date'] = pd.to_datetime(df_costs['date'])
    df_electricity_costs['date'] = pd.to_datetime(df_electricity_costs['date'])
     # Round all dates to start of day to ensure matching
    df_hashrate['date'] = df_hashrate['date'].dt.floor('D')
    df_price['date'] = df_price['date'].dt.floor('D')
    df_rewards['date'] = df_rewards['date'].dt.floor('D')
    df_costs['date'] = df_costs['date'].dt.floor('D')
    df_electricity_costs['date'] = df_electricity_costs['date'].dt.floor('D')
    # Remove any duplicate dates
    df_hashrate = df_hashrate.drop_duplicates('date')
    df_price = df_price.drop_duplicates('date')
    df_rewards = df_rewards.drop_duplicates('date')
    df_costs = df_costs.drop_duplicates('date')
    df_electricity_costs = df_electricity_costs.drop_duplicates('date')
    df_final = pd.merge(df_hashrate, df_price, on='date', how='inner')
    df_final = pd.merge(df_final, df_rewards, on='date', how='inner')
    df_final = pd.merge(df_final, df_costs, on='date', how='inner') 
    df_final = pd.merge(df_final, df_electricity_costs, on='date', how='inner')
    df_final = df_final.sort_values(by='date')
    # Verify no missing values
    print("\nMissing values check:")
    print(df_final.isnull().sum())
    
    # Save to CSV
    df_final.to_csv('bitcoin_mining_data_daily.csv', index=False)
    print(f"\nData saved to bitcoin_mining_data_daily.csv")
    print(f"Total records: {len(df_final)}")
    
    # Print sample of final dataset
    print("\nSample of final dataset:")
    print(df_final.head())
    
    return df_final

if __name__ == "__main__":
    prepare_dataset() 