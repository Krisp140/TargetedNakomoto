import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline

class DataCollector:
    def __init__(self):
        self.GLASSNODE_API_KEY = 'YOUR_API_KEY'
        self.COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'
        self.BLOCKS_PER_DAY = 144  # Average number of Bitcoin blocks per day

    def get_hashrate_data(self):
        """
        Fetch Bitcoin hashrate data from Blockchain.info
        Returns block-by-block hashrate in TH/s
        """
        url = "https://api.blockchain.info/charts/hash-rate"
        params = {
            "timespan": "1year",
            "format": "json",
            "interval": "24h"  # Get daily data first, then interpolate
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df_hashrate = pd.DataFrame(data['values'])
            df_hashrate.columns = ['timestamp', 'hashrate']
            df_hashrate['date'] = pd.to_datetime(df_hashrate['timestamp'], unit='s')
            
            # Create block timestamps (144 blocks per day)
            start_date = df_hashrate['date'].min()
            end_date = df_hashrate['date'].max()
            total_days = (end_date - start_date).days
            total_blocks = total_days * self.BLOCKS_PER_DAY
            
            # Create block-level timestamps
            block_timestamps = pd.date_range(
                start=start_date,
                end=end_date,
                periods=total_blocks
            )
            
            # Interpolate hashrate to block level
            df_hashrate = df_hashrate.set_index('date')
            df_block_hashrate = pd.DataFrame(index=block_timestamps)
            df_block_hashrate['hashrate'] = np.interp(
                x=block_timestamps.astype(np.int64) // 10**9,
                xp=df_hashrate.index.astype(np.int64) // 10**9,
                fp=df_hashrate['hashrate']
            )
            
            return df_block_hashrate.reset_index().rename(columns={'index': 'date'})
        else:
            raise Exception(f"Failed to fetch hashrate data: {response.status_code}")

    def get_price_data(self):
        """
        Fetch Bitcoin price history from CoinGecko and interpolate to block level
        """
        url = f'{self.COINGECKO_BASE_URL}/coins/bitcoin/market_chart'
        params = {
            'vs_currency': 'usd',
            'days': '365',
            'interval': 'daily'  # Changed to daily as hourly isn't supported
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df_price['date'] = pd.to_datetime(df_price['timestamp'], unit='ms')
            df_price = df_price.set_index('date')
            
            # Create block timestamps
            start_date = df_price.index.min()
            end_date = df_price.index.max()
            total_days = (end_date - start_date).days
            total_blocks = total_days * self.BLOCKS_PER_DAY
            
            # Create block-level timestamps
            block_timestamps = pd.date_range(
                start=start_date,
                end=end_date,
                periods=total_blocks
            )
            
            # Use cubic spline interpolation for smoother price transitions between daily points
            cs = CubicSpline(
                df_price.index.astype(np.int64) // 10**9,
                df_price['price']
            )
            
            # Interpolate price to block level using the cubic spline
            df_block_price = pd.DataFrame(index=block_timestamps)
            df_block_price['price'] = cs(block_timestamps.astype(np.int64) // 10**9)
            
            # Ensure no negative prices from interpolation
            df_block_price['price'] = df_block_price['price'].clip(lower=0)
            
            return df_block_price.reset_index().rename(columns={'index': 'date'})
        else:
            raise Exception(f"Failed to fetch price data: {response.status_code}")

    def get_block_reward_data(self):
        """
        Calculate block rewards based on block height
        """
        # Get current block height
        url = "https://blockchain.info/q/getblockcount"
        current_height = int(requests.get(url).text)
        
        # Calculate rewards for each block
        start_height = current_height - (365 * self.BLOCKS_PER_DAY)  # Last year of blocks
        block_heights = range(start_height, current_height + 1)
        
        rewards = []
        dates = []
        
        for height in block_heights:
            # Calculate reward based on halving epochs
            halving_epoch = height // 210000
            reward = 50.0 / (2 ** halving_epoch)
            
            # Estimate timestamp for this block
            # Note: This is an approximation, you might want to fetch actual block timestamps
            days_since_start = (height - start_height) / self.BLOCKS_PER_DAY
            date = pd.Timestamp.now() - pd.Timedelta(days=365-days_since_start)
            
            rewards.append(reward)
            dates.append(date)
        
        return pd.DataFrame({
            'date': dates,
            'block_reward': rewards
        })

    def get_mining_cost_data(self):
        """
        Estimate mining costs at block level
        """
        # Create block timestamps
        dates = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=365),
            end=pd.Timestamp.now(),
            periods=365 * self.BLOCKS_PER_DAY
        )
        
        # Base mining cost calculation
        base_electricity_price = 0.05  # $0.05 per kWh
        base_efficiency = 0.1  # TH/s per kW
        
        # Add some random variation per block
        electricity_prices = base_electricity_price + np.random.normal(0, 0.001, len(dates))
        mining_costs = electricity_prices * (1 / base_efficiency) * 24 / self.BLOCKS_PER_DAY
        
        df_costs = pd.DataFrame({
            'date': dates,
            'mining_cost': mining_costs
        })
        
        return df_costs
    
def prepare_dataset():
    collector = DataCollector()
    
    print("Fetching hashrate data...")
    df_hashrate = collector.get_hashrate_data()
    print(f"Hashrate data points: {len(df_hashrate)}")
    print(f"Hashrate date range: {df_hashrate['date'].min()} to {df_hashrate['date'].max()}")
    
    print("\nFetching price data...")
    df_price = collector.get_price_data()
    print(f"Price data points: {len(df_price)}")
    print(f"Price date range: {df_price['date'].min()} to {df_price['date'].max()}")
    
    print("\nCalculating block rewards...")
    df_rewards = collector.get_block_reward_data()
    print(f"Reward data points: {len(df_rewards)}")
    print(f"Reward date range: {df_rewards['date'].min()} to {df_rewards['date'].max()}")
    
    print("\nEstimating mining costs...")
    df_costs = collector.get_mining_cost_data()
    print(f"Cost data points: {len(df_costs)}")
    print(f"Cost date range: {df_costs['date'].min()} to {df_costs['date'].max()}")
    
    # Align date ranges
    start_date = max(
        df_hashrate['date'].min(),
        df_price['date'].min(),
        df_rewards['date'].min(),
        df_costs['date'].min()
    )
    
    end_date = min(
        df_hashrate['date'].max(),
        df_price['date'].max(),
        df_rewards['date'].max(),
        df_costs['date'].max()
    )
    
    print(f"\nAligned date range: {start_date} to {end_date}")
    
    # Create block-level timestamps (144 blocks per day)
    total_days = (end_date - start_date).days
    total_blocks = total_days * collector.BLOCKS_PER_DAY
    
    block_timestamps = pd.date_range(
        start=start_date,
        end=end_date,
        periods=total_blocks
    )
    
    # Create a DataFrame with block-level timestamps
    df_blocks = pd.DataFrame({'date': block_timestamps})
    
    # Interpolate each dataset to block level
    # Hashrate
    df_hashrate = df_hashrate.set_index('date')
    df_blocks['hashrate'] = np.interp(
        x=df_blocks['date'].astype(np.int64) // 10**9,
        xp=df_hashrate.index.astype(np.int64) // 10**9,
        fp=df_hashrate['hashrate']
    )
    
    # Price
    df_price = df_price.set_index('date')
    cs_price = CubicSpline(
        df_price.index.astype(np.int64) // 10**9,
        df_price['price']
    )
    df_blocks['exchange_rate'] = cs_price(df_blocks['date'].astype(np.int64) // 10**9)
    
    # Block rewards
    df_rewards = df_rewards.set_index('date')
    df_blocks['block_reward'] = np.interp(
        x=df_blocks['date'].astype(np.int64) // 10**9,
        xp=df_rewards.index.astype(np.int64) // 10**9,
        fp=df_rewards['block_reward']
    )
    
    # Mining costs
    df_costs = df_costs.set_index('date')
    df_blocks['mining_cost'] = np.interp(
        x=df_blocks['date'].astype(np.int64) // 10**9,
        xp=df_costs.index.astype(np.int64) // 10**9,
        fp=df_costs['mining_cost']
    )
    
    # Sort by date
    df_blocks = df_blocks.sort_values('date')
    
    # Save to CSV
    df_blocks.to_csv('bitcoin_mining_data_blocks.csv', index=False)
    print(f"\nData saved to bitcoin_mining_data_blocks.csv")
    print(f"Total blocks: {len(df_blocks)}")
    
    # Print sample of final dataset
    print("\nSample of final dataset:")
    print(df_blocks.head())
    print("\nVerifying blocks per day:")
    print(f"Number of days: {total_days}")
    print(f"Expected blocks: {total_days * collector.BLOCKS_PER_DAY}")
    print(f"Actual blocks: {len(df_blocks)}")
    
    return df_blocks

if __name__ == "__main__":
    prepare_dataset() 