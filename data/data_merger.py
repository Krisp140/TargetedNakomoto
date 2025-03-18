import pandas as pd
import os
import calendar
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Define file paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
block_speed_path = os.path.join(base_dir, 'block_speed.csv')
blockreward_path = os.path.join(base_dir, 'blockreward.csv')
hashrate_path = os.path.join(base_dir, 'hashrate.csv')
price_path = os.path.join(base_dir, 'price.csv')
efficiency_path = os.path.join(base_dir, 'efficiency.csv')
elect_price_path = os.path.join(base_dir, 'data', 'elect_price.csv')

# Output path
output_path = os.path.join(base_dir, 'data', 'merged_data.csv')

def read_efficiency_data(file_path):
    """Read efficiency data from CSV file and convert to DataFrame"""
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Convert date string to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # We only need the Date and estimated efficiency column
    df = df[['Date', 'Estimated efficiency, J/Th']]
    return df

def read_electricity_price_data(file_path):
    """Read electricity price data from CSV file and expand to daily data"""
    try:
        # First read in the raw CSV to get the column structure
        with open(file_path, 'r') as f:
            header_line = None
            for i, line in enumerate(f):
                if i == 2:  # The third line contains the header
                    header_line = line.strip()
                    break
        
        if not header_line:
            print("Could not find header line in electricity price CSV")
            return pd.DataFrame(columns=['Date', 'ElectricityPrice'])
            
        # Now read the file with pandas, skipping the first two rows
        df = pd.read_csv(file_path, skiprows=2)
        
        # Create a list to store monthly data
        monthly_data = []
        current_year = None
        
        # Process each row
        for idx, row in df.iterrows():
            period = row['Period']
            
            # Check if this is a year marker
            if isinstance(period, str) and period.startswith('Year ') and not period.startswith('Year to'):
                try:
                    year_parts = period.split(' ')
                    if len(year_parts) >= 2 and year_parts[1].isdigit():
                        current_year = int(year_parts[1])
                        print(f"Found year marker: {current_year}")
                except (ValueError, IndexError):
                    continue
            
            # Process month rows if we have a current year
            elif current_year and isinstance(period, str) and period in [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'Sept', 'October', 'November', 'December'
            ]:
                # Get the price from the All Sectors column
                if 'All Sectors' in df.columns and pd.notna(row['All Sectors']):
                    try:
                        price = float(row['All Sectors'])
                    except (ValueError, TypeError):
                        continue
                    
                    # Handle 'Sept' vs 'September'
                    month_name = 'September' if period == 'Sept' else period
                    month_num = list(calendar.month_name).index(month_name)
                    
                    if month_num > 0:  # Skip if month_num is 0 (not a valid month)
                        # Create a date for the first day of the month
                        date = pd.Timestamp(year=current_year, month=month_num, day=1)
                        
                        monthly_data.append({
                            'Date': date,
                            'ElectricityPrice': price
                        })
                        print(f"Added data for {month_name} {current_year}: {price}")
        
        if not monthly_data:
            # Try alternative approach if no data was found
            print("Using alternative approach to parse electricity prices...")
            
            # Read the CSV as raw text
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            current_year = None
            
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 6:  # Need enough columns for period and price
                    continue
                
                period = parts[0].strip()
                
                # Check for year markers
                if period.startswith('Year 20') and not period.startswith('Year to'):
                    try:
                        current_year = int(period.split(' ')[1])
                        print(f"Found year marker: {current_year}")
                    except (ValueError, IndexError):
                        continue
                
                # Check for month entries
                elif current_year and period in [
                    'January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'Sept', 'October', 'November', 'December'
                ]:
                    # Try to get price from the last column (All Sectors)
                    try:
                        price = float(parts[-1].strip())
                        
                        # Handle 'Sept' vs 'September'
                        month_name = 'September' if period == 'Sept' else period
                        month_num = list(calendar.month_name).index(month_name)
                        
                        if month_num > 0:  # Skip if month_num is 0 (not a valid month)
                            # Create a date for the first day of the month
                            date = pd.Timestamp(year=current_year, month=month_num, day=1)
                            
                            monthly_data.append({
                                'Date': date,
                                'ElectricityPrice': price
                            })
                            print(f"Added data for {month_name} {current_year}: {price}")
                    except (ValueError, IndexError):
                        continue
                        
        if not monthly_data:
            print("Warning: No valid monthly electricity price data found.")
            return pd.DataFrame(columns=['Date', 'ElectricityPrice'])
    except Exception as e:
        print(f"Error processing electricity price data: {e}")
        # Return empty DataFrame if there's an error
        return pd.DataFrame(columns=['Date', 'ElectricityPrice'])
    
    # Convert to DataFrame
    monthly_df = pd.DataFrame(monthly_data)
    
    # If no valid monthly data was found, return empty DataFrame
    if monthly_df.empty:
        print("Warning: No valid electricity price data found")
        return pd.DataFrame(columns=['Date', 'ElectricityPrice'])
    
    # Expand monthly data to daily data
    daily_data = []
    
    for i in range(len(monthly_df)):
        current_month = monthly_df.iloc[i]
        current_date = current_month['Date']
        current_price = current_month['ElectricityPrice']
        
        # Get the number of days in the current month
        days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]
        
        # Create a daily entry for each day in the month
        for day in range(1, days_in_month + 1):
            daily_date = pd.Timestamp(year=current_date.year, month=current_date.month, day=day)
            daily_data.append({
                'Date': daily_date,
                'ElectricityPrice': current_price
            })
    
    # Convert to DataFrame
    daily_df = pd.DataFrame(daily_data)
    daily_df = daily_df.sort_values('Date')
    
    print(f"Processed electricity price data: {len(daily_df)} daily records from {len(monthly_df)} months")
    return daily_df

def read_csv_data(file_path):
    """Read CSV data with DateTime column"""
    df = pd.read_csv(file_path)
    # Rename DateTime column to Date and convert to datetime
    df = df.rename(columns={'DateTime': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def main():
    # Read all the data
    print("Reading data files...")
    try:
        block_speed_df = read_csv_data(block_speed_path)
        blockreward_df = read_csv_data(blockreward_path)
        hashrate_df = read_csv_data(hashrate_path)
        price_df = read_csv_data(price_path)
        efficiency_df = read_efficiency_data(efficiency_path)
        electricity_df = read_electricity_price_data(elect_price_path)
        
        # Rename columns to avoid duplicate column names after merge
        block_speed_df = block_speed_df.rename(columns={'Block Speed': 'BlockSpeed'})
        hashrate_df = hashrate_df.rename(columns={'Hash Rate': 'HashRate'})
        price_df = price_df.rename(columns={'Price': 'BTCPrice'})
        # Drop the SMA column from price_df as requested
        if 'SMA (200)' in price_df.columns:
            price_df = price_df.drop(columns=['SMA (200)'])
        efficiency_df = efficiency_df.rename(columns={'Estimated efficiency, J/Th': 'Efficiency'})
        # Electricity price is already named appropriately
        
        # Merge all dataframes on Date
        print("Merging dataframes...")
        merged_df = efficiency_df
        
        # Only keep the Date and the value columns from each dataframe
        merged_df = pd.merge(merged_df, block_speed_df, on='Date', how='left')
        merged_df = pd.merge(merged_df, blockreward_df, on='Date', how='left')
        merged_df = pd.merge(merged_df, hashrate_df, on='Date', how='left')
        merged_df = pd.merge(merged_df, price_df, on='Date', how='left')
        merged_df = pd.merge(merged_df, electricity_df, on='Date', how='left')
        
        # Forward fill any missing electricity prices (if a month doesn't have data)
        if 'ElectricityPrice' in merged_df.columns:
            merged_df['ElectricityPrice'] = merged_df['ElectricityPrice'].fillna(method='ffill')
            # Backward fill for any dates at the beginning with no data
            merged_df['ElectricityPrice'] = merged_df['ElectricityPrice'].fillna(method='bfill')
        else:
            print("Warning: ElectricityPrice column not found in merged data")
        
        # Filter for dates after January 2022
        print("Filtering for dates after January 2022...")
        cutoff_date = pd.to_datetime('2022-01-01')
        filtered_df = merged_df[merged_df['Date'] >= cutoff_date]
        
        # Save the merged and filtered data
        print(f"Saving merged data to {output_path}...")
        filtered_df.to_csv(output_path, index=False)
        print(f"Successfully merged data files. Output saved to {output_path}")
        
        # Print statistics about the merged data
        print(f"Total rows in merged data: {len(filtered_df)}")
        print(f"Date range: {filtered_df['Date'].min()} to {filtered_df['Date'].max()}")
        print(f"Columns in merged data: {filtered_df.columns.tolist()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
