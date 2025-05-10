import pandas as pd
import os

def update_electricity_prices():
    # Read both CSV files
    print("Reading the datasets...")
    merged_data = pd.read_csv('data/merged_data.csv')
    new_elect_price = pd.read_csv('data/new_elect_price.csv')
    
    # Clean up column names (remove any spaces)
    merged_data.columns = [col.strip() for col in merged_data.columns]
    new_elect_price.columns = [col.strip() for col in new_elect_price.columns]
    
    # Convert Date columns to datetime for proper joining
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    new_elect_price['Date'] = pd.to_datetime(new_elect_price['Date'])
    
    # Create a backup of the original file
    backup_path = 'data/merged_data_backup.csv'
    if not os.path.exists(backup_path):
        print(f"Creating backup at {backup_path}")
        merged_data.to_csv(backup_path, index=False)
    
    # Create a dictionary mapping dates to new electricity prices
    print("Creating mapping of dates to new electricity prices...")
    elec_price_dict = dict(zip(new_elect_price['Date'], new_elect_price['ElectricityPrice']))
    
    # Update the ElectricityPrice column based on matching dates
    print("Updating electricity prices in the merged dataset...")
    updated_prices = []
    
    for date in merged_data['Date']:
        if date in elec_price_dict:
            updated_prices.append(elec_price_dict[date])
        else:
            # Keep the original value if no match is found
            idx = merged_data[merged_data['Date'] == date].index[0]
            updated_prices.append(merged_data.at[idx, 'ElectricityPrice'])
    
    # Replace the ElectricityPrice column
    merged_data['ElectricityPrice'] = updated_prices
    
    # Save the updated dataset
    print("Saving updated dataset...")
    merged_data.to_csv('data/merged_data_2.csv', index=False)
    print("Done! The ElectricityPrice column has been updated.")

if __name__ == "__main__":
    update_electricity_prices() 