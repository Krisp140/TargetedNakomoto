import pandas as pd
import numpy as np
import os
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta

# File paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
efficiency_csv_path = os.path.join(base_dir, 'efficiency.csv')

# Column names for better access
LOWER_COL = 'Lower bound efficiency, J/Th'
MAIN_COL = 'Estimated efficiency, J/Th'
UPPER_COL = 'Upper bound efficiency, J/Th'

def load_efficiency_csv():
    """Load and preprocess the energy efficiency CSV file"""
    # Read the CSV file
    df = pd.read_csv(efficiency_csv_path)
    print(f"Original columns: {df.columns}")
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def train_regression_model(df, column_name, recent_only=False):
    """Train a polynomial regression model on the efficiency data
    
    Args:
        df: DataFrame with efficiency data
        column_name: Name of the column to train on
        recent_only: If True, only use the most recent year of data for training
    """
    print(f"Training model for column: {column_name}")
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # If we only want recent data, filter to the last year
    if recent_only:
        # Get the last year of data
        one_year_ago = df_copy['Date'].max() - pd.DateOffset(years=1)
        df_copy = df_copy[df_copy['Date'] >= one_year_ago]
        print(f"Using only data from {one_year_ago} to {df_copy['Date'].max()} for training")
    
    # Convert dates to numerical values (days since first date)
    first_date = df_copy['Date'].min()
    df_copy['DaysSinceStart'] = (df_copy['Date'] - first_date).dt.days
    
    # Make sure the column exists
    if column_name not in df_copy.columns:
        print(f"Column '{column_name}' not found in dataframe!")
        print(f"Available columns: {df_copy.columns}")
        raise KeyError(f"Column '{column_name}' not found")
    
    # Create polynomial features for a better fit - use degree 1 (linear) to avoid overfitting
    X = df_copy['DaysSinceStart'].values.reshape(-1, 1)
    y = df_copy[column_name].values
    
    # Try linear regression first (more stable for extrapolation)
    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(X)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return model, poly, first_date


def predict_future_values(model, poly, first_date, start_date, end_date):
    """Predict efficiency values for future dates"""
    # Generate a sequence of dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Convert to days since start
    days_since_start = [(date - first_date).days for date in date_range]
    X_pred = np.array(days_since_start).reshape(-1, 1)
    X_poly_pred = poly.transform(X_pred)
    
    # Predict values
    predictions = model.predict(X_poly_pred)
    
    # Create a dataframe with the predictions
    predictions_df = pd.DataFrame({
        'Date': date_range,
        'Predicted': predictions
    })
    
    return predictions_df


def extend_efficiency_data():
    """Main function to extend efficiency data"""
    print("Loading data...")
    efficiency_csv = load_efficiency_csv()
    
    # Get the last available date in the CSV
    last_csv_date = efficiency_csv['Date'].max()
    print(f"Last date in efficiency CSV: {last_csv_date}")
    
    # Get today's date
    today = datetime.now()
    target_date = datetime(2025, 3, 10)  # Current date mentioned by user
    
    # Train models for each column using only the most recent data
    # This is important to capture the most recent trend rather than overall trend
    print("Training models using recent data only...")
    main_model, poly_main, first_date = train_regression_model(
        efficiency_csv, MAIN_COL, recent_only=True)
    lower_model, poly_lower, _ = train_regression_model(
        efficiency_csv, LOWER_COL, recent_only=True)
    upper_model, poly_upper, _ = train_regression_model(
        efficiency_csv, UPPER_COL, recent_only=True)
    
    # Predict future values (from last date in CSV to target date)
    next_day = last_csv_date + timedelta(days=1)
    print(f"Predicting from {next_day} to {target_date}...")
    
    main_predictions = predict_future_values(
        main_model, poly_main, first_date, next_day, target_date)
    lower_predictions = predict_future_values(
        lower_model, poly_lower, first_date, next_day, target_date)
    upper_predictions = predict_future_values(
        upper_model, poly_upper, first_date, next_day, target_date)
    
    # Get the most recent values from the CSV for each column
    last_row = efficiency_csv.iloc[-1]
    last_lower = last_row[LOWER_COL]
    last_main = last_row[MAIN_COL]
    last_upper = last_row[UPPER_COL]
    
    print(f"Last values in CSV: Lower={last_lower}, Main={last_main}, Upper={last_upper}")
    
    # Combine predictions
    predictions = pd.DataFrame({
        'Date': main_predictions['Date'],
        'Lower': lower_predictions['Predicted'],
        'Main': main_predictions['Predicted'],
        'Upper': upper_predictions['Predicted']
    })
    
    # Apply constraints to ensure predictions are reasonable
    # 1. Ensure no negative values
    predictions['Lower'] = predictions['Lower'].apply(lambda x: max(x, 5.0))
    predictions['Main'] = predictions['Main'].apply(lambda x: max(x, 10.0))
    predictions['Upper'] = predictions['Upper'].apply(lambda x: max(x, 10.0))
    
    # 2. Add a maximum cap for each value (3x the last value is a reasonable constraint)
    predictions['Lower'] = predictions['Lower'].apply(lambda x: min(x, last_lower * 1.2))
    predictions['Main'] = predictions['Main'].apply(lambda x: min(x, last_main * 1.2))
    predictions['Upper'] = predictions['Upper'].apply(lambda x: min(x, last_upper * 1.2))
    
    # 3. Ensure the values follow a logical order (Lower ≤ Main ≤ Upper)
    for i, row in predictions.iterrows():
        predictions.at[i, 'Main'] = max(predictions.at[i, 'Lower'], predictions.at[i, 'Main'])
        predictions.at[i, 'Upper'] = max(predictions.at[i, 'Main'], predictions.at[i, 'Upper'])
    
    # Format predictions for appending to the efficiency data file
    formatted_predictions = []
    for _, row in predictions.iterrows():
        formatted_row = f"{row['Date'].strftime('%Y-%m-%d')},{row['Lower']:.6f},{row['Main']:.8f},{row['Upper']:.1f}"
        formatted_predictions.append(formatted_row)
        
    # Print a few predictions for verification
    print("\nFirst few predictions:")
    for pred in formatted_predictions[:5]:
        print(pred)
    
    # Append to the efficiency CSV file
    if formatted_predictions:
        print(f"Appending {len(formatted_predictions)} new predictions to the efficiency CSV file...")
        with open(efficiency_csv_path, 'a') as f:
            for prediction in formatted_predictions:
                f.write('\n' + prediction)
        print("Efficiency data has been extended successfully.")
    else:
        print("No new predictions to append.")


if __name__ == "__main__":
    extend_efficiency_data()
