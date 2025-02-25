import pandas as pd
import numpy as np
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

def load_and_prepare_data(filepath):
    """
    Load historical data from CSV and prepare it for regression analysis.
    Expected columns: date, hashrate, exchange_rate, block_reward, mining_cost
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Create lagged variables
    df['log_hashrate'] = np.log(1+df['hashrate'] + 1e-12)
    df['log_eP'] = np.log(1+(df['price'] * df['block_reward']) + 1e-12)
    df['log_efficiency'] = np.log(1+ df['mining_cost'] + 1e-12)
    df['log_electricity_cost'] = np.log(1+df['electricity_price']+1e-12)
    df['log_eP_lag'] = df['log_eP'].shift(1) 
    df['log_eP_forward'] = df['log_eP'].shift(-1) 
    df['log_mining_cost'] = np.log(1+df['mining_cost'] + 1e-12) 
    df['log_mining_cost_lag'] = df['log_mining_cost'].shift(1)
    df['log_efficiency_forward'] = df['log_mining_cost'].shift(-1)
    df['log_electricity_cost_forward'] = np.log(1+df['electricity_price'] + 1e-12).shift(-1) 
    
    # Drop rows with NaN values (from lag creation)
    df = df.dropna()
    
    return df


def prepare_change_data(df):
    """
    Prepare data for the change in hashrate regression (equation 5)
    """
    # Calculate changes in variables
    df['delta_hashrate'] = df['log_hashrate'].diff()
    df['delta_eP'] = df['log_eP'].diff()
    df['delta_eP_lag'] = df['delta_eP'].shift(1)
    
    # Drop NaN values (first two rows will have NaN due to diff and lag)
    df_changes = df.dropna()
    
    return df_changes

def run_change_regression(df):
    """
    Run regression analysis for change in hashrate (equation 5)
    ∆Nt+1 = α1 + α2∆[ePt] + α5∆[ePt-1]
    """
    # Prepare variables for regression
    X = pd.DataFrame({
        'constant': 1,  # For alpha1
        'delta_eP': df['delta_eP'],  # For alpha2
        'delta_eP_lag': df['delta_eP_lag'],  # For alpha5
    })
    
    y = df['delta_hashrate']
    
    # Run regression using statsmodels
    model = sm.OLS(y, X)
    results = model.fit()
    
    print("\nChange in Hashrate Regression Results:")
    print(results.summary())
    
    # Extract coefficients
    change_alphas = {
        'alpha1': results.params['constant'],
        'alpha2': results.params['delta_eP'],
        'alpha5': results.params['delta_eP_lag']
    }
    return change_alphas, results

def plot_change_results(df, results):
    """
    Plot actual vs predicted change in hashrate
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['delta_hashrate'], label='Actual Change in Hashrate')
    plt.plot(df.index, results.fittedvalues, label='Predicted Change in Hashrate')
    plt.xlabel('Time')
    plt.ylabel('Change in Log Hashrate')
    plt.title('Actual vs Predicted Change in Hashrate')
    plt.legend()
    plt.grid(True)
    plt.show()
def run_regression(df):
    """
    Run regression analysis to find alpha coefficients
    Including new terms: eP(t+1), c(t+1), and c(t-1)
    """
    # Prepare variables for regression
    X = pd.DataFrame({
        'constant': 1,  # For alpha1
        #'log_eP': df['log_eP'],  # For alpha2
        #'log_efficiency': df['log_efficiency'],  # For alpha3
        #'log_electricity_cost': df['log_electricity_cost'],  # For alpha4
        #'log_eP_lag': df['log_eP_lag'],  # For alpha5
        #'log_mining_cost_lag': df['log_mining_cost_lag'],  # For alpha6
        'log_eP_forward': df['log_eP_forward'],  # For alpha7
        'log_effiencey_forward': df['log_efficiency_forward'],  # For alpha8
        'log_electricity_cost_forward': df['log_electricity_cost_forward'],  # For alpha9
    })
    
    y = df['log_hashrate']
    
    # Run regression using statsmodels for detailed statistics
    model = sm.OLS(y, X)
    results = model.fit()
    
    print("\nRegression Results:")
    print(results.summary())
    
    # Extract coefficients
    alphas = {
        'alpha1': results.params['constant'],
        #'alpha2': results.params['log_eP'],
        #'alpha3': results.params['log_efficiency'],
        #'alpha4': results.params['log_electricity_cost']
        #'alpha4': results.params['log_eP_lag'],
        #'alpha5': results.params['log_mining_cost_lag'],
        'alpha6': results.params['log_eP_forward'],
        'alpha7': results.params['log_effiencey_forward'],
        'alpha8': results.params['log_electricity_cost_forward']
    }
    
    return alphas, results


def plot_results(df, results):
    """
    Plot actual vs predicted hashrate
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['hashrate'], label='Actual Hashrate')
    plt.plot(df.index, np.exp(results.fittedvalues), label='Predicted Hashrate')
    plt.xlabel('Time')
    plt.ylabel('Hashrate')
    plt.title('Actual vs Predicted Hashrate')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def validate_model(df, alphas):
    """
    Validate the model by comparing predicted vs actual values
    Including new terms
    """
    # Calculate predicted log hashrate using our coefficients
    predicted_log_hashrate = (
        alphas['alpha1'] +
        #alphas['alpha2'] * df['log_eP'] +
        #alphas['alpha3'] * df['log_efficiency'] +
        #alphas['alpha4'] * df['log_electricity_cost']
        #alphas['alpha5'] * df['log_eP_lag'] +
        #alphas['alpha6'] * df['log_mining_cost_lag'] + 
        alphas['alpha6'] * df['log_eP_forward'] +
        alphas['alpha7'] * df['log_efficiency_forward'] +
        alphas['alpha8'] * df['log_electricity_cost_forward']
    )
    
    # Convert to actual hashrate
    predicted_hashrate = np.exp(predicted_log_hashrate)
    
    # Calculate error metrics
    mse = np.mean((df['hashrate'] - predicted_hashrate) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(df['hashrate'] - predicted_hashrate))
    
    print("\nValidation Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

def analyze_correlation(df):
    """
    Analyze correlation between log(eP) and its lag
    """
    # Calculate log(eP) correlation with its lag
    correlation = df['log_eP'].corr(df['log_eP_lag'])
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['log_eP'], df['log_eP_lag'], alpha=0.5)
    plt.xlabel('log(eP_t)')
    plt.ylabel('log(eP_{t-1})')
    plt.title(f'Correlation between log(eP) and its lag\nCorrelation: {correlation:.3f}')
    
    # Add trend line
    z = np.polyfit(df['log_eP'], df['log_eP_lag'], 1)
    p = np.poly1d(z)
    plt.plot(df['log_eP'], p(df['log_eP']), "r--", alpha=0.8)
    
    plt.grid(True)
    plt.show()
    
    print("\nCorrelation Analysis:")
    print(f"Correlation between log(eP) and its lag: {correlation:.3f}")
    
    # Test for multicollinearity
    if correlation > 0.8:
        print("\nWarning: High correlation detected between log(eP) and its lag.")
        print("This might indicate multicollinearity issues in the regression model.")

def main():
    # Load and prepare data
    # Replace 'your_data.csv' with your actual data file
    try:
        df = load_and_prepare_data('bitcoin_mining_data_daily.csv')
    except FileNotFoundError:
        print("Please provide a CSV file with historical data containing:")
        print("date, hashrate, exchange_rate, block_reward, mining_cost")
        return
    
    # Add correlation analysis before running regression
    #print("\nAnalyzing correlation between log(eP) and its lag...")
    #analyze_correlation(df)
    print("\n=== Change Regression (Equation 5) ===")
    # Change regression
    df_changes = prepare_change_data(df)
    change_alphas, change_results = run_change_regression(df_changes)
    #print("\nEstimated Alpha Values (Changes):")
    #for name, value in change_alphas.items():
    #    print(f"{name}: {value:.4f}")
    #plot_change_results(df_changes, change_results)
    # Run regression
    alphas, results = run_regression(df)
    
    # Print results
    print("\nEstimated Alpha Values:")
    for name, value in alphas.items():
        print(f"{name}: {value:.4f}")
    
    # Plot results
    plot_results(df, results)
    
    # Validate model
    validate_model(df, alphas)

if __name__ == "__main__":
    main()