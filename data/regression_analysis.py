import pandas as pd
import numpy as np
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_and_prepare_data(filepath, dropna=True):
    """
    Load historical data from CSV and prepare it for regression analysis.
    Expected columns: date, hashrate, exchange_rate, block_reward, mining_cost
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Create lagged variables
    df['log_hashrate'] = np.log(1+df['HashRate'] + 1e-12)
    df['log_eP'] = np.log(1+(df['BTCPrice'] * df['BTC']) + 1e-12)
    df['log_efficiency'] = np.log(1+ df['Efficiency'] + 1e-12)
    df['log_electricity_cost'] = np.log(1+df['ElectricityPrice']+1e-12)
    
    # Calculate deltaT as 10 - blockSpeed and then compute (1 + deltaT)
    df['deltaT'] = 10 - df['BlockSpeed']
    df['block_speed_adjusted'] = 1 + df['deltaT']
    df['log_block_speed_change'] = np.log(df['block_speed_adjusted'] + 1e-12)
    
    df['log_eP_forward'] = df['log_eP'].shift(-1) 
    df['log_efficiency_forward'] = df['log_efficiency'].shift(-1)
    df['log_electricity_cost_forward'] = df['log_electricity_cost'].shift(-1) 
    df['log_block_speed_change_forward'] = df['log_block_speed_change'].shift(-1)

    df['log_hashrate_forward'] = df['log_hashrate'].shift(-1)
    
    # Drop rows with NaN values (from lag creation) if requested
    if dropna:
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
        'log_eP_forward': df['log_eP_forward'],  # For alpha2
        'log_efficiency_forward': df['log_efficiency_forward'],  # For alpha3
        'log_electricity_cost_forward': df['log_electricity_cost_forward'],  # For alpha4
        'log_block_speed_change_forward': df['log_block_speed_change_forward'],  # For alpha5
    })
    
    y = df['log_hashrate_forward']
    
    # Run regression using statsmodels for detailed statistics
    model = sm.OLS(y, X)
    results = model.fit()
    
    print("\nRegression Results:")
    print(results.summary())
    
    # Extract coefficients
    alphas = {
        'alpha1': results.params['constant'],
        'alpha2': results.params['log_eP_forward'],
        'alpha3': results.params['log_efficiency_forward'],
        'alpha4': results.params['log_electricity_cost_forward'],
        'alpha5': results.params['log_block_speed_change_forward']
    }
    
    return alphas, results


def plot_results(df, results):
    """
    Plot actual vs predicted hashrate
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['HashRate'], label='Actual Hashrate')
    plt.plot(df.index, np.exp(results.fittedvalues), label='Predicted Hashrate')
    plt.xlabel('Time')
    plt.ylabel('Hashrate')
    plt.title('Actual vs Predicted Hashrate')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def validate_model(df, alphas, feature_set=None):
    """
    Validate the model by comparing predicted vs actual values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    alphas : dict
        Dictionary of model coefficients
    feature_set : list, optional
        List of features to use. If None, uses the default set of features
    """
    # Handle feature set
    if feature_set is None:
        # Default feature set for the original model
        feature_set = ['log_eP_forward', 'log_efficiency_forward', 
                       'log_electricity_cost_forward', 'log_block_speed_change_forward']
    
    # Initialize with constant term (intercept)
    predicted_log_hashrate = alphas.get('alpha1', 0)  # Use 0 if alpha1 not present
    
    # Add contribution from each feature
    for i, feature in enumerate(feature_set, 2):
        alpha_key = f'alpha{i}'
        if alpha_key in alphas and feature in df.columns:
            predicted_log_hashrate += alphas[alpha_key] * df[feature]
    
    # Convert to actual hashrate
    predicted_hashrate = np.exp(predicted_log_hashrate)
    
    # Drop rows with NaN in actual or predicted values
    valid_mask = ~(df['HashRate'].isna() | np.isnan(predicted_hashrate))
    actual = df.loc[valid_mask, 'HashRate']
    predicted = predicted_hashrate[valid_mask]
    
    # Calculate error metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    print("\nValidation Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'predicted': predicted,
        'actual': actual
    }

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

def run_lasso_regression(df, manual_alpha=None):
    """
    Run Lasso regression to address multicollinearity and perform feature selection
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with the data
    manual_alpha : float, optional
        Manually specified alpha value to override the one found by cross-validation
    """
    # First clean the dataframe to remove all NaN values
    df_clean = df.dropna()
    
    # Prepare variables for regression from the cleaned dataframe
    X = pd.DataFrame({
        'log_eP_forward': df_clean['log_eP_forward'],
        'log_efficiency_forward': df_clean['log_efficiency_forward'],
        'log_electricity_cost_forward': df_clean['log_electricity_cost_forward'],
        'log_block_speed_change_forward': df_clean['log_block_speed_change_forward'],
    })
    
    y = df_clean['log_hashrate_forward']
    
    print(f"Number of rows for Lasso regression: {len(X)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use LassoCV to find the optimal alpha parameter through cross-validation
    if manual_alpha is None:
        print("\nFinding optimal alpha through cross-validation...")
        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso_cv.fit(X_scaled, y)
        optimal_alpha = lasso_cv.alpha_
        print(f"Optimal Lasso alpha from CV: {optimal_alpha:.6f}")
    else:
        optimal_alpha = manual_alpha
        print(f"Using manually specified alpha: {optimal_alpha:.6f}")
    
    # Try a range of alpha values to see feature selection behavior
    alphas = np.logspace(-6, 2, 100)  # Expanded range including larger values
    coefs = []
    condition_numbers = []
    
    print("\nAnalyzing feature selection across different alpha values...")
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y)
        coefs.append(lasso.coef_)
        
        # Calculate condition number for non-zero features
        selected = [i for i, coef in enumerate(lasso.coef_) if abs(coef) > 1e-5]
        if len(selected) >= 2:  # Need at least 2 features to calculate condition number
            X_selected = X.iloc[:, selected].to_numpy()
            condition_numbers.append(np.linalg.cond(X_selected))
        else:
            condition_numbers.append(np.nan)
    
    # Find the alpha with the minimum condition number (if any valid ones)
    valid_condition_numbers = [cn for cn in condition_numbers if not np.isnan(cn)]
    if valid_condition_numbers:
        best_cn_idx = np.nanargmin(condition_numbers)
        best_alpha_for_cn = alphas[best_cn_idx]
        best_cn = condition_numbers[best_cn_idx]
        
        print(f"\nAlpha with lowest condition number: {best_alpha_for_cn:.6f}")
        print(f"Lowest achievable condition number: {best_cn:.2f}")
        
        # Check if we should use this alpha instead
        if manual_alpha is None and best_cn < np.linalg.cond(X):
            print(f"Suggesting to use alpha={best_alpha_for_cn:.6f} instead of CV-optimal alpha={optimal_alpha:.6f}")
            print(f"Run with manual_alpha={best_alpha_for_cn:.6f} to use this value")
    
    # Fit the Lasso model with the chosen alpha
    lasso = Lasso(alpha=optimal_alpha, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # Get feature names and coefficients
    feature_names = X.columns
    coef_dict = dict(zip(feature_names, lasso.coef_))
    
    # Sort coefficients by absolute magnitude to see relative importance
    sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\nLasso Regression Coefficients (sorted by importance):")
    for feature, coef in sorted_coefs:
        status = "KEEP" if abs(coef) > 1e-5 else "REMOVE (coefficient ≈ 0)"
        print(f"{feature}: {coef:.6f} - {status}")
    
    # Identify non-zero coefficients (selected features)
    selected_features = [feature for feature, coef in coef_dict.items() if abs(coef) > 1e-5]
    removed_features = [feature for feature, coef in coef_dict.items() if abs(coef) <= 1e-5]
    
    print(f"\nSelected features by Lasso: {selected_features}")
    if removed_features:
        print(f"Features to remove (coefficients ≈ 0): {removed_features}")
    else:
        print("No features were removed. Try a larger alpha value to force feature selection.")
    
    # Calculate model metrics
    y_pred = lasso.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Lasso MSE: {mse:.6f}")
    print(f"Lasso R²: {r2:.6f}")
    
    # Calculate condition number to check if multicollinearity was addressed
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        condition_number = np.linalg.cond(X_selected)
        condition_number_original = np.linalg.cond(X)
        print(f"Original condition number: {condition_number_original:.2f}")
        print(f"Condition number after feature selection: {condition_number:.2f}")
        reduction = (1 - condition_number/condition_number_original)*100
        print(f"Reduction in condition number: {reduction:.1f}%")
        
        if reduction < 1.0:
            print("\nWARNING: Minimal or no reduction in condition number.")
            print("Try using a larger alpha value (e.g., manual_alpha=0.1 or higher)")
    else:
        condition_number = None
        print("No features were selected by Lasso.")
    
    # Calculate correlations to help understand multicollinearity
    correlation_matrix = X.corr()
    print("\nCorrelation Matrix (to identify multicollinear features):")
    print(correlation_matrix)
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.7:  # Threshold for high correlation
                high_corr_pairs.append((feature_names[i], feature_names[j], corr))
    
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (potential multicollinearity):")
        for f1, f2, corr in high_corr_pairs:
            f1_status = "KEPT" if f1 in selected_features else "REMOVED"
            f2_status = "KEPT" if f2 in selected_features else "REMOVED"
            print(f"{f1} and {f2}: correlation = {corr:.3f} | {f1} {f1_status}, {f2} {f2_status}")
    
    return {
        'lasso_model': lasso,
        'selected_features': selected_features,
        'removed_features': removed_features,
        'optimal_alpha': optimal_alpha,
        'feature_coefs': coef_dict,
        'condition_number': condition_number,
        'df_clean': df_clean,  # Return cleaned dataframe
        'coefs_path': coefs,
        'alphas_path': alphas,
        'corr_matrix': correlation_matrix,
        'condition_numbers': condition_numbers,
        'best_alpha_for_condition': best_alpha_for_cn if valid_condition_numbers else None
    }

def plot_lasso_path(lasso_results):
    """
    Plot the Lasso regularization path to visualize how coefficients
    change with different alpha values. This helps determine which features 
    to eliminate due to multicollinearity.
    """
    alphas = lasso_results['alphas_path']
    coefs = lasso_results['coefs_path']
    feature_names = lasso_results['selected_features'] + lasso_results['removed_features']
    
    # Plot the Lasso paths
    plt.figure(figsize=(14, 8))
    
    # Convert to array for easier plotting
    coef_array = np.array(coefs)
    
    # Plot each coefficient progression
    for i, feature in enumerate(feature_names):
        plt.plot(-np.log10(alphas), coef_array[:, i], label=feature)
    
    # Add vertical line at the selected alpha
    optimal_alpha = lasso_results['optimal_alpha']
    plt.axvline(x=-np.log10(optimal_alpha), color='k', linestyle='--', 
                label=f'Optimal α = {optimal_alpha:.6f}')
    
    # Highlight zero line to show when features are excluded
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    
    plt.title('Lasso Regularization Path - Feature Selection Visualization', fontsize=14)
    plt.xlabel('-log(α)', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add annotation explaining interpretation
    plt.figtext(0.5, 0.01, 
                "Interpretation Guide:\n"
                "• Features whose lines cross zero early (at small -log(α)) should be removed\n"
                "• Features with lines that remain non-zero at the optimal α (dashed line) should be kept\n"
                "• Features that have been zeroed out at the optimal α are identified as redundant and should be removed",
                ha='center', bbox={'facecolor':'lightyellow', 'alpha':0.5, 'pad':5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust for the figtext
    plt.show()
    
    # Create a second plot showing coefficient magnitudes at optimal alpha
    plt.figure(figsize=(10, 6))
    
    # Get coefficients at optimal alpha
    feature_coefs = lasso_results['feature_coefs']
    features = list(feature_coefs.keys())
    coef_values = list(feature_coefs.values())
    
    # Sort by absolute coefficient value
    idx = np.argsort(np.abs(coef_values))
    sorted_features = [features[i] for i in idx]
    sorted_coefs = [coef_values[i] for i in idx]
    
    # Create bar chart colored by whether feature is selected
    colors = ['green' if abs(c) > 1e-5 else 'red' for c in sorted_coefs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_features, sorted_coefs, color=colors)
    
    # Add labels
    plt.xlabel('Coefficient Value (Importance)', fontsize=12)
    plt.title('Feature Importance from Lasso Regression', fontsize=14)
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Keep this feature'),
        Patch(facecolor='red', label='Remove this feature (coefficient ≈ 0)')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_regression_with_selected_features(df_clean, selected_features):
    """
    Run regression analysis using only features selected by Lasso
    to address multicollinearity issues
    
    Parameters:
    -----------
    df_clean : pandas.DataFrame
        Cleaned dataframe with no NaN values
    selected_features : list
        List of features selected by Lasso
    """
    # First make a completely fresh copy of the dataframe to avoid any index issues
    df_copy = df_clean.copy().reset_index(drop=True)
    
    # Create features and target from this same dataframe to ensure index alignment
    X = pd.DataFrame({
        'constant': np.ones(len(df_copy))  # For alpha1
    })
    
    # Add selected features
    for feature in selected_features:
        X[feature] = df_copy[feature]
    
    y = df_copy['log_hashrate_forward']
    
    # Check for any remaining NaNs
    nan_sum = X.isna().sum().sum() + y.isna().sum()
    if nan_sum > 0:
        print("WARNING: Found NaN values - performing final cleaning")
        
        # Create a mask for rows that have no NaN values in either X or y
        mask = ~(X.isna().any(axis=1) | y.isna())
        
        # Apply the same mask to both X and y to keep them aligned
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        
        print(f"Final X shape after cleaning: {X.shape}")
        print(f"Final y shape after cleaning: {y.shape}")
        
        # Quick check to confirm indices are aligned
        print(f"X index range: {X.index.min()} to {X.index.max()}")
        print(f"y index range: {y.index.min()} to {y.index.max()}")
    
    # Verify that X and y have the same length
    assert len(X) == len(y), f"Length mismatch: X has {len(X)} rows, y has {len(y)} rows"
    print('X', X)
    print('y', y)
    # Run regression using statsmodels for detailed statistics
    model = sm.OLS(y, X)
    results = model.fit()
    
    print("\nRegression Results with Selected Features:")
    print(results.summary())
    
    # Extract coefficients
    alphas = {'alpha1': results.params['constant']}
    for i, feature in enumerate(selected_features, 2):
        alphas[f'alpha{i}'] = results.params[feature]
    
    return alphas, results

def main():
    # Load and prepare data
    df_clean = load_and_prepare_data('data/merged_data_2.csv', dropna=True)  # for fitting
    df_full = load_and_prepare_data('data/merged_data_2.csv', dropna=False)  # for full prediction export
    
    # Analyze correlation to check for multicollinearity
    print("\n==== Correlation Analysis ====")
    
    # Calculate correlation on clean data
    correlation_matrix = df_clean[['log_eP_forward', 'log_efficiency_forward', 
                                  'log_electricity_cost_forward', 'log_block_speed_change_forward']].corr()
    print(correlation_matrix)
    
    # Calculate condition number to quantify multicollinearity
    X = df_clean[['log_eP_forward', 'log_efficiency_forward', 
                 'log_electricity_cost_forward', 'log_block_speed_change_forward']]
    condition_number = np.linalg.cond(X)
    print(f"\nCondition number: {condition_number:.2f}")
    print("A condition number > 30 indicates multicollinearity")
    
    # Run original regression on clean data
    print("\n==== Original Regression ====")
    alphas, results = run_regression(df_clean)
    
    # Run Lasso regression to address multicollinearity
    print("\n==== Lasso Regression for Feature Selection ====")
    
    # Try Lasso with default CV-optimal alpha first
    lasso_results = run_lasso_regression(df_clean)
    
    # If no features were removed and condition number didn't improve, try with a larger alpha
    if not lasso_results['removed_features'] or (lasso_results['condition_number'] and 
                                                lasso_results['condition_number'] > condition_number * 0.9):
        print("\n==== Retrying Lasso with Larger Alpha ====")
        # Try with manual alpha = 0.1 as a starting point
        lasso_results = run_lasso_regression(df_clean, manual_alpha=0.040370)
        
        # If still no improvement, try an even larger alpha
        if not lasso_results['removed_features']:
            print("\n==== Retrying Lasso with Even Larger Alpha ====")
            lasso_results = run_lasso_regression(df_clean, manual_alpha=0.040370)
    
    # Plot Lasso regularization path using the results from Lasso
    print("\n==== Lasso Path Visualization ====")
    plot_lasso_path(lasso_results)
    
    # Additional plot to show condition number vs alpha
    plt.figure(figsize=(10, 6))
    valid_indices = ~np.isnan(lasso_results['condition_numbers'])
    if np.any(valid_indices):
        plt.semilogx(lasso_results['alphas_path'][valid_indices], 
                    np.array(lasso_results['condition_numbers'])[valid_indices])
        plt.axhline(y=condition_number, color='r', linestyle='--', 
                   label=f'Original condition number: {condition_number:.2f}')
        plt.axvline(x=lasso_results['optimal_alpha'], color='g', linestyle='--',
                   label=f'Used alpha: {lasso_results["optimal_alpha"]:.6f}')
        if lasso_results['best_alpha_for_condition']:
            plt.axvline(x=lasso_results['best_alpha_for_condition'], color='b', linestyle=':',
                       label=f'Best alpha for condition: {lasso_results["best_alpha_for_condition"]:.6f}')
        plt.xlabel('Alpha')
        plt.ylabel('Condition Number')
        plt.title('Effect of Alpha on Condition Number')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    # Run regression with only selected features
    if len(lasso_results['selected_features']) > 0:
        print("\n==== Regression with Selected Features ====")
        selected_alphas, selected_results = run_regression_with_selected_features(
            lasso_results['df_clean'], lasso_results['selected_features'])
        
        # Validate and compare models
        print("\n==== Model Validation ====")
        print("Original Model:")
        original_metrics = validate_model(df_clean, alphas)

        # === Export predicted hashrate from original model to CSV (cleaned, as before) ===
        predicted = original_metrics['predicted']
        actual = original_metrics['actual']
        if 'Date' in df_clean.columns:
            date_index = df_clean.loc[predicted.index, 'Date']
        else:
            date_index = predicted.index
        export_df = pd.DataFrame({
            'Date': date_index,
            'Predicted_HashRate': predicted.values,
            'Actual_HashRate': actual.values
        })
        export_df.to_csv('predicted_hashrate_original.csv', index=False)
        print(f"\nPredicted hashrate (original model) exported to predicted_hashrate_original.csv with {len(export_df)} rows.")
        
        # === Export predicted hashrate for ALL dates (full, not dropping NaNs) ===
        # Use the same regression coefficients, but apply to df_full
        feature_set = ['log_eP_forward', 'log_efficiency_forward', 'log_electricity_cost_forward', 'log_block_speed_change_forward']
        predicted_log_hashrate_full = alphas.get('alpha1', 0)
        for i, feature in enumerate(feature_set, 2):
            alpha_key = f'alpha{i}'
            if alpha_key in alphas and feature in df_full.columns:
                predicted_log_hashrate_full += alphas[alpha_key] * df_full[feature]
        predicted_hashrate_full = np.exp(predicted_log_hashrate_full)
        # Export with Date and Actual_HashRate (if available)
        export_full_df = pd.DataFrame({
            'Date': df_full['Date'] if 'Date' in df_full.columns else df_full.index,
            'Predicted_HashRate': predicted_hashrate_full,
            'Actual_HashRate': df_full['HashRate'] if 'HashRate' in df_full.columns else np.nan
        })
        export_full_df.to_csv('predicted_hashrate_full.csv', index=False)
        print(f"\nPredicted hashrate (original model, all dates) exported to predicted_hashrate_full.csv with {len(export_full_df)} rows.")

        print("\nModel with Selected Features:")
        feature_set = {'constant': selected_alphas['alpha1']}
        for i, feature in enumerate(lasso_results['selected_features'], 2):
            feature_set[feature] = selected_alphas[f'alpha{i}']
        lasso_metrics = validate_model(df_clean, selected_alphas, lasso_results['selected_features'])
        
        # Compare models
        print("\nModel Comparison:")
        improvement_mse = (original_metrics['mse'] - lasso_metrics['mse']) / original_metrics['mse'] * 100
        print(f"MSE improvement: {improvement_mse:.2f}%")
        
        # Print final conclusion about which features to keep and remove
        print("\n==== FINAL RECOMMENDATION ====")
        if lasso_results['removed_features']:
            print(f"Based on Lasso analysis, REMOVE these features due to multicollinearity:")
            for feature in lasso_results['removed_features']:
                print(f"  - {feature}")
            
            print(f"\nKEEP these features in your model:")
            for feature in lasso_results['selected_features']:
                print(f"  - {feature}")
        else:
            print("Lasso did not identify any features to remove.")
            print("You may want to try a larger alpha value manually or use ridge regression.")
        
        # Print condition number improvement
        if 'condition_number' in lasso_results and lasso_results['condition_number'] is not None:
            print(f"\nCondition number improvement: {condition_number:.2f} → {lasso_results['condition_number']:.2f}")
            
        # Plot results
        plot_results(df_clean, results)
        plot_results(df_clean, selected_results)

        # === Export predicted hashrate to CSV ===
        # Use the predicted and actual values from lasso_metrics
        predicted = lasso_metrics['predicted']
        actual = lasso_metrics['actual']
        # Try to get the date index if available
        if 'Date' in df_clean.columns:
            date_index = df_clean.loc[predicted.index, 'Date']
        else:
            date_index = predicted.index  # fallback to integer index
        export_df = pd.DataFrame({
            'Date': date_index,
            'Predicted_HashRate': predicted.values,
            'Actual_HashRate': actual.values
        })
        export_df.to_csv('predicted_hashrate_lasso.csv', index=False)
        print(f"\nPredicted hashrate (Lasso) exported to predicted_hashrate_lasso.csv with {len(export_df)} rows.")
    else:
        print("\nNo features were selected by Lasso. Skipping regression with selected features.")
        print("Try using a smaller alpha value manually for Lasso.")
        
        # Validate original model only
        print("\n==== Model Validation ====")
        print("Original Model:")
        validate_model(df_clean, alphas)
        
        # Plot results for original model only
        plot_results(df_clean, results)

if __name__ == "__main__":
    main()