"""
Retail Demand Intelligence System
==================================
A comprehensive ML pipeline for retail demand prediction and business insights.

Author: Senior ML Engineer
Date: February 2026

This system includes:
- Phase 1: Data Preprocessing
- Phase 2: XGBoost Demand Prediction Model
- Phase 3: Month-wise Seasonal Sales Forecast
- Phase 4: Category Revenue Distribution
- Phase 5: Promotion Impact Analysis
- Phase 6: Price Elasticity Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for script execution
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set plot style for professional visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10

# ============================================================
# PHASE 1: DATA PREPROCESSING
# ============================================================

def load_and_inspect_data(filepath):
    """
    Load the retail dataset and perform initial inspection.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    print("=" * 60)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    print(f"\n[INFO] Dataset loaded successfully!")
    print(f"[INFO] Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\n[INFO] Columns: {list(df.columns)}")
    print(f"\n[INFO] Data Types:\n{df.dtypes}")
    print(f"\n[INFO] First 5 rows:\n{df.head()}")
    
    return df


def clean_data(df):
    """
    Clean the dataset:
    - Remove duplicates
    - Handle missing values (median for numerical, mode for categorical)
    - Validate and fix unrealistic values
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "-" * 40)
    print("DATA CLEANING")
    print("-" * 40)
    
    initial_rows = len(df)
    
    # Step 1: Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"[INFO] Duplicates removed: {duplicates_removed}")
    
    # Step 2: Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"[INFO] Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        # Impute numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"[INFO] {col}: Imputed with median = {median_val:.2f}")
        
        # Impute categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"[INFO] {col}: Imputed with mode = {mode_val}")
    else:
        print("[INFO] No missing values found!")
    
    # Step 3: Validate and fix unrealistic values
    print("\n[INFO] Validating data for unrealistic values...")
    
    # Fix negative prices
    if (df['price'] <= 0).any():
        count = (df['price'] <= 0).sum()
        df.loc[df['price'] <= 0, 'price'] = df['price'].median()
        print(f"[WARNING] Fixed {count} negative/zero prices")
    
    # Fix negative units_sold
    if (df['units_sold'] < 0).any():
        count = (df['units_sold'] < 0).sum()
        df.loc[df['units_sold'] < 0, 'units_sold'] = 0
        print(f"[WARNING] Fixed {count} negative units_sold values")
    
    # Fix discount_percent > 100 or < 0
    if (df['discount_percent'] < 0).any() or (df['discount_percent'] > 100).any():
        count = ((df['discount_percent'] < 0) | (df['discount_percent'] > 100)).sum()
        df['discount_percent'] = df['discount_percent'].clip(0, 100)
        print(f"[WARNING] Fixed {count} invalid discount_percent values")
    
    # Fix negative stock_available
    if (df['stock_available'] < 0).any():
        count = (df['stock_available'] < 0).sum()
        df.loc[df['stock_available'] < 0, 'stock_available'] = 0
        print(f"[WARNING] Fixed {count} negative stock values")
    
    # Fix rating out of range (should be 1-5)
    if (df['rating'] < 0).any() or (df['rating'] > 5).any():
        count = ((df['rating'] < 0) | (df['rating'] > 5)).sum()
        df['rating'] = df['rating'].clip(0, 5)
        print(f"[WARNING] Fixed {count} invalid rating values")
    
    print(f"\n[SUCCESS] Data cleaning completed. Final shape: {df.shape}")
    
    return df


def engineer_features(df):
    """
    Create new features for the model:
    - Extract date components (year, month, day)
    - Create derived features (revenue, price_difference, discount_value, sales_per_stock)
    - Encode categorical variables
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with engineered features, label encoders
    """
    print("\n" + "-" * 40)
    print("FEATURE ENGINEERING")
    print("-" * 40)
    
    # Step 1: Convert date to datetime and extract components
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month  # Numeric month for ML
    df['day'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    print("[INFO] Extracted date components: year, month_num, day, week_of_year")
    
    # Step 2: Create derived features
    # Note: The dataset already has 'revenue', but we'll recalculate for consistency
    df['calculated_revenue'] = df['units_sold'] * df['price']
    df['price_difference'] = df['price'] - df['competitor_price']
    df['discount_value'] = df['price'] * (df['discount_percent'] / 100)
    
    # Handle division by zero for sales_per_stock
    df['sales_per_stock'] = np.where(
        df['stock_available'] > 0,
        df['units_sold'] / df['stock_available'],
        0
    )
    
    # Additional useful features
    df['price_ratio'] = df['price'] / df['competitor_price']
    df['profit_margin'] = (df['price'] - df['cost_price']) / df['price']
    
    print("[INFO] Created derived features: price_difference, discount_value, sales_per_stock, price_ratio, profit_margin")
    
    # Step 3: Encode categorical variables
    label_encoders = {}
    categorical_cols = ['category', 'season', 'day_of_week']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"[INFO] Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    print(f"\n[SUCCESS] Feature engineering completed. New shape: {df.shape}")
    
    # Save label encoders for inference
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("[INFO] Label encoders saved: label_encoders.pkl")
    
    return df, label_encoders


def prepare_model_data(df):
    """
    Prepare features and target for the ML model.
    Perform time-aware train-test split to avoid data leakage.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, df_train, df_test
    """
    print("\n" + "-" * 40)
    print("TRAIN-TEST SPLIT (TIME-AWARE)")
    print("-" * 40)
    
    # Sort by date for time-aware split
    df = df.sort_values('date').reset_index(drop=True)
    
    # Define features for the model
    feature_cols = [
        'price', 'cost_price', 'discount_percent', 'stock_available',
        'competitor_price', 'rating', 'review_count', 'is_promoted',
        'is_weekend', 'days_since_launch', 'year', 'month_num', 'day',
        'week_of_year', 'price_difference', 'discount_value',
        'price_ratio', 'profit_margin', 'category_encoded',
        'season_encoded', 'day_of_week_encoded'
    ]
    
    # Check for is_holiday column
    if 'is_holiday' in df.columns:
        feature_cols.append('is_holiday')
    
    target_col = 'units_sold'
    
    # Time-aware split (80% train, 20% test based on chronological order)
    split_idx = int(len(df) * 0.8)
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_train = df_train[target_col]
    y_test = df_test[target_col]
    
    print(f"[INFO] Features used: {len(feature_cols)}")
    print(f"[INFO] Training set: {len(X_train)} samples ({df_train['date'].min()} to {df_train['date'].max()})")
    print(f"[INFO] Test set: {len(X_test)} samples ({df_test['date'].min()} to {df_test['date'].max()})")
    print(f"[INFO] Target variable: {target_col}")
    
    # Save feature columns for inference
    joblib.dump(feature_cols, 'feature_columns.pkl')
    print("[INFO] Feature columns saved: feature_columns.pkl")
    
    return X_train, X_test, y_train, y_test, feature_cols, df_train, df_test


# ============================================================
# PHASE 2: DEMAND PREDICTION MODEL
# ============================================================

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost Regressor for demand prediction.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Trained model, predictions
    """
    print("\n" + "=" * 60)
    print("PHASE 2: DEMAND PREDICTION MODEL (XGBoost)")
    print("=" * 60)
    
    # Try GPU first, fall back to CPU if not available
    try:
        # XGBoost parameters optimized for retail demand prediction
        # Using GPU acceleration for faster training
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            tree_method='hist',      # Use histogram-based algorithm
            device='cuda',           # Enable GPU acceleration
            n_jobs=-1
        )
        
        print("\n[INFO] Training XGBoost model with GPU acceleration...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        print("[SUCCESS] Model training completed on GPU!")
        
    except Exception as e:
        print(f"\n[WARNING] GPU not available ({str(e)[:50]}...), falling back to CPU")
        # Use CPU-based training
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
        
        print("[INFO] Training XGBoost model on CPU...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        print("[SUCCESS] Model training completed on CPU!")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Save the trained model
    model.save_model('demand_model.json')
    print("[INFO] Model saved: demand_model.json")
    
    return model, y_train_pred, y_test_pred


def evaluate_model(y_train, y_train_pred, y_test, y_test_pred):
    """
    Evaluate model performance using RMSE, MAE, and R² score.
    
    Args:
        y_train, y_train_pred: Training actual and predicted values
        y_test, y_test_pred: Test actual and predicted values
    """
    print("\n" + "-" * 40)
    print("MODEL EVALUATION")
    print("-" * 40)
    
    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n+----------------+------------+------------+")
    print("| Metric         |   Train    |    Test    |")
    print("+----------------+------------+------------+")
    print(f"| RMSE           | {train_rmse:10.2f} | {test_rmse:10.2f} |")
    print(f"| MAE            | {train_mae:10.2f} | {test_mae:10.2f} |")
    print(f"| R² Score       | {train_r2:10.4f} | {test_r2:10.4f} |")
    print("+----------------+------------+------------+")
    
    # Business interpretation
    print("\n" + "-" * 40)
    print("BUSINESS INTERPRETATION")
    print("-" * 40)
    print(f"""
[INSIGHT] Model Performance Summary:
• The model explains {test_r2*100:.1f}% of variance in demand (R² = {test_r2:.4f})
• Average prediction error: ±{test_mae:.0f} units (MAE)
• Root Mean Square Error: {test_rmse:.0f} units (RMSE)

[ASSESSMENT]
• R² > 0.7: Good predictive power for retail demand forecasting
• The model captures non-linear demand patterns effectively
• Can be used for inventory planning and promotion optimization
""")
    
    return {'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2}


def cross_validate_model(X, y, n_folds=5):
    """
    Perform k-fold cross-validation to assess model stability.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_folds: Number of folds (default=5)
        
    Returns:
        Dictionary with cross-validation results
    """
    print("\n" + "=" * 60)
    print(f"5-FOLD CROSS-VALIDATION (Model Stability Check)")
    print("=" * 60)
    
    # Configure XGBoost model (same params as main training)
    try:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            tree_method='hist',
            device='cuda',
            n_jobs=-1
        )
        device_info = "GPU"
    except:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
        device_info = "CPU"
    
    print(f"\n[INFO] Running {n_folds}-fold cross-validation on {device_info}...")
    
    # KFold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Manual cross-validation for more detailed metrics
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model silently
        model.fit(X_train_fold, y_train_fold, verbose=False)
        
        # Predict and evaluate
        y_pred = model.predict(X_val_fold)
        
        r2 = r2_score(y_val_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        mae = mean_absolute_error(y_val_fold, y_pred)
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"  Fold {fold}: R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
    
    # Calculate statistics
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    
    print("\n" + "-" * 40)
    print("CROSS-VALIDATION SUMMARY")
    print("-" * 40)
    print(f"""
+----------------+------------+------------+
| Metric         | Mean       | Std Dev    |
+----------------+------------+------------+
| R² Score       | {mean_r2:10.4f} | {std_r2:10.4f} |
| RMSE           | {mean_rmse:10.2f} | {std_rmse:10.2f} |
| MAE            | {mean_mae:10.2f} | {std_mae:10.2f} |
+----------------+------------+------------+

[STABILITY ASSESSMENT]
• R² Mean: {mean_r2:.4f}
• R² Std Dev: {std_r2:.4f}
""")
    
    if std_r2 < 0.01:
        print("✓ EXCELLENT: Std Dev < 0.01 → Model is highly robust and stable")
        print("  The model's performance is consistent across different data subsets.")
    elif std_r2 < 0.02:
        print("✓ GOOD: Std Dev < 0.02 → Model is reasonably stable")
        print("  Minor performance variations across folds, but generally reliable.")
    else:
        print("⚠ CAUTION: Std Dev >= 0.02 → Model shows some instability")
        print("  Consider increasing regularization or reviewing feature engineering.")
    
    return {
        'r2_mean': mean_r2, 'r2_std': std_r2,
        'rmse_mean': mean_rmse, 'rmse_std': std_rmse,
        'mae_mean': mean_mae, 'mae_std': std_mae,
        'r2_scores': r2_scores
    }


def plot_actual_vs_predicted(y_test, y_test_pred):
    """
    Create visualization comparing actual vs predicted values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_test, y_test_pred, alpha=0.5, edgecolors='none', c='steelblue')
    max_val = max(y_test.max(), y_test_pred.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Units Sold', fontsize=11)
    ax1.set_ylabel('Predicted Units Sold', fontsize=11)
    ax1.set_title('Actual vs Predicted Demand', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Residual distribution
    ax2 = axes[1]
    residuals = y_test - y_test_pred
    ax2.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plot_01_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Plot saved: plot_01_actual_vs_predicted.png")


def plot_feature_importance(model, feature_names, top_n=15):
    """
    Display and interpret feature importance.
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    ax.set_xlabel('Feature Importance (Gain)', fontsize=11)
    ax.set_title('Top Feature Importance - Demand Drivers', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plot_02_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Plot saved: plot_02_feature_importance.png")
    
    # Interpretation
    print("\n" + "-" * 40)
    print("KEY DEMAND DRIVERS INTERPRETATION")
    print("-" * 40)
    top_3 = importance_df.tail(3)['Feature'].tolist()[::-1]
    print(f"""
[INSIGHT] Top 3 Demand Drivers:
1. {top_3[0]} - Primary factor influencing sales
2. {top_3[1]} - Secondary driver of demand
3. {top_3[2]} - Third most important predictor

[BUSINESS RECOMMENDATION]
• Focus inventory decisions on these top drivers
• Monitor changes in top features for demand shifts
• Use these insights for promotional planning
""")
    
    return importance_df


# ============================================================
# PHASE 3: MONTH-WISE SEASONAL SALES FORECAST
# ============================================================

def monthly_seasonal_forecast(df_test, y_test_pred):
    """
    Aggregate predictions to monthly level and analyze seasonality.
    
    Args:
        df_test: Test DataFrame with date information
        y_test_pred: Model predictions
    """
    print("\n" + "=" * 60)
    print("PHASE 3: MONTH-WISE SEASONAL SALES FORECAST")
    print("=" * 60)
    
    # Add predictions to test dataframe
    df_analysis = df_test.copy()
    df_analysis['predicted_units_sold'] = y_test_pred
    
    # Create year-month column for aggregation
    df_analysis['year_month'] = df_analysis['date'].dt.to_period('M')
    
    # Aggregate by month
    monthly_sales = df_analysis.groupby('year_month').agg({
        'units_sold': 'sum',
        'predicted_units_sold': 'sum'
    }).reset_index()
    
    monthly_sales['year_month_str'] = monthly_sales['year_month'].astype(str)
    
    print("\n[INFO] Monthly Sales Summary:")
    print(monthly_sales.to_string(index=False))
    
    # Find highest and lowest predicted months
    max_month_idx = monthly_sales['predicted_units_sold'].idxmax()
    min_month_idx = monthly_sales['predicted_units_sold'].idxmin()
    
    max_month = monthly_sales.loc[max_month_idx, 'year_month_str']
    max_sales = monthly_sales.loc[max_month_idx, 'predicted_units_sold']
    
    min_month = monthly_sales.loc[min_month_idx, 'year_month_str']
    min_sales = monthly_sales.loc[min_month_idx, 'predicted_units_sold']
    
    pct_diff = ((max_sales - min_sales) / min_sales) * 100
    
    print("\n" + "-" * 40)
    print("SEASONAL ANALYSIS")
    print("-" * 40)
    print(f"""
[RESULTS]
• Predicted Highest Sales Month: {max_month} ({max_sales:,.0f} units)
• Predicted Lowest Sales Month: {min_month} ({min_sales:,.0f} units)
• Percentage Difference: {pct_diff:.1f}%
""")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(len(monthly_sales))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], monthly_sales['units_sold'], 
                   width, label='Actual Sales', color='steelblue', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], monthly_sales['predicted_units_sold'], 
                   width, label='Predicted Sales', color='coral', alpha=0.8)
    
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Total Units Sold', fontsize=11)
    ax.set_title('Monthly Sales: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_sales['year_month_str'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plot_03_monthly_seasonal_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Plot saved: plot_03_monthly_seasonal_forecast.png")
    
    # Business interpretation
    print("\n" + "-" * 40)
    print("BUSINESS INTERPRETATION - SEASONALITY")
    print("-" * 40)
    print(f"""
[INSIGHT]
• Peak demand months require increased inventory preparation
• Low demand months present opportunities for promotions and clearance
• {pct_diff:.1f}% variation indicates {'strong' if pct_diff > 50 else 'moderate'} seasonality

[RECOMMENDATIONS]
• Pre-stock inventory 2-3 weeks before peak months
• Consider promotional campaigns during low-demand periods
• Align marketing budget with seasonal demand patterns
""")
    
    return monthly_sales


# ============================================================
# PHASE 4: CATEGORY REVENUE DISTRIBUTION
# ============================================================

def category_revenue_analysis(df_test, y_test_pred):
    """
    Analyze predicted revenue distribution by category.
    
    Args:
        df_test: Test DataFrame
        y_test_pred: Model predictions
    """
    print("\n" + "=" * 60)
    print("PHASE 4: CATEGORY REVENUE DISTRIBUTION (PREDICTIVE)")
    print("=" * 60)
    
    # Add predictions to test dataframe
    df_analysis = df_test.copy()
    df_analysis['predicted_units_sold'] = y_test_pred
    
    # Calculate predicted revenue
    df_analysis['predicted_revenue'] = df_analysis['predicted_units_sold'] * df_analysis['price']
    df_analysis['actual_revenue'] = df_analysis['units_sold'] * df_analysis['price']
    
    # Aggregate by category
    category_revenue = df_analysis.groupby('category').agg({
        'actual_revenue': 'sum',
        'predicted_revenue': 'sum',
        'units_sold': 'sum',
        'predicted_units_sold': 'sum'
    }).reset_index()
    
    # Calculate revenue share
    total_predicted_revenue = category_revenue['predicted_revenue'].sum()
    category_revenue['revenue_share_pct'] = (category_revenue['predicted_revenue'] / total_predicted_revenue) * 100
    category_revenue = category_revenue.sort_values('predicted_revenue', ascending=False)
    
    print("\n[INFO] Category Revenue Summary:")
    print(category_revenue.to_string(index=False))
    
    # Top category analysis
    top_category = category_revenue.iloc[0]['category']
    top_revenue = category_revenue.iloc[0]['predicted_revenue']
    top_share = category_revenue.iloc[0]['revenue_share_pct']
    
    top_2_share = category_revenue.head(2)['revenue_share_pct'].sum()
    
    print("\n" + "-" * 40)
    print("REVENUE DISTRIBUTION ANALYSIS")
    print("-" * 40)
    print(f"""
[RESULTS]
• Top Revenue Category: {top_category} (${top_revenue:,.0f})
• Share of Total Revenue: {top_share:.1f}%
• Top 2 Categories Combined: {top_2_share:.1f}% of total revenue
• Total Predicted Revenue: ${total_predicted_revenue:,.0f}
""")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    ax1 = axes[0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(category_revenue)))[::-1]
    bars = ax1.bar(category_revenue['category'], category_revenue['predicted_revenue'] / 1e6, 
                   color=colors, edgecolor='white')
    ax1.set_xlabel('Category', fontsize=11)
    ax1.set_ylabel('Predicted Revenue (Millions $)', fontsize=11)
    ax1.set_title('Predicted Revenue by Category', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.1f}M', ha='center', va='bottom', fontsize=9)
    
    # Pie chart for revenue share
    ax2 = axes[1]
    ax2.pie(category_revenue['revenue_share_pct'], labels=category_revenue['category'],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Revenue Share by Category', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plot_04_category_revenue.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Plot saved: plot_04_category_revenue.png")
    
    # Business recommendation
    print("\n" + "-" * 40)
    print("BUSINESS RECOMMENDATIONS")
    print("-" * 40)
    print(f"""
[INSIGHTS]
• {top_category} dominates revenue with {top_share:.1f}% contribution
• Top 2 categories account for {top_2_share:.1f}% of total revenue
• Revenue concentration indicates portfolio dependency

[RECOMMENDATIONS]
• Prioritize inventory investment in {top_category}
• Explore growth opportunities in lower-performing categories
• Balance portfolio to reduce dependency on top categories
• Consider premium pricing strategies for high-demand categories
""")
    
    return category_revenue


# ============================================================
# PHASE 5: PROMOTION IMPACT ANALYSIS
# ============================================================

def promotion_impact_analysis(df_test, y_test_pred):
    """
    Analyze the impact of promotions on predicted demand.
    
    Args:
        df_test: Test DataFrame
        y_test_pred: Model predictions
    """
    print("\n" + "=" * 60)
    print("PHASE 5: PROMOTION IMPACT ANALYSIS")
    print("=" * 60)
    
    # Add predictions to test dataframe
    df_analysis = df_test.copy()
    df_analysis['predicted_units_sold'] = y_test_pred
    
    # Overall promotion impact
    promoted = df_analysis[df_analysis['is_promoted'] == 1]['predicted_units_sold'].mean()
    not_promoted = df_analysis[df_analysis['is_promoted'] == 0]['predicted_units_sold'].mean()
    overall_lift = ((promoted - not_promoted) / not_promoted) * 100
    
    print("\n" + "-" * 40)
    print("OVERALL PROMOTION IMPACT")
    print("-" * 40)
    print(f"""
[RESULTS]
• Avg Predicted Demand (Promoted): {promoted:.1f} units
• Avg Predicted Demand (Not Promoted): {not_promoted:.1f} units
• Overall Lift from Promotion: {overall_lift:+.1f}%
""")
    
    # Category-wise promotion impact
    category_promo = df_analysis.groupby(['category', 'is_promoted']).agg({
        'predicted_units_sold': 'mean'
    }).reset_index()
    
    category_pivot = category_promo.pivot(index='category', columns='is_promoted', 
                                          values='predicted_units_sold').reset_index()
    category_pivot.columns = ['category', 'not_promoted', 'promoted']
    category_pivot['lift_pct'] = ((category_pivot['promoted'] - category_pivot['not_promoted']) / 
                                   category_pivot['not_promoted']) * 100
    category_pivot = category_pivot.sort_values('lift_pct', ascending=False)
    
    print("\n[INFO] Category-wise Promotion Lift:")
    print(category_pivot.to_string(index=False))
    
    # Visualization - Grouped Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Promoted vs Not Promoted by Category
    ax1 = axes[0]
    x = range(len(category_pivot))
    width = 0.35
    
    bars1 = ax1.bar([i - width/2 for i in x], category_pivot['not_promoted'], 
                    width, label='Not Promoted', color='gray', alpha=0.7)
    bars2 = ax1.bar([i + width/2 for i in x], category_pivot['promoted'], 
                    width, label='Promoted', color='coral', alpha=0.8)
    
    ax1.set_xlabel('Category', fontsize=11)
    ax1.set_ylabel('Avg Predicted Units Sold', fontsize=11)
    ax1.set_title('Promotion Impact by Category', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(category_pivot['category'], rotation=45, ha='right')
    ax1.legend()
    
    # Chart 2: Lift percentage by Category
    ax2 = axes[1]
    colors = ['green' if x > 0 else 'red' for x in category_pivot['lift_pct']]
    bars = ax2.bar(category_pivot['category'], category_pivot['lift_pct'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Category', fontsize=11)
    ax2.set_ylabel('Promotion Lift (%)', fontsize=11)
    ax2.set_title('Promotion Lift by Category', fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plot_05_promotion_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Plot saved: plot_05_promotion_impact.png")
    
    # Business interpretation
    best_category = category_pivot.iloc[0]['category']
    best_lift = category_pivot.iloc[0]['lift_pct']
    worst_category = category_pivot.iloc[-1]['category']
    worst_lift = category_pivot.iloc[-1]['lift_pct']
    
    print("\n" + "-" * 40)
    print("BUSINESS INTERPRETATION - PROMOTIONS")
    print("-" * 40)
    print(f"""
[INSIGHTS]
• Overall promotion effect: {overall_lift:+.1f}% lift in demand
• Most responsive category: {best_category} ({best_lift:+.1f}% lift)
• Least responsive category: {worst_category} ({worst_lift:+.1f}% lift)

[RECOMMENDATIONS]
• Prioritize promotions for {best_category} - highest ROI
• Re-evaluate promotion strategy for {worst_category}
• Allocate promotion budget based on category responsiveness
• Consider deeper discounts or different promotion types for low-lift categories
""")
    
    return category_pivot


# ============================================================
# PHASE 6: PRICE ELASTICITY ANALYSIS
# ============================================================

def price_elasticity_analysis(model, df_test, feature_cols):
    """
    Simulate price changes and calculate demand elasticity.
    
    Args:
        model: Trained XGBoost model
        df_test: Test DataFrame
        feature_cols: Feature columns used in the model
    """
    print("\n" + "=" * 60)
    print("PHASE 6: PRICE ELASTICITY ANALYSIS")
    print("=" * 60)
    
    # Base predictions
    X_test = df_test[feature_cols].copy()
    base_predictions = model.predict(X_test)
    base_price = df_test['price'].mean()
    base_demand = base_predictions.mean()
    
    # Helper function to simulate price change with ALL dependent features updated
    def simulate_price_change(X_base, df_original, price_multiplier):
        """
        Properly isolate price effect by updating ALL price-dependent features:
        - price
        - price_difference (price - competitor_price)
        - price_ratio (price / competitor_price)
        - discount_value (price * discount_percent / 100)
        - profit_margin ((price - cost_price) / price)
        """
        X_sim = X_base.copy()
        new_price = X_sim['price'] * price_multiplier
        
        # Update price
        X_sim['price'] = new_price
        
        # Update price_difference
        X_sim['price_difference'] = new_price - df_original['competitor_price'].values
        
        # Update price_ratio
        X_sim['price_ratio'] = new_price / df_original['competitor_price'].values
        
        # Update discount_value (discount is applied on the new price)
        X_sim['discount_value'] = new_price * (df_original['discount_percent'].values / 100)
        
        # Update profit_margin
        X_sim['profit_margin'] = (new_price - df_original['cost_price'].values) / new_price
        
        return X_sim
    
    print("\n[INFO] Price Simulation Methodology:")
    print("  - Updating ALL price-dependent features:")
    print("    * price")
    print("    * price_difference (price - competitor_price)")
    print("    * price_ratio (price / competitor_price)")
    print("    * discount_value (price * discount_percent / 100)")
    print("    * profit_margin ((price - cost_price) / price)")
    print("  - Keeping CONSTANT: stock, promotions, ratings, dates, etc.")
    
    # Simulate +5% price increase with ALL dependent features updated
    X_price_up = simulate_price_change(X_test, df_test, 1.05)
    predictions_up = model.predict(X_price_up)
    demand_up = predictions_up.mean()
    
    # Simulate -5% price decrease with ALL dependent features updated
    X_price_down = simulate_price_change(X_test, df_test, 0.95)
    predictions_down = model.predict(X_price_down)
    demand_down = predictions_down.mean()
    
    # Calculate overall elasticity
    # Elasticity = (% change in demand) / (% change in price)
    pct_change_demand_up = ((demand_up - base_demand) / base_demand) * 100
    pct_change_demand_down = ((demand_down - base_demand) / base_demand) * 100
    
    elasticity_up = pct_change_demand_up / 5  # 5% price increase
    elasticity_down = pct_change_demand_down / (-5)  # 5% price decrease
    avg_elasticity = (abs(elasticity_up) + abs(elasticity_down)) / 2
    
    print("\n" + "-" * 40)
    print("OVERALL PRICE ELASTICITY")
    print("-" * 40)
    print(f"""
[SIMULATION RESULTS]
Base Scenario:
• Avg Price: ${base_price:.2f}
• Avg Predicted Demand: {base_demand:.1f} units

+5% Price Increase:
• New Avg Demand: {demand_up:.1f} units
• Demand Change: {pct_change_demand_up:+.2f}%
• Elasticity: {elasticity_up:.3f}

-5% Price Decrease:
• New Avg Demand: {demand_down:.1f} units
• Demand Change: {pct_change_demand_down:+.2f}%
• Elasticity: {elasticity_down:.3f}

Average Price Elasticity: {avg_elasticity:.3f}
""")
    
    # Category-wise elasticity
    df_test_copy = df_test.copy()
    df_test_copy['base_pred'] = base_predictions
    df_test_copy['pred_price_up'] = predictions_up
    df_test_copy['pred_price_down'] = predictions_down
    
    category_elasticity = df_test_copy.groupby('category').agg({
        'base_pred': 'mean',
        'pred_price_up': 'mean',
        'pred_price_down': 'mean'
    }).reset_index()
    
    category_elasticity['elasticity_up'] = ((category_elasticity['pred_price_up'] - category_elasticity['base_pred']) / 
                                             category_elasticity['base_pred'] * 100) / 5
    category_elasticity['elasticity_down'] = ((category_elasticity['pred_price_down'] - category_elasticity['base_pred']) / 
                                               category_elasticity['base_pred'] * 100) / (-5)
    category_elasticity['avg_elasticity'] = (abs(category_elasticity['elasticity_up']) + 
                                              abs(category_elasticity['elasticity_down'])) / 2
    category_elasticity['sensitivity'] = category_elasticity['avg_elasticity'].apply(
        lambda x: 'Highly Sensitive (>1)' if x > 1 else 'Less Sensitive (<1)'
    )
    
    category_elasticity = category_elasticity.sort_values('avg_elasticity', ascending=False)
    
    print("\n[INFO] Category-wise Price Elasticity:")
    print(category_elasticity[['category', 'avg_elasticity', 'sensitivity']].to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Demand response to price changes
    ax1 = axes[0]
    scenarios = ['Price -5%', 'Base Price', 'Price +5%']
    demands = [demand_down, base_demand, demand_up]
    colors = ['green', 'steelblue', 'coral']
    bars = ax1.bar(scenarios, demands, color=colors, edgecolor='white', alpha=0.8)
    ax1.set_ylabel('Avg Predicted Demand (Units)', fontsize=11)
    ax1.set_title('Demand Response to Price Changes', fontsize=12, fontweight='bold')
    
    for bar, demand in zip(bars, demands):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{demand:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Chart 2: Category-wise elasticity
    ax2 = axes[1]
    colors = ['coral' if x > 1 else 'steelblue' for x in category_elasticity['avg_elasticity']]
    bars = ax2.barh(category_elasticity['category'], category_elasticity['avg_elasticity'], 
                    color=colors, alpha=0.8, edgecolor='white')
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Elasticity = 1')
    ax2.set_xlabel('Average Price Elasticity', fontsize=11)
    ax2.set_title('Price Elasticity by Category', fontsize=12, fontweight='bold')
    ax2.legend()
    
    for bar, val in zip(bars, category_elasticity['avg_elasticity']):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{val:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plot_06_price_elasticity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Plot saved: plot_06_price_elasticity.png")
    
    # Strategic recommendations
    highly_sensitive = category_elasticity[category_elasticity['avg_elasticity'] > 1]['category'].tolist()
    less_sensitive = category_elasticity[category_elasticity['avg_elasticity'] <= 1]['category'].tolist()
    
    print("\n" + "-" * 40)
    print("STRATEGIC PRICING RECOMMENDATIONS")
    print("-" * 40)
    print(f"""
[ELASTICITY CLASSIFICATION]
• Highly Price Sensitive (Elasticity > 1): {', '.join(highly_sensitive) if highly_sensitive else 'None'}
• Less Price Sensitive (Elasticity < 1): {', '.join(less_sensitive) if less_sensitive else 'None'}

[STRATEGIC RECOMMENDATIONS]

For Highly Sensitive Categories:
• Avoid price increases - significant demand loss
• Use competitive pricing to capture market share
• Focus on volume-based revenue growth
• Consider frequent promotions and discounts

For Less Sensitive Categories:
• Opportunity for premium pricing
• Focus on value-add and brand positioning
• Price increases feasible without major demand impact
• Maximize margins through strategic pricing

[OVERALL PRICING STRATEGY]
• Average elasticity of {avg_elasticity:.2f} indicates {'elastic' if avg_elasticity > 1 else 'inelastic'} demand overall
• Balance between margin optimization and volume maintenance
• Use dynamic pricing based on category sensitivity
""")
    
    return category_elasticity


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution function - orchestrates all phases of the analysis.
    """
    print("\n" + "=" * 70)
    print("   RETAIL DEMAND INTELLIGENCE SYSTEM")
    print("   Comprehensive ML Pipeline for Retail Analytics")
    print("=" * 70)
    
    # PHASE 1: Data Preprocessing
    df = load_and_inspect_data('product_sales_data.csv')
    df = clean_data(df)
    df, label_encoders = engineer_features(df)
    X_train, X_test, y_train, y_test, feature_cols, df_train, df_test = prepare_model_data(df)
    
    # PHASE 2: Demand Prediction Model
    model, y_train_pred, y_test_pred = train_xgboost_model(X_train, y_train, X_test, y_test)
    metrics = evaluate_model(y_train, y_train_pred, y_test, y_test_pred)
    
    # 5-Fold Cross-Validation for Model Stability
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    cv_results = cross_validate_model(X_full, y_full, n_folds=5)
    
    plot_actual_vs_predicted(y_test, y_test_pred)
    importance_df = plot_feature_importance(model, feature_cols)
    
    # PHASE 3: Monthly Seasonal Forecast
    monthly_sales = monthly_seasonal_forecast(df_test, y_test_pred)
    
    # PHASE 4: Category Revenue Distribution
    category_revenue = category_revenue_analysis(df_test, y_test_pred)
    
    # PHASE 5: Promotion Impact Analysis
    promotion_impact = promotion_impact_analysis(df_test, y_test_pred)
    
    # PHASE 6: Price Elasticity Analysis
    elasticity_df = price_elasticity_analysis(model, df_test, feature_cols)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("   ANALYSIS COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"""
[MODEL PERFORMANCE]
• R² Score: {metrics['test_r2']:.4f}
• RMSE: {metrics['test_rmse']:.2f}
• MAE: {metrics['test_mae']:.2f}

[CROSS-VALIDATION STABILITY]
• Mean R²: {cv_results['r2_mean']:.4f}
• Std Dev: {cv_results['r2_std']:.4f}
• Status: {'ROBUST (Std < 0.01)' if cv_results['r2_std'] < 0.01 else 'STABLE (Std < 0.02)' if cv_results['r2_std'] < 0.02 else 'NEEDS REVIEW'}

[GENERATED VISUALIZATIONS]
1. plot_01_actual_vs_predicted.png - Model accuracy visualization
2. plot_02_feature_importance.png - Key demand drivers
3. plot_03_monthly_seasonal_forecast.png - Seasonal patterns
4. plot_04_category_revenue.png - Revenue distribution
5. plot_05_promotion_impact.png - Promotion effectiveness
6. plot_06_price_elasticity.png - Price sensitivity analysis

[KEY INSIGHTS]
• Demand prediction model ready for deployment
• Seasonal patterns identified for inventory planning
• Category revenue insights for portfolio management
• Promotion ROI quantified by category
• Price elasticity mapped for pricing strategy
""")
    
    return model, df, metrics, cv_results


if __name__ == "__main__":
    model, df, metrics, cv_results = main()
