"""
Retail Demand Model - Without Stock Feature
=============================================
Test model robustness by removing stock_available feature.
This helps determine if the model is inventory-driven.

Author: ML Engineer
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
import os

warnings.filterwarnings('ignore')
os.chdir(r"D:\gtbit hackathon")

print("=" * 70)
print("   ROBUSTNESS TEST: MODEL WITHOUT STOCK_AVAILABLE")
print("   Testing if model is inventory-driven")
print("=" * 70)

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

print("\n[INFO] Loading data...")
df = pd.read_csv('product_sales_data.csv')
print(f"[SUCCESS] Loaded {len(df)} rows")

# Data cleaning
df = df.drop_duplicates()
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
df['year'] = df['date'].dt.year
df['month_num'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

df['price_difference'] = df['price'] - df['competitor_price']
df['discount_value'] = df['price'] * (df['discount_percent'] / 100)
df['price_ratio'] = df['price'] / df['competitor_price']
df['profit_margin'] = (df['price'] - df['cost_price']) / df['price']

# Encode categoricals
label_encoders = {}
for col in ['category', 'season', 'day_of_week']:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# ============================================================
# DEFINE FEATURES - WITHOUT STOCK_AVAILABLE
# ============================================================

# Original features (22 total)
original_features = [
    'price', 'cost_price', 'discount_percent', 'stock_available',
    'competitor_price', 'rating', 'review_count', 'is_promoted',
    'is_weekend', 'days_since_launch', 'year', 'month_num', 'day',
    'week_of_year', 'price_difference', 'discount_value',
    'price_ratio', 'profit_margin', 'category_encoded',
    'season_encoded', 'day_of_week_encoded', 'is_holiday'
]

# Features WITHOUT stock_available (21 total)
features_no_stock = [
    'price', 'cost_price', 'discount_percent',  # Removed stock_available
    'competitor_price', 'rating', 'review_count', 'is_promoted',
    'is_weekend', 'days_since_launch', 'year', 'month_num', 'day',
    'week_of_year', 'price_difference', 'discount_value',
    'price_ratio', 'profit_margin', 'category_encoded',
    'season_encoded', 'day_of_week_encoded', 'is_holiday'
]

print(f"\n[INFO] Original model: {len(original_features)} features")
print(f"[INFO] No-stock model: {len(features_no_stock)} features")
print(f"[INFO] Removed: stock_available")

# ============================================================
# PREPARE DATA SPLITS
# ============================================================

df = df.sort_values('date').reset_index(drop=True)
split_idx = int(len(df) * 0.8)

df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]

target_col = 'units_sold'

print(f"\n[INFO] Training: {len(df_train)} samples")
print(f"[INFO] Test: {len(df_test)} samples")

# ============================================================
# TRAIN MODEL WITHOUT STOCK
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL WITHOUT STOCK_AVAILABLE")
print("=" * 60)

X_train = df_train[features_no_stock]
y_train = df_train[target_col]
X_test = df_test[features_no_stock]
y_test = df_test[target_col]

# Try GPU, fallback to CPU
try:
    model_no_stock = xgb.XGBRegressor(
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
    print("\n[INFO] Training on GPU...")
    model_no_stock.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("[SUCCESS] Model trained on GPU!")
except:
    model_no_stock = xgb.XGBRegressor(
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
    print("\n[INFO] Training on CPU...")
    model_no_stock.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("[SUCCESS] Model trained on CPU!")

# Predictions
y_train_pred = model_no_stock.predict(X_train)
y_test_pred = model_no_stock.predict(X_test)

# Evaluate
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n" + "-" * 40)
print("MODEL PERFORMANCE (WITHOUT STOCK)")
print("-" * 40)

print(f"""
+----------------+------------+------------+
| Metric         |   Train    |    Test    |
+----------------+------------+------------+
| RMSE           | {train_rmse:10.2f} | {test_rmse:10.2f} |
| MAE            | {train_mae:10.2f} | {test_mae:10.2f} |
| R² Score       | {train_r2:10.4f} | {test_r2:10.4f} |
+----------------+------------+------------+
""")

# ============================================================
# COMPARISON WITH ORIGINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON: ORIGINAL VS NO-STOCK MODEL")
print("=" * 60)

# Original model metrics (from previous runs)
orig_train_r2 = 0.9961
orig_test_r2 = 0.9868
orig_train_rmse = 8.22
orig_test_rmse = 14.28
orig_train_mae = 5.31
orig_test_mae = 7.02

r2_drop = orig_test_r2 - test_r2
r2_drop_pct = (r2_drop / orig_test_r2) * 100
rmse_increase = test_rmse - orig_test_rmse
rmse_increase_pct = (rmse_increase / orig_test_rmse) * 100

print(f"""
+-------------------------+-------------+-------------+-------------+
| Metric                  | Original    | No Stock    | Change      |
+-------------------------+-------------+-------------+-------------+
| Test R²                 | {orig_test_r2:11.4f} | {test_r2:11.4f} | {-r2_drop:+11.4f} |
| Test RMSE               | {orig_test_rmse:11.2f} | {test_rmse:11.2f} | {rmse_increase:+11.2f} |
| Test MAE                | {orig_test_mae:11.2f} | {test_mae:11.2f} | {test_mae - orig_test_mae:+11.2f} |
+-------------------------+-------------+-------------+-------------+
| R² Drop                 |             |             | {r2_drop_pct:10.1f}% |
| RMSE Increase           |             |             | {rmse_increase_pct:10.1f}% |
+-------------------------+-------------+-------------+-------------+
""")

# ============================================================
# 5-FOLD CROSS-VALIDATION (NO STOCK)
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION (NO STOCK MODEL)")
print("=" * 60)

X_full = df[features_no_stock]
y_full = df[target_col]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = []
cv_rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_full), 1):
    X_fold_train = X_full.iloc[train_idx]
    y_fold_train = y_full.iloc[train_idx]
    X_fold_val = X_full.iloc[val_idx]
    y_fold_val = y_full.iloc[val_idx]
    
    model_no_stock.fit(X_fold_train, y_fold_train, verbose=False)
    y_pred = model_no_stock.predict(X_fold_val)
    
    fold_r2 = r2_score(y_fold_val, y_pred)
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
    
    cv_r2_scores.append(fold_r2)
    cv_rmse_scores.append(fold_rmse)
    print(f"  Fold {fold}: R² = {fold_r2:.4f}, RMSE = {fold_rmse:.2f}")

cv_mean_r2 = np.mean(cv_r2_scores)
cv_std_r2 = np.std(cv_r2_scores)

print(f"""
Cross-Validation Summary:
• Mean R²: {cv_mean_r2:.4f}
• Std R²: {cv_std_r2:.4f}
• Status: {'ROBUST' if cv_std_r2 < 0.01 else 'STABLE' if cv_std_r2 < 0.02 else 'VARIABLE'}
""")

# ============================================================
# FEATURE IMPORTANCE (NO STOCK MODEL)
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (NO STOCK MODEL)")
print("=" * 60)

# Retrain final model
model_no_stock.fit(X_train, y_train, verbose=False)

importance = pd.DataFrame({
    'feature': features_no_stock,
    'importance': model_no_stock.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features (No Stock Model):")
print(importance.head(10).to_string(index=False))

# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION: IS THE MODEL INVENTORY-DRIVEN?")
print("=" * 60)

if r2_drop_pct > 30:
    verdict = "YES - HEAVILY INVENTORY-DRIVEN"
    explanation = f"""
The model R² dropped by {r2_drop_pct:.1f}% when stock_available was removed.
This confirms that inventory levels are the PRIMARY driver of demand in this data.

IMPLICATIONS:
• The original model is essentially predicting "what was available to sell"
• True demand signal may be masked by stock constraints
• For genuine demand forecasting, you'd need:
  - Out-of-stock indicators
  - Lost sales estimation
  - Uncensored demand modeling
"""
elif r2_drop_pct > 15:
    verdict = "MODERATELY INVENTORY-DRIVEN"
    explanation = f"""
The model R² dropped by {r2_drop_pct:.1f}% when stock_available was removed.
Stock is significant but not the only predictive factor.

IMPLICATIONS:
• Other features (promotions, ratings, etc.) still contribute
• Model captures both supply and demand signals
• Consider using stock as a constraint, not a predictor
"""
elif r2_drop_pct > 5:
    verdict = "PARTIALLY INVENTORY-INFLUENCED"
    explanation = f"""
The model R² dropped by {r2_drop_pct:.1f}% when stock_available was removed.
Stock contributes but model relies on other features too.

IMPLICATIONS:
• Model is reasonably balanced
• Can be used for demand forecasting with caution
• Stock effect is present but not dominant
"""
else:
    verdict = "NO - NOT INVENTORY-DRIVEN"
    explanation = f"""
The model R² only dropped by {r2_drop_pct:.1f}% when stock_available was removed.
Other features are the primary predictors.

IMPLICATIONS:
• Model captures genuine demand drivers
• Safe to use for demand forecasting
• Stock availability doesn't dominate predictions
"""

print(f"""
VERDICT: {verdict}

R² DROP: {r2_drop_pct:.1f}%
Original R²: {orig_test_r2:.4f}
No-Stock R²: {test_r2:.4f}
{explanation}
""")

# ============================================================
# SAVE MODEL AND ARTIFACTS
# ============================================================

model_no_stock.save_model('demand_model_no_stock.json')
joblib.dump(label_encoders, 'label_encoders_no_stock.pkl')
joblib.dump(features_no_stock, 'feature_columns_no_stock.pkl')

print("[INFO] Saved model artifacts:")
print("  - demand_model_no_stock.json")
print("  - label_encoders_no_stock.pkl")
print("  - feature_columns_no_stock.pkl")

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model Comparison
ax1 = axes[0]
metrics = ['R² Score', 'RMSE', 'MAE']
original_vals = [orig_test_r2, orig_test_rmse / 100, orig_test_mae / 10]  # Scaled for visibility
no_stock_vals = [test_r2, test_rmse / 100, test_mae / 10]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, original_vals, width, label='Original (with stock)', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, no_stock_vals, width, label='No Stock', color='coral', alpha=0.8)

ax1.set_ylabel('Score (scaled)')
ax1.set_title('Model Performance Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()

# Add value labels
for bars, vals in [(bars1, [orig_test_r2, orig_test_rmse, orig_test_mae]), 
                   (bars2, [test_r2, test_rmse, test_mae])]:
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Feature Importance (No Stock)
ax2 = axes[1]
top_features = importance.head(10)
colors = ['coral' if 'price' in f else 'steelblue' for f in top_features['feature']]
ax2.barh(top_features['feature'], top_features['importance'], color=colors, alpha=0.8)
ax2.set_xlabel('Importance')
ax2.set_title('Top 10 Features (No Stock Model)', fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('plot_07_no_stock_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[INFO] Plot saved: plot_07_no_stock_comparison.png")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
