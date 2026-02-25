"""
LightGBM Model 2: Two-Group Model (Helsingfors vs Norway)
==========================================================

This script trains two separate LightGBM models:
- Model A: Helsingfors (location_id = 0) - Finland
- Model B: Norwegian cities (location_id = 1-5) - Norway

Location mapping:
- 0: Helsingfors (Finland) → Model A
- 1: Oslo (Norway) → Model B
- 2: Stavanger (Norway) → Model B
- 3: Trondheim (Norway) → Model B
- 4: Tromsø (Norway) → Model B
- 5: Bergen (Norway) → Model B
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from helpers.data_retrieval import load_preprocessed_data
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LIGHTGBM MODEL 2: TWO-GROUP MODEL (HELSINGFORS vs NORWAY)")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n1. Loading preprocessed data...")
df = load_preprocessed_data()

print(f"   Dataset shape: {df.shape}")
print(f"   Locations: {sorted(df['location_id'].unique())}")

# ============================================================================
# 2. SPLIT DATA BY REGION
# ============================================================================

print("\n2. Splitting data by region...")

# Helsingfors (Finland)
df_helsingfors = df[df['location_id'] == 0].copy()
print(f"   Helsingfors (ID=0): {len(df_helsingfors):,} samples")

# Norwegian cities (location_id 1-5)
df_norway = df[df['location_id'].isin([1, 2, 3, 4, 5])].copy()
print(f"   Norway (ID=1-5):    {len(df_norway):,} samples")

# ============================================================================
# 3. PREPARE FEATURES FOR EACH REGION
# ============================================================================

print("\n3. Preparing features and targets...")

# Feature columns (excluding location_id since we're modeling separately)
feature_cols = ['month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'temperature']
feature_cols_norway = ['month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'location_id', 'temperature']

# Helsingfors (no location_id needed - only one location)
X_hel = df_helsingfors[feature_cols].copy()
y_hel = df_helsingfors['consumption'].copy()

# Norway (keep location_id to distinguish between cities)
X_nor = df_norway[feature_cols_norway].copy()
y_nor = df_norway['consumption'].copy()

print(f"\n   Helsingfors features: {feature_cols}")
print(f"   Norway features: {feature_cols_norway}")

# ============================================================================
# 4. TRAIN-TEST SPLIT FOR EACH REGION
# ============================================================================

print("\n4. Splitting data (70% train, 15% val, 15% test)...")

# --- Helsingfors Split ---
n_hel = len(df_helsingfors)
train_idx_hel = int(0.70 * n_hel)
val_idx_hel = int(0.85 * n_hel)

X_train_hel = X_hel.iloc[:train_idx_hel]
y_train_hel = y_hel.iloc[:train_idx_hel]
X_val_hel = X_hel.iloc[train_idx_hel:val_idx_hel]
y_val_hel = y_hel.iloc[train_idx_hel:val_idx_hel]
X_test_hel = X_hel.iloc[val_idx_hel:]
y_test_hel = y_hel.iloc[val_idx_hel:]

print(f"\n   HELSINGFORS:")
print(f"      Training:   {X_train_hel.shape[0]:,} samples")
print(f"      Validation: {X_val_hel.shape[0]:,} samples")
print(f"      Test:       {X_test_hel.shape[0]:,} samples")

# --- Norway Split ---
n_nor = len(df_norway)
train_idx_nor = int(0.70 * n_nor)
val_idx_nor = int(0.85 * n_nor)

X_train_nor = X_nor.iloc[:train_idx_nor]
y_train_nor = y_nor.iloc[:train_idx_nor]
X_val_nor = X_nor.iloc[train_idx_nor:val_idx_nor]
y_val_nor = y_nor.iloc[train_idx_nor:val_idx_nor]
X_test_nor = X_nor.iloc[val_idx_nor:]
y_test_nor = y_nor.iloc[val_idx_nor:]

print(f"\n   NORWAY:")
print(f"      Training:   {X_train_nor.shape[0]:,} samples")
print(f"      Validation: {X_val_nor.shape[0]:,} samples")
print(f"      Test:       {X_test_nor.shape[0]:,} samples")

# ============================================================================
# 5. CREATE LIGHTGBM DATASETS
# ============================================================================

print("\n5. Creating LightGBM datasets...")

# Helsingfors datasets
train_data_hel = lgb.Dataset(X_train_hel, label=y_train_hel, feature_name=feature_cols)
val_data_hel = lgb.Dataset(X_val_hel, label=y_val_hel, reference=train_data_hel, feature_name=feature_cols)

# Norway datasets
train_data_nor = lgb.Dataset(X_train_nor, label=y_train_nor, feature_name=feature_cols_norway)
val_data_nor = lgb.Dataset(X_val_nor, label=y_val_nor, reference=train_data_nor, feature_name=feature_cols_norway)

# ============================================================================
# 6. SET HYPERPARAMETERS
# ============================================================================

print("\n6. Configuring hyperparameters...")

params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42
}

print("   Using same parameters for both models:")
for key, value in params.items():
    if key != 'verbose':
        print(f"      {key}: {value}")

# ============================================================================
# 7. TRAIN MODEL A - HELSINGFORS
# ============================================================================

print("\n" + "=" * 80)
print("7. Training Model A: HELSINGFORS")
print("=" * 80)

start_time_hel = datetime.now()

model_hel = lgb.train(
    params,
    train_data_hel,
    num_boost_round=1000,
    valid_sets=[train_data_hel, val_data_hel],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

end_time_hel = datetime.now()
training_time_hel = (end_time_hel - start_time_hel).total_seconds()

print(f"\n   Training completed in {training_time_hel:.2f} seconds")
print(f"   Best iteration: {model_hel.best_iteration}")
print(f"   Best validation MAE: {model_hel.best_score['valid']['l1']:.4f}")

# Save Helsingfors model
model_hel.save_model('model_helsingfors.txt')
print("   ✓ Model saved as: model_helsingfors.txt")

# ============================================================================
# 8. TRAIN MODEL B - NORWAY
# ============================================================================

print("\n" + "=" * 80)
print("8. Training Model B: NORWAY (5 cities)")
print("=" * 80)

start_time_nor = datetime.now()

model_nor = lgb.train(
    params,
    train_data_nor,
    num_boost_round=1000,
    valid_sets=[train_data_nor, val_data_nor],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

end_time_nor = datetime.now()
training_time_nor = (end_time_nor - start_time_nor).total_seconds()

print(f"\n   Training completed in {training_time_nor:.2f} seconds")
print(f"   Best iteration: {model_nor.best_iteration}")
print(f"   Best validation MAE: {model_nor.best_score['valid']['l1']:.4f}")

# Save Norway model
model_nor.save_model('model_norway.txt')
print("   ✓ Model saved as: model_norway.txt")

# ============================================================================
# 9. EVALUATE MODEL A - HELSINGFORS
# ============================================================================

print("\n" + "=" * 80)
print("9. Evaluating Model A: HELSINGFORS")
print("=" * 80)

# Predictions
y_train_pred_hel = model_hel.predict(X_train_hel, num_iteration=model_hel.best_iteration)
y_val_pred_hel = model_hel.predict(X_val_hel, num_iteration=model_hel.best_iteration)
y_test_pred_hel = model_hel.predict(X_test_hel, num_iteration=model_hel.best_iteration)

def calculate_metrics(y_true, y_pred, set_name):
    """Calculate and print evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 0.1))) * 100  # Avoid division by zero
    
    print(f"\n{set_name} Set:")
    print(f"   MAE:  {mae:.4f} MW")
    print(f"   RMSE: {rmse:.4f} MW")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

train_metrics_hel = calculate_metrics(y_train_hel, y_train_pred_hel, "Training")
val_metrics_hel = calculate_metrics(y_val_hel, y_val_pred_hel, "Validation")
test_metrics_hel = calculate_metrics(y_test_hel, y_test_pred_hel, "Test")

# ============================================================================
# 10. EVALUATE MODEL B - NORWAY
# ============================================================================

print("\n" + "=" * 80)
print("10. Evaluating Model B: NORWAY")
print("=" * 80)

# Predictions
y_train_pred_nor = model_nor.predict(X_train_nor, num_iteration=model_nor.best_iteration)
y_val_pred_nor = model_nor.predict(X_val_nor, num_iteration=model_nor.best_iteration)
y_test_pred_nor = model_nor.predict(X_test_nor, num_iteration=model_nor.best_iteration)

train_metrics_nor = calculate_metrics(y_train_nor, y_train_pred_nor, "Training")
val_metrics_nor = calculate_metrics(y_val_nor, y_val_pred_nor, "Validation")
test_metrics_nor = calculate_metrics(y_test_nor, y_test_pred_nor, "Test")

# Performance by Norwegian city
print("\nTest Set Performance by Norwegian City:")
location_names = {1: 'Oslo', 2: 'Stavanger', 3: 'Trondheim', 4: 'Tromsø', 5: 'Bergen'}

test_results_nor = pd.DataFrame({
    'location_id': X_test_nor['location_id'],
    'actual': y_test_nor,
    'predicted': y_test_pred_nor
})

for loc_id in sorted([1, 2, 3, 4, 5]):
    loc_data = test_results_nor[test_results_nor['location_id'] == loc_id]
    if len(loc_data) > 0:
        mae = mean_absolute_error(loc_data['actual'], loc_data['predicted'])
        rmse = np.sqrt(mean_squared_error(loc_data['actual'], loc_data['predicted']))
        print(f"   {location_names[loc_id]:12s} (ID={loc_id}): MAE={mae:.4f} MW, RMSE={rmse:.4f} MW")

# ============================================================================
# 11. COMBINED EVALUATION (WEIGHTED BY SAMPLE SIZE)
# ============================================================================

print("\n" + "=" * 80)
print("11. COMBINED TEST SET PERFORMANCE (Both Models)")
print("=" * 80)

# Combine predictions
n_test_hel = len(y_test_hel)
n_test_nor = len(y_test_nor)
n_test_total = n_test_hel + n_test_nor

# Calculate weighted average metrics
combined_mae = (test_metrics_hel['mae'] * n_test_hel + test_metrics_nor['mae'] * n_test_nor) / n_test_total
combined_rmse = np.sqrt((test_metrics_hel['rmse']**2 * n_test_hel + test_metrics_nor['rmse']**2 * n_test_nor) / n_test_total)

print(f"\nWeighted Average (Test Set):")
print(f"   MAE:  {combined_mae:.4f} MW")
print(f"   RMSE: {combined_rmse:.4f} MW")
print(f"\nBreakdown:")
print(f"   Helsingfors: {n_test_hel:,} samples, MAE={test_metrics_hel['mae']:.4f} MW")
print(f"   Norway:      {n_test_nor:,} samples, MAE={test_metrics_nor['mae']:.4f} MW")

# ============================================================================
# 12. FEATURE IMPORTANCE
# ============================================================================

print("\n12. Feature Importance Analysis")
print("-" * 80)

print("\nModel A (Helsingfors) - Feature Importance:")
importance_hel = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_hel.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

for idx, row in importance_hel.iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:8.0f}")

print("\nModel B (Norway) - Feature Importance:")
importance_nor = pd.DataFrame({
    'feature': feature_cols_norway,
    'importance': model_nor.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

for idx, row in importance_nor.iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:8.0f}")

# ============================================================================
# 13. VISUALIZATIONS
# ============================================================================

print("\n13. Creating visualizations...")

fig = plt.figure(figsize=(18, 12))

# 13.1 Feature Importance - Helsingfors
ax1 = plt.subplot(3, 3, 1)
importance_hel_plot = importance_hel.sort_values('importance', ascending=True)
ax1.barh(importance_hel_plot['feature'], importance_hel_plot['importance'], color='steelblue')
ax1.set_xlabel('Importance (Gain)')
ax1.set_title('Model A: Helsingfors Feature Importance')
ax1.grid(True, alpha=0.3, axis='x')

# 13.2 Feature Importance - Norway
ax2 = plt.subplot(3, 3, 2)
importance_nor_plot = importance_nor.sort_values('importance', ascending=True)
ax2.barh(importance_nor_plot['feature'], importance_nor_plot['importance'], color='coral')
ax2.set_xlabel('Importance (Gain)')
ax2.set_title('Model B: Norway Feature Importance')
ax2.grid(True, alpha=0.3, axis='x')

# 13.3 Comparison of Test MAE
ax3 = plt.subplot(3, 3, 3)
models = ['Helsingfors', 'Norway', 'Combined\n(Weighted)']
maes = [test_metrics_hel['mae'], test_metrics_nor['mae'], combined_mae]
colors = ['steelblue', 'coral', 'green']
ax3.bar(models, maes, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Test MAE (MW)')
ax3.set_title('Test Set Performance Comparison')
ax3.grid(True, alpha=0.3, axis='y')
for i, mae in enumerate(maes):
    ax3.text(i, mae, f'{mae:.3f}', ha='center', va='bottom', fontweight='bold')

# 13.4 Actual vs Predicted - Helsingfors
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(y_test_hel, y_test_pred_hel, alpha=0.5, s=20, color='steelblue')
ax4.plot([y_test_hel.min(), y_test_hel.max()], [y_test_hel.min(), y_test_hel.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Consumption (MW)')
ax4.set_ylabel('Predicted Consumption (MW)')
ax4.set_title('Model A: Helsingfors (Actual vs Predicted)')
ax4.grid(True, alpha=0.3)

# 13.5 Actual vs Predicted - Norway
ax5 = plt.subplot(3, 3, 5)
sample_size = min(3000, len(y_test_nor))
sample_idx = np.random.choice(len(y_test_nor), sample_size, replace=False)
ax5.scatter(y_test_nor.iloc[sample_idx], y_test_pred_nor[sample_idx], alpha=0.5, s=20, color='coral')
ax5.plot([y_test_nor.min(), y_test_nor.max()], [y_test_nor.min(), y_test_nor.max()], 'r--', lw=2)
ax5.set_xlabel('Actual Consumption (MW)')
ax5.set_ylabel('Predicted Consumption (MW)')
ax5.set_title(f'Model B: Norway (Actual vs Predicted, n={sample_size})')
ax5.grid(True, alpha=0.3)

# 13.6 Residuals - Helsingfors
ax6 = plt.subplot(3, 3, 6)
residuals_hel = y_test_hel.values - y_test_pred_hel
ax6.hist(residuals_hel, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
ax6.axvline(0, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('Residuals (MW)')
ax6.set_ylabel('Frequency')
ax6.set_title(f'Model A: Helsingfors Residuals (μ={residuals_hel.mean():.3f})')
ax6.grid(True, alpha=0.3)

# 13.7 Residuals - Norway
ax7 = plt.subplot(3, 3, 7)
residuals_nor = y_test_nor.values - y_test_pred_nor
ax7.hist(residuals_nor, bins=40, edgecolor='black', alpha=0.7, color='coral')
ax7.axvline(0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Residuals (MW)')
ax7.set_ylabel('Frequency')
ax7.set_title(f'Model B: Norway Residuals (μ={residuals_nor.mean():.3f})')
ax7.grid(True, alpha=0.3)

# 13.8 Time series - Helsingfors
ax8 = plt.subplot(3, 3, 8)
plot_range_hel = slice(0, min(300, len(y_test_hel)))
ax8.plot(y_test_hel.iloc[plot_range_hel].values, label='Actual', linewidth=1.5, alpha=0.7)
ax8.plot(y_test_pred_hel[plot_range_hel], label='Predicted', linewidth=1.5, alpha=0.7)
ax8.set_xlabel('Time (hours)')
ax8.set_ylabel('Consumption (MW)')
ax8.set_title('Model A: Helsingfors Test Set (First 300h)')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 13.9 MAE by Norwegian City
ax9 = plt.subplot(3, 3, 9)
city_maes = []
city_names = []
for loc_id in sorted([1, 2, 3, 4, 5]):
    loc_data = test_results_nor[test_results_nor['location_id'] == loc_id]
    if len(loc_data) > 0:
        mae = mean_absolute_error(loc_data['actual'], loc_data['predicted'])
        city_maes.append(mae)
        city_names.append(location_names[loc_id])

ax9.barh(city_names, city_maes, color='coral', edgecolor='black')
ax9.set_xlabel('Test MAE (MW)')
ax9.set_title('Model B: Test MAE by Norwegian City')
ax9.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_two_group_evaluation.png', dpi=300, bbox_inches='tight')
print("   ✓ Visualizations saved as: model_two_group_evaluation.png")

plt.close()

# ============================================================================
# 14. SAVE METADATA
# ============================================================================

print("\n14. Saving model metadata...")

with open('model_two_group_info.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TWO-GROUP MODEL INFORMATION\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("MODEL A: HELSINGFORS\n")
    f.write("-" * 80 + "\n")
    f.write(f"File: model_helsingfors.txt\n")
    f.write(f"Features: {', '.join(feature_cols)}\n")
    f.write(f"Best iteration: {model_hel.best_iteration}\n")
    f.write(f"Test MAE: {test_metrics_hel['mae']:.4f} MW\n")
    f.write(f"Test RMSE: {test_metrics_hel['rmse']:.4f} MW\n\n")
    
    f.write("MODEL B: NORWAY (5 cities)\n")
    f.write("-" * 80 + "\n")
    f.write(f"File: model_norway.txt\n")
    f.write(f"Features: {', '.join(feature_cols_norway)}\n")
    f.write(f"Best iteration: {model_nor.best_iteration}\n")
    f.write(f"Test MAE: {test_metrics_nor['mae']:.4f} MW\n")
    f.write(f"Test RMSE: {test_metrics_nor['rmse']:.4f} MW\n\n")
    
    f.write("COMBINED PERFORMANCE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Weighted Test MAE: {combined_mae:.4f} MW\n")
    f.write(f"Weighted Test RMSE: {combined_rmse:.4f} MW\n")

print("   ✓ Model info saved as: model_two_group_info.txt")

# ============================================================================
# 15. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 2 TRAINING COMPLETE - SUMMARY")
print("=" * 80)

print("\nMODEL A: HELSINGFORS")
print(f"   Training time: {training_time_hel:.2f} seconds")
print(f"   Test MAE:      {test_metrics_hel['mae']:.4f} MW")
print(f"   Test RMSE:     {test_metrics_hel['rmse']:.4f} MW")

print("\nMODEL B: NORWAY (5 cities)")
print(f"   Training time: {training_time_nor:.2f} seconds")
print(f"   Test MAE:      {test_metrics_nor['mae']:.4f} MW")
print(f"   Test RMSE:     {test_metrics_nor['rmse']:.4f} MW")

print("\nCOMBINED PERFORMANCE:")
print(f"   Weighted Test MAE:  {combined_mae:.4f} MW")
print(f"   Weighted Test RMSE: {combined_rmse:.4f} MW")

print("\nFiles Created:")
print("   - model_helsingfors.txt (Helsingfors model)")
print("   - model_norway.txt (Norway model)")
print("   - model_two_group_evaluation.png (Visualizations)")
print("   - model_two_group_info.txt (Model metadata)")

print("\n" + "=" * 80)
print("To use these models for prediction:")
print("   # For Helsingfors (location_id = 0)")
print("   model_hel = lgb.Booster(model_file='model_helsingfors.txt')")
print("   # For Norway (location_id = 1-5)")
print("   model_nor = lgb.Booster(model_file='model_norway.txt')")
print("=" * 80)
