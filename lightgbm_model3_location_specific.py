"""
LightGBM Model 3: Location-Specific Models (6 Individual Models)
=================================================================

This script trains 6 separate LightGBM models, one for each location:
- Model 0: Helsingfors (Finland)
- Model 1: Oslo (Norway)
- Model 2: Stavanger (Norway)
- Model 3: Trondheim (Norway)
- Model 4: Tromsø (Norway)
- Model 5: Bergen (Norway)

Each model is optimized for its specific location's consumption patterns.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from helpers.data_retrieval import load_preprocessed_data
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LIGHTGBM MODEL 3: LOCATION-SPECIFIC MODELS (6 Individual Models)")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n1. Loading preprocessed data...")
df = load_preprocessed_data()

print(f"   Dataset shape: {df.shape}")
print(f"   Locations: {sorted(df['location_id'].unique())}")

# Location mapping
location_names = {
    0: 'Helsingfors',
    1: 'Oslo',
    2: 'Stavanger',
    3: 'Trondheim',
    4: 'Tromsø',
    5: 'Bergen'
}

# ============================================================================
# 2. SET HYPERPARAMETERS
# ============================================================================

print("\n2. Configuring hyperparameters...")

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

print("   Using same base parameters for all location models")

# ============================================================================
# 3. TRAIN MODELS FOR EACH LOCATION
# ============================================================================

# Features (excluding location_id since each model is location-specific)
feature_cols = ['month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'temperature']

# Storage for models and results
models = {}
test_metrics = {}
predictions = {}
training_times = {}

print("\n3. Training individual models for each location...")
print("=" * 80)

for loc_id in sorted(df['location_id'].unique()):
    location_name = location_names[loc_id]
    
    print(f"\n{'='*80}")
    print(f"Training Model {loc_id}: {location_name.upper()}")
    print(f"{'='*80}")
    
    # --- Filter data for this location ---
    df_loc = df[df['location_id'] == loc_id].copy()
    print(f"\n   Total samples: {len(df_loc):,}")
    
    # --- Prepare features and target ---
    X_loc = df_loc[feature_cols].copy()
    y_loc = df_loc['consumption'].copy()
    
    # --- Train-test split (chronological) ---
    n = len(df_loc)
    train_idx = int(0.70 * n)
    val_idx = int(0.85 * n)
    
    X_train = X_loc.iloc[:train_idx]
    y_train = y_loc.iloc[:train_idx]
    X_val = X_loc.iloc[train_idx:val_idx]
    y_val = y_loc.iloc[train_idx:val_idx]
    X_test = X_loc.iloc[val_idx:]
    y_test = y_loc.iloc[val_idx:]
    
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # --- Create LightGBM datasets ---
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)
    
    # --- Train model ---
    print(f"   Training...")
    start_time = datetime.now()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    training_times[loc_id] = training_time
    
    print(f"\n   ✓ Training completed in {training_time:.2f} seconds")
    print(f"   Best iteration: {model.best_iteration}")
    print(f"   Best validation MAE: {model.best_score['valid']['l1']:.4f} MW")
    
    # --- Make predictions ---
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # --- Evaluate ---
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / np.maximum(y_test, 0.1))) * 100
    
    print(f"\n   Test Set Performance:")
    print(f"      MAE:  {test_mae:.4f} MW")
    print(f"      RMSE: {test_rmse:.4f} MW")
    print(f"      R²:   {test_r2:.4f}")
    print(f"      MAPE: {test_mape:.2f}%")
    
    # --- Store results ---
    models[loc_id] = model
    test_metrics[loc_id] = {
        'mae': test_mae,
        'rmse': test_rmse,
        'r2': test_r2,
        'mape': test_mape,
        'n_test': len(y_test)
    }
    predictions[loc_id] = {
        'y_test': y_test,
        'y_pred': y_test_pred,
        'y_train': y_train,
        'y_train_pred': y_train_pred
    }
    
    # --- Save model ---
    model_filename = f'model_location_{loc_id}_{location_name.lower()}.txt'
    model.save_model(model_filename)
    print(f"   ✓ Model saved as: {model_filename}")

# ============================================================================
# 4. AGGREGATE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("4. AGGREGATE RESULTS ACROSS ALL LOCATIONS")
print("=" * 80)

# Calculate weighted average metrics
total_test_samples = sum(m['n_test'] for m in test_metrics.values())
weighted_mae = sum(m['mae'] * m['n_test'] for m in test_metrics.values()) / total_test_samples
weighted_rmse = np.sqrt(sum(m['rmse']**2 * m['n_test'] for m in test_metrics.values()) / total_test_samples)

print(f"\nWeighted Average Test Performance:")
print(f"   MAE:  {weighted_mae:.4f} MW")
print(f"   RMSE: {weighted_rmse:.4f} MW")

print(f"\nPerformance Summary by Location:")
print(f"{'Location':<15} {'Model ID':<10} {'Test MAE':<12} {'Test RMSE':<12} {'R²':<10} {'Samples':<10}")
print("-" * 80)
for loc_id in sorted(test_metrics.keys()):
    m = test_metrics[loc_id]
    print(f"{location_names[loc_id]:<15} {loc_id:<10} {m['mae']:<12.4f} {m['rmse']:<12.4f} {m['r2']:<10.4f} {m['n_test']:<10}")

# ============================================================================
# 5. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n5. Feature Importance Analysis")
print("=" * 80)

# Collect feature importance across all models
importance_data = []
for loc_id, model in models.items():
    importance = model.feature_importance(importance_type='gain')
    for i, feat in enumerate(feature_cols):
        importance_data.append({
            'location_id': loc_id,
            'location': location_names[loc_id],
            'feature': feat,
            'importance': importance[i]
        })

importance_df = pd.DataFrame(importance_data)

print("\nAverage Feature Importance Across All Models:")
avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
for feat, imp in avg_importance.items():
    print(f"   {feat:<20s}: {imp:>10.0f}")

print("\nFeature Importance by Location:")
for loc_id in sorted(models.keys()):
    print(f"\n   {location_names[loc_id]} (Model {loc_id}):")
    loc_importance = importance_df[importance_df['location_id'] == loc_id].sort_values('importance', ascending=False)
    for _, row in loc_importance.iterrows():
        print(f"      {row['feature']:<20s}: {row['importance']:>10.0f}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

print("\n6. Creating visualizations...")

# --- Figure 1: Performance Comparison ---
fig1 = plt.figure(figsize=(16, 10))

# 6.1 Test MAE by Location
ax1 = plt.subplot(2, 3, 1)
locations = [location_names[i] for i in sorted(test_metrics.keys())]
maes = [test_metrics[i]['mae'] for i in sorted(test_metrics.keys())]
colors = plt.cm.viridis(np.linspace(0, 1, len(locations)))
bars = ax1.barh(locations, maes, color=colors, edgecolor='black')
ax1.set_xlabel('Test MAE (MW)')
ax1.set_title('Test Set MAE by Location')
ax1.grid(True, alpha=0.3, axis='x')
for i, (loc, mae) in enumerate(zip(locations, maes)):
    ax1.text(mae, i, f' {mae:.3f}', va='center', fontsize=9)

# 6.2 Test RMSE by Location
ax2 = plt.subplot(2, 3, 2)
rmses = [test_metrics[i]['rmse'] for i in sorted(test_metrics.keys())]
bars = ax2.barh(locations, rmses, color=colors, edgecolor='black')
ax2.set_xlabel('Test RMSE (MW)')
ax2.set_title('Test Set RMSE by Location')
ax2.grid(True, alpha=0.3, axis='x')
for i, (loc, rmse) in enumerate(zip(locations, rmses)):
    ax2.text(rmse, i, f' {rmse:.3f}', va='center', fontsize=9)

# 6.3 R² by Location
ax3 = plt.subplot(2, 3, 3)
r2s = [test_metrics[i]['r2'] for i in sorted(test_metrics.keys())]
bars = ax3.barh(locations, r2s, color=colors, edgecolor='black')
ax3.set_xlabel('R² Score')
ax3.set_title('Test Set R² by Location')
ax3.grid(True, alpha=0.3, axis='x')
ax3.set_xlim([0, 1])
for i, (loc, r2) in enumerate(zip(locations, r2s)):
    ax3.text(r2, i, f' {r2:.3f}', va='center', fontsize=9)

# 6.4 Feature Importance Heatmap
ax4 = plt.subplot(2, 3, 4)
importance_pivot = importance_df.pivot(index='location', columns='feature', values='importance')
importance_pivot = importance_pivot.reindex([location_names[i] for i in sorted(test_metrics.keys())])
sns.heatmap(importance_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Importance'})
ax4.set_title('Feature Importance Heatmap by Location')
ax4.set_xlabel('Feature')
ax4.set_ylabel('Location')

# 6.5 Average Feature Importance
ax5 = plt.subplot(2, 3, 5)
avg_importance_sorted = avg_importance.sort_values(ascending=True)
ax5.barh(avg_importance_sorted.index, avg_importance_sorted.values, color='steelblue', edgecolor='black')
ax5.set_xlabel('Average Importance (Gain)')
ax5.set_title('Average Feature Importance Across All Models')
ax5.grid(True, alpha=0.3, axis='x')

# 6.6 Training Time by Location
ax6 = plt.subplot(2, 3, 6)
train_times = [training_times[i] for i in sorted(training_times.keys())]
bars = ax6.bar(locations, train_times, color=colors, edgecolor='black')
ax6.set_ylabel('Training Time (seconds)')
ax6.set_title('Training Time by Location')
ax6.grid(True, alpha=0.3, axis='y')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
for i, (loc, time) in enumerate(zip(locations, train_times)):
    ax6.text(i, time, f'{time:.1f}s', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_location_specific_performance.png', dpi=300, bbox_inches='tight')
print("   ✓ Performance visualization saved as: model_location_specific_performance.png")

# --- Figure 2: Predictions Comparison ---
fig2 = plt.figure(figsize=(18, 12))

for idx, loc_id in enumerate(sorted(predictions.keys())):
    # Actual vs Predicted scatter
    ax = plt.subplot(2, 3, idx + 1)
    
    y_test = predictions[loc_id]['y_test']
    y_pred = predictions[loc_id]['y_pred']
    
    ax.scatter(y_test, y_pred, alpha=0.5, s=15, color=colors[idx])
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Consumption (MW)')
    ax.set_ylabel('Predicted Consumption (MW)')
    ax.set_title(f'{location_names[loc_id]} (MAE: {test_metrics[loc_id]["mae"]:.3f} MW)')
    ax.grid(True, alpha=0.3)
    
    # Add R² annotation
    r2 = test_metrics[loc_id]['r2']
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('model_location_specific_predictions.png', dpi=300, bbox_inches='tight')
print("   ✓ Predictions visualization saved as: model_location_specific_predictions.png")

# --- Figure 3: Residuals Analysis ---
fig3 = plt.figure(figsize=(18, 12))

for idx, loc_id in enumerate(sorted(predictions.keys())):
    ax = plt.subplot(2, 3, idx + 1)
    
    y_test = predictions[loc_id]['y_test']
    y_pred = predictions[loc_id]['y_pred']
    residuals = y_test.values - y_pred
    
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color=colors[idx])
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residuals (MW)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{location_names[loc_id]} Residuals (μ={residuals.mean():.3f}, σ={residuals.std():.3f})')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_location_specific_residuals.png', dpi=300, bbox_inches='tight')
print("   ✓ Residuals visualization saved as: model_location_specific_residuals.png")

plt.close('all')

# ============================================================================
# 7. SAVE SUMMARY REPORT
# ============================================================================

print("\n7. Saving summary report...")

with open('model_location_specific_summary.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("LOCATION-SPECIFIC MODELS - SUMMARY REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Total Models Trained: {len(models)}\n")
    f.write(f"Total Test Samples: {total_test_samples:,}\n")
    f.write(f"Weighted Average Test MAE: {weighted_mae:.4f} MW\n")
    f.write(f"Weighted Average Test RMSE: {weighted_rmse:.4f} MW\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("INDIVIDUAL MODEL PERFORMANCE\n")
    f.write("=" * 80 + "\n\n")
    
    for loc_id in sorted(models.keys()):
        model = models[loc_id]
        metrics = test_metrics[loc_id]
        
        f.write(f"MODEL {loc_id}: {location_names[loc_id].upper()}\n")
        f.write("-" * 80 + "\n")
        f.write(f"File: model_location_{loc_id}_{location_names[loc_id].lower()}.txt\n")
        f.write(f"Features: {', '.join(feature_cols)}\n")
        f.write(f"Best Iteration: {model.best_iteration}\n")
        f.write(f"Training Time: {training_times[loc_id]:.2f} seconds\n")
        f.write(f"Test Samples: {metrics['n_test']:,}\n")
        f.write(f"Test MAE: {metrics['mae']:.4f} MW\n")
        f.write(f"Test RMSE: {metrics['rmse']:.4f} MW\n")
        f.write(f"Test R²: {metrics['r2']:.4f}\n")
        f.write(f"Test MAPE: {metrics['mape']:.2f}%\n")
        
        # Top features
        loc_importance = importance_df[importance_df['location_id'] == loc_id].sort_values('importance', ascending=False)
        f.write(f"\nTop 3 Features:\n")
        for _, row in loc_importance.head(3).iterrows():
            f.write(f"   {row['feature']}: {row['importance']:.0f}\n")
        f.write("\n")
    
    f.write("=" * 80 + "\n")
    f.write("USAGE INSTRUCTIONS\n")
    f.write("=" * 80 + "\n\n")
    f.write("To load models for prediction:\n\n")
    f.write("import lightgbm as lgb\n\n")
    for loc_id in sorted(models.keys()):
        f.write(f"# {location_names[loc_id]} (location_id={loc_id})\n")
        f.write(f"model_{loc_id} = lgb.Booster(model_file='model_location_{loc_id}_{location_names[loc_id].lower()}.txt')\n")
    f.write("\n")

print("   ✓ Summary report saved as: model_location_specific_summary.txt")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 3 TRAINING COMPLETE - FINAL SUMMARY")
print("=" * 80)

print(f"\n✓ Successfully trained {len(models)} location-specific models")
print(f"\nOverall Performance:")
print(f"   Weighted Test MAE:  {weighted_mae:.4f} MW")
print(f"   Weighted Test RMSE: {weighted_rmse:.4f} MW")

print(f"\nBest Performing Location:")
best_loc = min(test_metrics.items(), key=lambda x: x[1]['mae'])
print(f"   {location_names[best_loc[0]]}: MAE = {best_loc[1]['mae']:.4f} MW")

print(f"\nMost Challenging Location:")
worst_loc = max(test_metrics.items(), key=lambda x: x[1]['mae'])
print(f"   {location_names[worst_loc[0]]}: MAE = {worst_loc[1]['mae']:.4f} MW")

print(f"\nTotal Training Time: {sum(training_times.values()):.2f} seconds")

print(f"\nFiles Created:")
print(f"   - 6 model files (model_location_X_[name].txt)")
print(f"   - model_location_specific_performance.png")
print(f"   - model_location_specific_predictions.png")
print(f"   - model_location_specific_residuals.png")
print(f"   - model_location_specific_summary.txt")

print("\n" + "=" * 80)
print("Models are ready for deployment!")
print("=" * 80)
