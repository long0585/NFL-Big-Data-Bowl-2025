import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, 
    roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import load_data
from transformer_model import TransformerModel

def build_pre_snapshot(input_df: pd.DataFrame, frames_before: int) -> pd.DataFrame:
    """Build snapshot of tracking data two frames before the throw"""
    throw_frames = (
        input_df
        .groupby(['unique_play_id'])["frame_id"]
        .max()
        .reset_index(name="frame_id_throw")
    )

    pre_throw_frames = throw_frames.copy()
    pre_throw_frames["frame_id_pre"] = np.maximum(
        pre_throw_frames["frame_id_throw"] - frames_before,
        1,
    )

    snapshot_pre_throw = input_df.merge(
        pre_throw_frames[['unique_play_id', "frame_id_pre"]],
        left_on=['unique_play_id', "frame_id"],
        right_on=['unique_play_id', "frame_id_pre"],
        how="inner",
    )

    snapshot_pre_throw = snapshot_pre_throw.drop(columns=["frame_id_pre"])
    return snapshot_pre_throw

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find the threshold that maximizes F1 score.
    
    Returns:
        optimal_threshold: float
        optimal_f1: float
        precision_at_optimal: float
        recall_at_optimal: float
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 for each threshold
    # Note: precision_recall_curve returns n+1 precision/recall values but n thresholds
    # So we trim the last precision/recall value
    precision = precision[:-1]
    recall = recall[:-1]
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find best threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    
    return optimal_threshold, optimal_f1, optimal_precision, optimal_recall

# Load data
print("="*70)
print("LOADING DATA")
print("="*70)

all_input = []
all_output = []
for week in range(1, 19):
    print(f"Loading week {week}...")
    week_input, week_output = load_data.load_data(week)
    all_input.append(week_input)
    all_output.append(week_output)

input_df = pd.concat(all_input)
output_df = pd.concat(all_output)

# Preprocess
input_df = build_pre_snapshot(input_df, 2)
output_df = build_pre_snapshot(output_df, 0)

# Filters
input_df = input_df[input_df['team_coverage_type'] != 'COVER_0_MAN']
output_df = output_df[output_df['team_coverage_type'] != 'COVER_0_MAN']
input_df = input_df[input_df['team_coverage_type'] != 'COVER_6_ZONE']
output_df = output_df[output_df['team_coverage_type'] != 'COVER_6_ZONE']
input_df = input_df[input_df['helper'] == 1]
output_df = output_df[output_df['helper'] == 1]
input_df = input_df[input_df['pass_length'] >= 15]
output_df = output_df[output_df['pass_length'] >= 15]

input_df = input_df.sort_values(by='unique_play_id')
output_df = output_df.sort_values(by='unique_play_id')

# Check alignment
X_check = input_df[['unique_play_id', 'nfl_id']].reset_index(drop=True)
y_check = output_df[['unique_play_id', 'nfl_id']].reset_index(drop=True)
if not (X_check == y_check).all().all():
    print("[WARNING]: Input and output not aligned!")
else:
    print("[INFO]: Data aligned ✓")

# Define target
output_df['distance_to_ball'] = (output_df['ball_vec_x_clean']**2 + output_df['ball_vec_y_clean']**2)**0.5
output_df['gets_to_ball_in_time'] = (output_df['distance_to_ball'] <= 4)

# Features
features = ['ball_vec_x_clean', 'ball_vec_y_clean', 'v_x', 'v_y', 'num_frames_output', 'a_clean']
X = input_df[features].values
y = output_df['gets_to_ball_in_time'].values

print(f"\nDataset: {len(X)} samples")
print(f"Class distribution: {np.bincount(y)}")

# Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=14)
fold_results = []
all_thresholds = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*70}")
    print(f"FOLD {fold + 1}/10")
    print(f"{'='*70}")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Class weights
    class_counts = np.bincount(y_train)
    class_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32)
    print(f"Class distribution - 0: {class_counts[0]}, 1: {class_counts[1]}")
    print(f"Class weight: {class_weight.item():.4f}")
    
    # Model
    input_dim = X_train_scaled.shape[1]
    model = TransformerModel(input_dim, 1, 2, 2, 128)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train_scaled, dtype=torch.float32))
        loss = criterion(outputs.squeeze(), torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_scaled, dtype=torch.float32)).squeeze()
        y_pred_proba = torch.sigmoid(logits).numpy()
    
    # Find optimal threshold for THIS fold
    optimal_threshold, optimal_f1, optimal_prec, optimal_rec = find_optimal_threshold(
        y_test, y_pred_proba
    )
    all_thresholds.append(optimal_threshold)
    
    print(f"\n--- THRESHOLD OPTIMIZATION ---")
    print(f"Optimal threshold: {optimal_threshold:.4f} (default is 0.5)")
    print(f"F1 at optimal:     {optimal_f1:.4f}")
    print(f"Precision at opt:  {optimal_prec:.4f}")
    print(f"Recall at opt:     {optimal_rec:.4f}")
    
    # Evaluate with DEFAULT threshold (0.5)
    y_pred_default = (y_pred_proba > 0.5).astype(int)
    acc_default = accuracy_score(y_test, y_pred_default)
    prec_default = precision_score(y_test, y_pred_default, zero_division=0)
    rec_default = recall_score(y_test, y_pred_default, zero_division=0)
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    
    print(f"\n--- WITH DEFAULT THRESHOLD (0.5) ---")
    print(f"Accuracy:  {acc_default:.4f}")
    print(f"Precision: {prec_default:.4f}")
    print(f"Recall:    {rec_default:.4f}")
    print(f"F1:        {f1_default:.4f}")
    
    # Evaluate with OPTIMAL threshold
    y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
    acc_optimal = accuracy_score(y_test, y_pred_optimal)
    prec_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
    rec_optimal = recall_score(y_test, y_pred_optimal, zero_division=0)
    f1_optimal = f1_score(y_test, y_pred_optimal, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n--- WITH OPTIMAL THRESHOLD ({optimal_threshold:.4f}) ---")
    print(f"Accuracy:  {acc_optimal:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {prec_optimal:.4f}")
    print(f"Recall:    {rec_optimal:.4f}")
    print(f"F1:        {f1_optimal:.4f}")
    
    improvement = f1_optimal - f1_default
    print(f"\n✓ F1 improvement: {improvement:+.4f} ({improvement/f1_default*100:+.1f}%)" if f1_default > 0 else "")
    
    # Store results (using optimal threshold)
    fold_results.append({
        'fold': fold + 1,
        'threshold': optimal_threshold,
        'accuracy': acc_optimal,
        'auc': auc,
        'precision': prec_optimal,
        'recall': rec_optimal,
        'f1': f1_optimal,
        'f1_default': f1_default,
        'improvement': improvement
    })
    
    # Save model and scaler
    torch.save(model, f"./../models/safety_model_fold_{fold+1}.pth")
    joblib.dump(scaler, f"./../models/scaler_fold_{fold+1}.pkl")

# Summary
results_df = pd.DataFrame(fold_results)
print(f"\n{'='*70}")
print("CROSS-VALIDATION SUMMARY (WITH OPTIMAL THRESHOLDS)")
print(f"{'='*70}")
print(results_df[['fold', 'threshold', 'accuracy', 'auc', 'precision', 'recall', 'f1']].to_string(index=False))

print(f"\n{'='*70}")
print("MEAN ± STD")
print(f"{'='*70}")
for metric in ['threshold', 'accuracy', 'auc', 'precision', 'recall', 'f1']:
    mean_val = results_df[metric].mean()
    std_val = results_df[metric].std()
    print(f"{metric.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")

# Average improvement from using optimal threshold
avg_improvement = results_df['improvement'].mean()
print(f"\nAverage F1 improvement from optimal threshold: {avg_improvement:+.4f}")

# Best fold
best_fold = results_df['f1'].idxmax() + 1
best_threshold = results_df.loc[best_fold - 1, 'threshold']
print(f"\n{'='*70}")
print(f"BEST MODEL: Fold {best_fold}")
print(f"  F1 Score: {results_df.loc[best_fold-1, 'f1']:.4f}")
print(f"  Threshold: {best_threshold:.4f}")
print(f"{'='*70}")

# Save best model, scaler, AND threshold
shutil.copy(f"./../models/safety_model_fold_{best_fold}.pth", "safety_model_best.pth")
shutil.copy(f"./../models/scaler_fold_{best_fold}.pkl", "scaler_best.pkl")

# Save threshold to a file
threshold_info = {
    'best_threshold': best_threshold,
    'mean_threshold': results_df['threshold'].mean(),
    'best_fold': best_fold,
    'best_f1': results_df.loc[best_fold-1, 'f1']
}
joblib.dump(threshold_info, "threshold_best.pkl")

print(f"\nSaved files:")
print(f"  - safety_model_best.pth")
print(f"  - scaler_best.pkl")
print(f"  - threshold_best.pkl (optimal threshold: {best_threshold:.4f})")

# Optional: Plot threshold distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(results_df['threshold'], bins=10, edgecolor='black')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.3f}')
plt.axvline(0.5, color='blue', linestyle='--', label='Default: 0.500')
plt.xlabel('Optimal Threshold')
plt.ylabel('Frequency')
plt.title('Distribution of Optimal Thresholds Across Folds')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(results_df['threshold'], results_df['f1'], s=100, alpha=0.6)
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.axvline(0.5, color='blue', linestyle='--', alpha=0.3, label='Default')
plt.legend()

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150)
print(f"  - threshold_analysis.png")
print("\nTraining complete!")