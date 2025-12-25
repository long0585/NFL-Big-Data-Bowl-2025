import numpy as np
import torch
import pandas as pd
import joblib
import sys
import load_data
from transformer_model import TransformerModel

def build_pre_snapshot(input_df: pd.DataFrame, frames_before: int) -> pd.DataFrame:
    """Build snapshot of tracking data frames before the throw"""
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

def run_manual_test(input_df, output_df):
    print("[INFO]: Running prediction model")
    
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
    
    if len(input_df) == 0 or len(output_df) == 0:
        print("[WARNING]: No valid plays after filtering")
        print("This play either:")
        print("  - Is not a deep pass (pass_length < 15 yards)")
        print("  - Has no helper safety assigned")
        print("  - Uses COVER_0_MAN or COVER_6_ZONE coverage")
        return 0, 0
    
    input_df = input_df.sort_values(by='unique_play_id')
    output_df = output_df.sort_values(by='unique_play_id')

    # Alignment check
    X_check = input_df[['unique_play_id', 'nfl_id']].reset_index(drop=True)
    y_check = output_df[['unique_play_id', 'nfl_id']].reset_index(drop=True)
    if not (X_check == y_check).all().all():
        print("[WARNING]: Input/output misalignment!")

    # Define ground truth
    output_df['distance_to_ball'] = np.sqrt(
        output_df['ball_vec_x_clean']**2 + 
        output_df['ball_vec_y_clean']**2
    )
    output_df['gets_to_ball_in_time'] = (output_df['distance_to_ball'] <= 4)
    
    # Display play info
    print(f"\n{'='*70}")
    print("PLAY INFORMATION")
    print(f"{'='*70}")
    print(f"Play ID:           {input_df['unique_play_id'].iloc[0]}")
    print(f"Coverage:          {input_df['team_coverage_type'].iloc[0]}")
    print(f"Pass Length:       {input_df['pass_length'].iloc[0]:.1f} yards")
    print(f"Pass Result:       {input_df['pass_result'].iloc[0]}")
    print(f"Helper Position:   {input_df[input_df['helper']==1]['player_position'].iloc[0]}")
    print(f"Distance to ball:  {output_df['distance_to_ball'].iloc[0]:.2f} yards")
    print(f"Ground truth:      {'ARRIVED (≤4 yds)' if output_df['gets_to_ball_in_time'].iloc[0] else 'DID NOT ARRIVE (>4 yds)'}")

    # Features (must match training)
    features = ['ball_vec_x_clean', 'ball_vec_y_clean', 'v_x', 'v_y', 'num_frames_output', 'a_clean']
    
    missing = [f for f in features if f not in input_df.columns]
    if missing:
        print(f"[ERROR]: Missing features: {missing}")
        return 0, 0
    
    X = input_df[features].values
    y = output_df['gets_to_ball_in_time'].values

    # Load model
    try:
        model = torch.load("./safety_model_best.pth", weights_only=False)
        model.eval()
        print("\n[INFO]: ✓ Model loaded")
    except FileNotFoundError:
        print("\n[ERROR]: safety_model_best.pth not found!")
        return 0, 0

    # Load scaler
    try:
        scaler = joblib.load("./scaler_best.pkl")
        print("[INFO]: ✓ Scaler loaded")
    except FileNotFoundError:
        print("\n[ERROR]: scaler_best.pkl not found!")
        return 0, 0
    
    # Load optimal threshold
    try:
        threshold_info = joblib.load("./threshold_best.pkl")
        optimal_threshold = threshold_info['best_threshold']
        print(f"[INFO]: ✓ Optimal threshold loaded: {optimal_threshold:.4f}")
        use_optimal = True
    except FileNotFoundError:
        print("[WARNING]: threshold_best.pkl not found, using default 0.5")
        optimal_threshold = 0.5
        use_optimal = False
    
    # Scale and predict
    X_test = scaler.transform(X)
    
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
        y_pred_proba = torch.sigmoid(logits).numpy()
    
    # Predictions with both thresholds
    y_pred_default = int(y_pred_proba > 0.5)
    y_pred_optimal = int(y_pred_proba > optimal_threshold)
    actual = int(y)
    
    # Display results
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"Model output (logit):      {logits.item():8.4f}")
    print(f"Predicted probability:     {y_pred_proba:8.4f} ({y_pred_proba*100:5.1f}%)")
    print(f"\n--- WITH DEFAULT THRESHOLD (0.5) ---")
    print(f"Prediction:                {y_pred_default} ({'ARRIVES' if y_pred_default else 'DOES NOT ARRIVE'})")
    print(f"Correct?                   {'✓ YES' if y_pred_default == actual else '✗ NO'}")
    
    if use_optimal:
        print(f"\n--- WITH OPTIMAL THRESHOLD ({optimal_threshold:.4f}) ---")
        print(f"Prediction:                {y_pred_optimal} ({'ARRIVES' if y_pred_optimal else 'DOES NOT ARRIVE'})")
        print(f"Correct?                   {'✓ YES' if y_pred_optimal == actual else '✗ NO'}")
        
        if y_pred_default != y_pred_optimal:
            print(f"\n⚠️  THRESHOLD MATTERS! Predictions differ:")
            print(f"   Default (0.5):          {y_pred_default}")
            print(f"   Optimal ({optimal_threshold:.3f}):      {y_pred_optimal}")
    
    print(f"\n--- GROUND TRUTH ---")
    print(f"Actual:                    {actual} ({'ARRIVED' if actual else 'DID NOT ARRIVE'})")
    print(f"{'='*70}\n")
    
    # Return optimal prediction if available, otherwise default
    predict = y_pred_optimal if use_optimal else y_pred_default
    return predict, actual