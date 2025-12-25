import numpy as np
import torch
import pandas as pd
import joblib
import load_data
from transformer_model import TransformerModel

def build_pre_snapshot(input_df: pd.DataFrame, frames_before: int) -> pd.DataFrame:
    """Build snapshot of tracking data frames before the throw"""
    throw_frames = (
        input_df.groupby("unique_play_id")["frame_id"]
        .max()
        .reset_index(name="frame_id_throw")
    )
    
    throw_frames["frame_id_pre"] = np.maximum(
        throw_frames["frame_id_throw"] - frames_before, 1
    )
    
    snap = input_df.merge(
        throw_frames[["unique_play_id", "frame_id_pre"]],
        left_on=["unique_play_id", "frame_id"],
        right_on=["unique_play_id", "frame_id_pre"],
        how="inner",
    )
    
    return snap.drop(columns=["frame_id_pre"])


def main():
    print("="*90)
    print("LOADING MODEL AND DATA")
    print("="*90)
    
    # Load model and scaler
    try:
        model = torch.load("./safety_model_best.pth", weights_only=False)
        model.eval()
        scaler = joblib.load("./scaler_best.pkl")
        print("✓ Model and scaler loaded")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please train the model first!")
        return
    
    # Load threshold (optional)
    try:
        threshold_info = joblib.load("./threshold_best.pkl")
        optimal_threshold = threshold_info['best_threshold']
        print(f"✓ Optimal threshold loaded: {optimal_threshold:.4f}")
    except FileNotFoundError:
        optimal_threshold = 0.5
        print("⚠ Using default threshold: 0.5")
    
    # --------------------------------------------------------
    # Load all weeks of data
    # --------------------------------------------------------
    print("\nLoading all weeks...")
    all_input = []
    week_map = {}  # Map unique_play_id to week
    
    for week in range(1, 19):
        print(f"  Week {week}...", end=" ")
        inp, _ = load_data.load_data(week)
        
        # Track which week each play is from
        for play_id in inp['unique_play_id'].unique():
            week_map[play_id] = week
        
        all_input.append(inp)
        print("✓")
    
    input_df = pd.concat(all_input, ignore_index=True)
    
    # Add week column
    input_df['week'] = input_df['unique_play_id'].map(week_map)
    
    print(f"\nTotal plays loaded: {input_df['unique_play_id'].nunique()}")
    
    # --------------------------------------------------------
    # Build pre-throw snapshot
    # --------------------------------------------------------
    print("\nBuilding pre-throw snapshots...")
    input_df = build_pre_snapshot(input_df, 2)
    
    # --------------------------------------------------------
    # Filter to helper safeties on deep passes
    # --------------------------------------------------------
    print("Applying filters...")
    initial_count = len(input_df)
    
    input_df = input_df[
        (input_df["team_coverage_type"] != "COVER_0_MAN") &
        (input_df["team_coverage_type"] != "COVER_6_ZONE") &
        (input_df["helper"] == 1) &
        (input_df["pass_length"] >= 15)
    ].copy()
    
    print(f"  {initial_count} → {len(input_df)} rows after filtering")
    print(f"  {input_df['unique_play_id'].nunique()} unique plays")
    
    # --------------------------------------------------------
    # Split unique_play_id into game_id and play_id
    # --------------------------------------------------------
    ids = input_df["unique_play_id"].astype(str).str.split("_", expand=True)
    input_df["game_id"] = ids[0].astype(int)
    input_df["play_id"] = ids[1].astype(int)
    
    # --------------------------------------------------------
    # Calculate ground truth (need output data)
    # --------------------------------------------------------
    print("\nCalculating ground truth from output data...")
    all_output = []
    for week in range(1, 19):
        _, out = load_data.load_data(week)
        all_output.append(out)
    
    output_df = pd.concat(all_output, ignore_index=True)
    output_df = build_pre_snapshot(output_df, 0)
    
    # Match output to input (only keep plays that are in filtered input)
    output_df = output_df.merge(
        input_df[["unique_play_id", "nfl_id"]],
        on=["unique_play_id", "nfl_id"],
        how="inner"
    )
    
    # Calculate distance to ball
    output_df["distance_to_ball"] = np.sqrt(
        output_df["ball_vec_x_clean"]**2 +
        output_df["ball_vec_y_clean"]**2
    )
    output_df["actual"] = (output_df["distance_to_ball"] <= 4).astype(int)
    
    # Merge actual back to input
    input_df = input_df.merge(
        output_df[["unique_play_id", "nfl_id", "actual", "distance_to_ball"]],
        on=["unique_play_id", "nfl_id"],
        how="left"
    )
    
    # --------------------------------------------------------
    # Make predictions
    # --------------------------------------------------------
    print("\nGenerating predictions...")
    features = [
        "ball_vec_x_clean",
        "ball_vec_y_clean",
        "v_x",
        "v_y",
        "num_frames_output",
        "a_clean",
    ]
    
    X = scaler.transform(input_df[features].values)
    
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze()
        input_df["pred_prob"] = torch.sigmoid(logits).numpy()
    
    input_df["predicted"] = (input_df["pred_prob"] >= optimal_threshold).astype(int)
    input_df["correct"] = (input_df["predicted"] == input_df["actual"]).astype(int)
    
    # --------------------------------------------------------
    # Filter to specific players
    # --------------------------------------------------------
    TARGET_PLAYERS = [
        "Daxton Hill",
        "Harrison Smith", 
        "Jordan Whitehead",
        "Xavier McKinney"
    ]
    
    results = input_df[input_df["player_name"].isin(TARGET_PLAYERS)].copy()
    
    print(f"\n{'='*90}")
    print(f"FOUND {len(results)} PLAYS WITH TARGET PLAYERS AS HELPER DEFENDERS")
    print(f"{'='*90}")
    
    # --------------------------------------------------------
    # Display results for each player
    # --------------------------------------------------------
    for player in TARGET_PLAYERS:
        player_plays = results[results["player_name"] == player]
        
        if len(player_plays) == 0:
            print(f"\n{player}: NO PLAYS FOUND")
            continue
        
        # Summary stats
        total = len(player_plays)
        correct = player_plays["correct"].sum()
        accuracy = correct / total * 100
        
        pred_1_actual_1 = ((player_plays["predicted"] == 1) & (player_plays["actual"] == 1)).sum()
        pred_1_actual_0 = ((player_plays["predicted"] == 1) & (player_plays["actual"] == 0)).sum()
        pred_0_actual_1 = ((player_plays["predicted"] == 0) & (player_plays["actual"] == 1)).sum()
        pred_0_actual_0 = ((player_plays["predicted"] == 0) & (player_plays["actual"] == 0)).sum()
        
        print(f"\n{'='*90}")
        print(f"{player.upper()}")
        print(f"{'='*90}")
        print(f"Total plays: {total}")
        print(f"Correct predictions: {correct}/{total} ({accuracy:.1f}%)")
        print(f"\nConfusion Matrix:")
        print(f"  Pred=1, Actual=1 (True Positive):  {pred_1_actual_1}")
        print(f"  Pred=1, Actual=0 (False Positive): {pred_1_actual_0}")
        print(f"  Pred=0, Actual=1 (False Negative): {pred_0_actual_1}")
        print(f"  Pred=0, Actual=0 (True Negative):  {pred_0_actual_0}")
        
        # Show all plays
        print(f"\n{'='*90}")
        print("ALL PLAYS:")
        print(f"{'='*90}")
        
        display_cols = [
            "week",
            "game_id", 
            "play_id",
            "team_coverage_type",
            "pass_result",
            "pass_length",
            "pred_prob",
            "predicted",
            "actual",
            "correct",
            "distance_to_ball"
        ]
        
        player_display = player_plays[display_cols].copy()
        player_display = player_display.sort_values(["week", "game_id", "play_id"])
        
        # Format for better display
        player_display["pred_prob"] = player_display["pred_prob"].apply(lambda x: f"{x:.3f}")
        player_display["distance_to_ball"] = player_display["distance_to_ball"].apply(lambda x: f"{x:.1f}")
        player_display["pass_length"] = player_display["pass_length"].apply(lambda x: f"{x:.0f}")
        player_display["correct"] = player_display["correct"].apply(lambda x: "✓" if x == 1 else "✗")
        
        print(player_display.to_string(index=False))
    
    # --------------------------------------------------------
    # Save to CSV for easy reference
    # --------------------------------------------------------
    output_cols = [
        "player_name",
        "week",
        "game_id",
        "play_id",
        "team_coverage_type",
        "pass_result",
        "pass_length",
        "pred_prob",
        "predicted",
        "actual",
        "correct",
        "distance_to_ball"
    ]
    
    results[output_cols].sort_values(["player_name", "week", "game_id", "play_id"]).to_csv(
        "helper_safety_plays.csv", 
        index=False
    )
    
    print(f"\n{'='*90}")
    print("SAVED TO: helper_safety_plays.csv")
    print(f"{'='*90}")
    
    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------
    print(f"\n{'='*90}")
    print("OVERALL SUMMARY")
    print(f"{'='*90}")
    
    summary = results.groupby("player_name").agg({
        "unique_play_id": "count",
        "correct": "sum",
        "predicted": "sum",
        "actual": "sum"
    }).rename(columns={
        "unique_play_id": "total_plays",
        "correct": "correct_predictions",
        "predicted": "predicted_arrives",
        "actual": "actual_arrives"
    })
    
    summary["accuracy"] = (summary["correct_predictions"] / summary["total_plays"] * 100).round(1)
    
    print(summary.to_string())
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()