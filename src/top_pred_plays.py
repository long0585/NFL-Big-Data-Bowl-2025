import numpy as np
import torch
import pandas as pd
import joblib
import load_data
from transformer_model import TransformerModel

# ============================================================
# Snapshot helper (MUST match training)
# ============================================================
def build_pre_snapshot(input_df: pd.DataFrame, frames_before: int) -> pd.DataFrame:
    throw_frames = (
        input_df
        .groupby("unique_play_id")["frame_id"]
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


# ============================================================
# Main
# ============================================================
def main():

    print("[INFO]: Loading model, scaler, and threshold")

    # --------------------------------------------------------
    # Load model + scaler + threshold
    # --------------------------------------------------------
    model = torch.load("./safety_model_best.pth", weights_only=False)
    model.eval()

    scaler = joblib.load("./scaler_best.pkl")

    try:
        threshold_info = joblib.load("./threshold_best.pkl")
        threshold = threshold_info["best_threshold"]
        print(f"[INFO]: Using optimal threshold = {threshold:.4f}")
    except FileNotFoundError:
        threshold = 0.5
        print("[WARNING]: threshold_best.pkl not found, using 0.5")

    # --------------------------------------------------------
    # Load ALL DATA — Weeks 1–18 ONLY
    # --------------------------------------------------------
    all_input, all_output = [], []

    for week in range(1, 19):
        print(f"[INFO]: Loading week {week}")
        inp, out = load_data.load_data(week)
        all_input.append(inp)
        all_output.append(out)

    input_df = pd.concat(all_input, ignore_index=True)
    output_df = pd.concat(all_output, ignore_index=True)

    # --------------------------------------------------------
    # Snapshot timing (MATCH TRAINING)
    # --------------------------------------------------------
    input_df = build_pre_snapshot(input_df, frames_before=2)
    output_df = build_pre_snapshot(output_df, frames_before=0)

    # --------------------------------------------------------
    # Filters (MUST match training)
    # --------------------------------------------------------
    mask = (
        (input_df["team_coverage_type"] != "COVER_0_MAN") &
        (input_df["team_coverage_type"] != "COVER_6_ZONE") &
        (input_df["helper"] == 1) &
        (input_df["pass_length"] >= 15)
    )

    input_df = input_df[mask].copy()

    # --------------------------------------------------------
    # REALIGN INPUT / OUTPUT (KEY-BASED)
    # --------------------------------------------------------
    output_df = output_df.merge(
        input_df[["unique_play_id", "nfl_id"]],
        on=["unique_play_id", "nfl_id"],
        how="inner"
    )

    input_df = input_df.merge(
        output_df[["unique_play_id", "nfl_id"]],
        on=["unique_play_id", "nfl_id"],
        how="inner"
    )

    input_df = input_df.sort_values("unique_play_id").reset_index(drop=True)
    output_df = output_df.sort_values("unique_play_id").reset_index(drop=True)

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------
    output_df["distance_to_ball"] = np.sqrt(
        output_df["ball_vec_x_clean"]**2 +
        output_df["ball_vec_y_clean"]**2
    )

    output_df["actual"] = (output_df["distance_to_ball"] <= 4).astype(int)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------
    features = [
        "ball_vec_x_clean",
        "ball_vec_y_clean",
        "v_x",
        "v_y",
        "num_frames_output",
        "a_clean",
    ]

    X = scaler.transform(input_df[features].values)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze()
        prob = torch.sigmoid(logits).numpy()

    input_df["pred_prob"] = prob
    input_df["predicted"] = (prob > threshold).astype(int)
    input_df["actual"] = output_df["actual"].values

    # --------------------------------------------------------
    # KEEP: ACTUAL = 0 AND PREDICTED = 0
    # --------------------------------------------------------
    # KEEP: ACTUAL = 0 AND PREDICTED = 0 AND PASS LENGTH >= 25
    df_00 = input_df[
        (input_df["actual"] == 0) &
        (input_df["predicted"] == 0) &
        (input_df["pass_length"] >= 25)
    ].copy()

    # Sort: lowest confidence first, then longest passes
    df_00 = (
        df_00
        .sort_values(
            by=["pred_prob", "pass_length"],
            ascending=[True, False]
        )
        .head(30)
    )


    # --------------------------------------------------------
    # Pretty print results
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("TOP 30 PLAYS — ACTUAL = 0, PREDICTED = 0 (LOWEST CONFIDENCE FIRST)")
    print("=" * 90)

    for i, row in df_00.iterrows():
        game_id, play_id = row["unique_play_id"].split("_")
        print(
            f"{i+1:>2}. "
            f"Game {game_id} | Play {play_id} | "
            f"Coverage: {row['team_coverage_type']:>12} | "
            f"PassLen: {row['pass_length']:>4.1f} | "
            f"Pos: {row['player_position']:>3} | "
            f"P(arrive): {row['pred_prob']:.3f}"
        )

    print("=" * 90)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
