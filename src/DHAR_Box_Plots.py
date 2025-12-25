# ============================================================
# TEAM-LEVEL DHAR ANALYSIS
# Structural Difficulty vs Execution
# ============================================================

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

import load_data
from transformer_model import TransformerModel


# ============================================================
# SNAPSHOT HELPER (MUST MATCH TRAINING)
# ============================================================
def build_pre_snapshot(df: pd.DataFrame, frames_before: int) -> pd.DataFrame:
    throw_frames = (
        df.groupby("unique_play_id")["frame_id"]
        .max()
        .reset_index(name="frame_id_throw")
    )

    throw_frames["frame_id_pre"] = np.maximum(
        throw_frames["frame_id_throw"] - frames_before, 1
    )

    snap = df.merge(
        throw_frames[["unique_play_id", "frame_id_pre"]],
        left_on=["unique_play_id", "frame_id"],
        right_on=["unique_play_id", "frame_id_pre"],
        how="inner",
    )

    return snap.drop(columns=["frame_id_pre"])


# ============================================================
# MAIN
# ============================================================
def main():

    print("=" * 90)
    print("TEAM-LEVEL DEEP HELP ANALYSIS (STRUCTURE vs EXECUTION)")
    print("=" * 90)

    # --------------------------------------------------------
    # Load model + scaler
    # --------------------------------------------------------
    model = torch.load("safety_model_best.pth", weights_only=False)
    model.eval()
    scaler = joblib.load("scaler_best.pkl")

    # --------------------------------------------------------
    # Load data (Weeks 1–18)
    # --------------------------------------------------------
    all_input, all_output = [], []

    for week in range(1, 19):
        inp, out = load_data.load_data(week)
        all_input.append(inp)
        all_output.append(out)

    input_df = pd.concat(all_input, ignore_index=True)
    output_df = pd.concat(all_output, ignore_index=True)

    # --------------------------------------------------------
    # Build snapshots
    # --------------------------------------------------------
    input_df = build_pre_snapshot(input_df, frames_before=2)
    output_df = build_pre_snapshot(output_df, frames_before=0)

    # --------------------------------------------------------
    # Filters (MATCH TRAINING)
    # --------------------------------------------------------
    mask = (
        (input_df["team_coverage_type"] != "COVER_0_MAN") &
        (input_df["team_coverage_type"] != "COVER_6_ZONE") &
        (input_df["helper"] == 1) &
        (input_df["pass_length"] >= 15)
    )

    input_df = input_df[mask].copy()

    # --------------------------------------------------------
    # Align output USING unique_play_id (CRITICAL FIX)
    # --------------------------------------------------------
    valid_plays = input_df["unique_play_id"].unique()
    output_df = output_df[
        output_df["unique_play_id"].isin(valid_plays)
    ].copy()

    # --------------------------------------------------------
    # Defensive team (play-level)
    # --------------------------------------------------------
    play_def_team = (
        input_df.groupby("unique_play_id")["defensive_team"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index(name="play_defensive_team")
    )

    input_df = input_df.merge(
        play_def_team,
        on="unique_play_id",
        how="left"
    )

    # --------------------------------------------------------
    # Catch time (seconds)
    # --------------------------------------------------------
    FRAME_TIME = 0.1

    play_throw_frame = (
        input_df.groupby("unique_play_id")["frame_id"]
        .max()
        .reset_index(name="throw_frame")
    )

    play_catch_frame = (
        output_df.groupby("unique_play_id")["frame_id"]
        .max()
        .reset_index(name="catch_frame")
    )

    timing = play_throw_frame.merge(
        play_catch_frame,
        on="unique_play_id",
        how="inner"
    )

    timing["catch_time_sec"] = (
        (timing["catch_frame"] - timing["throw_frame"]) * FRAME_TIME
    )

    input_df = input_df.merge(
        timing[["unique_play_id", "catch_time_sec"]],
        on="unique_play_id",
        how="left"
    )

    input_df = input_df[input_df["catch_time_sec"] > 0].copy()

    # --------------------------------------------------------
    # Distance, speed, feasibility
    # --------------------------------------------------------
    input_df["distance_at_throw"] = np.sqrt(
        input_df["ball_vec_x_clean"]**2 +
        input_df["ball_vec_y_clean"]**2
    )

    input_df["defender_speed"] = np.sqrt(
        input_df["v_x"]**2 +
        input_df["v_y"]**2
    )

    input_df = input_df[input_df["defender_speed"] > 0.5].copy()

    input_df["time_to_ball"] = (
        input_df["distance_at_throw"] /
        input_df["defender_speed"]
    )

    input_df["time_ratio"] = (
        input_df["time_to_ball"] /
        input_df["catch_time_sec"]
    )

    # --------------------------------------------------------
    # Ground truth (PLAY-LEVEL, NOT ROW-LEVEL)
    # --------------------------------------------------------
    output_df["distance_to_ball"] = np.sqrt(
        output_df["ball_vec_x_clean"]**2 +
        output_df["ball_vec_y_clean"]**2
    )

    play_outcomes = (
        output_df.groupby("unique_play_id")["distance_to_ball"]
        .min()
        .reset_index(name="min_distance_to_ball")
    )

    input_df = input_df.merge(
        play_outcomes,
        on="unique_play_id",
        how="left"
    )

    input_df["actual"] = (
        input_df["min_distance_to_ball"] <= 4
    ).astype(int)

    # --------------------------------------------------------
    # Predictions (DHAR)
    # --------------------------------------------------------
    features = [
        "ball_vec_x_clean", "ball_vec_y_clean",
        "v_x", "v_y",
        "num_frames_output", "a_clean"
    ]

    X = scaler.transform(input_df[features].values)

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze()

    input_df["pred_prob"] = torch.sigmoid(logits).numpy()

    # --------------------------------------------------------
    # TEAM-LEVEL AGGREGATION
    # --------------------------------------------------------
    team_stats = (
        input_df.groupby("play_defensive_team")
        .agg(
            mean_dhar=("pred_prob", "mean"),
            arrival_rate=("actual", "mean"),
            mean_distance=("distance_at_throw", "mean"),
            mean_time_ratio=("time_ratio", "mean"),
            n_plays=("unique_play_id", "nunique")
        )
        .reset_index()
        .rename(columns={"play_defensive_team": "defensive_team"})
        .sort_values("mean_dhar", ascending=False)
    )

    print(team_stats.head(10))

    return input_df, team_stats


# ============================================================
if __name__ == "__main__":
    input_df, team_stats = main()
