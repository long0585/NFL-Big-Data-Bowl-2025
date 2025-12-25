import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import load_data

from transformer_model import TransformerModel
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ============================================================
# TEAM → LOGO NAME MAP
# ============================================================
TEAM_LOGO_MAP = {
    "ARI": "Cardinals.png",
    "ATL": "Falcons.png",
    "BAL": "Ravens.png",
    "BUF": "Bills.png",
    "CAR": "Panthers.png",
    "CHI": "Bears.png",
    "CIN": "Bengals.png",
    "CLE": "Browns.png",
    "DAL": "Cowboys.png",
    "DEN": "Broncos.png",
    "DET": "Lions.png",
    "GB": "Packers.png",
    "HOU": "Texans.png",
    "IND": "Colts.png",
    "JAX": "Jaguars.png",
    "KC": "Chiefs.png",
    "LAC": "Chargers.png",
    "LA": "Rams.png",
    "LV": "Raiders.png",
    "MIA": "Dolphins.png",
    "MIN": "Vikings.png",
    "NE": "Patriots.png",
    "NO": "Saints.png",
    "NYG": "Giants.png",
    "NYJ": "Jets.png",
    "PHI": "Eagles.png",
    "PIT": "Steelers.png",
    "SEA": "Seahawks.png",
    "SF": "49ers.png",
    "TB": "Buccaneers.png",
    "TEN": "Titans.png",
    "WAS": "Commanders.png",
}

# ============================================================
# Snapshot helper (MUST MATCH TRAINING)
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

    print("=" * 90)
    print("TEAM-LEVEL DEEP HELP ANALYSIS (STRUCTURE vs EXECUTION)")
    print("=" * 90)

    # --------------------------------------------------------
    # Load model + scaler
    # --------------------------------------------------------
    model = torch.load("./safety_model_best.pth", weights_only=False)
    model.eval()
    scaler = joblib.load("./scaler_best.pkl")

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
    input_df = build_pre_snapshot(input_df, 2)
    output_df = build_pre_snapshot(output_df, 0)

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
    # Align input/output
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

    # --------------------------------------------------------
    # Ground truth: arrival in time
    # --------------------------------------------------------
    output_df["distance_to_ball"] = np.sqrt(
        output_df["ball_vec_x_clean"]**2 +
        output_df["ball_vec_y_clean"]**2
    )
    input_df["actual"] = (output_df["distance_to_ball"] <= 4).astype(int)

    # --------------------------------------------------------
    # Predictions (DHAR)
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

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze()
        input_df["pred_prob"] = torch.sigmoid(logits).numpy()

    # --------------------------------------------------------
    # Team aggregation
    # --------------------------------------------------------
    team_stats = (
        input_df
        .groupby("defensive_team")
        .agg(
            mean_dhar=("pred_prob", "mean"),
            arrival_rate=("actual", "mean"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
<<<<<<< HEAD
    # BIVARIATE QUADRANT PLOT WITH UNIFORM LOGO SIZES
=======
    # BIVARIATE QUADRANT PLOT
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
    # --------------------------------------------------------
    mean_dhar = team_stats["mean_dhar"].mean()
    mean_arrival = team_stats["arrival_rate"].mean()

    x_min, x_max = team_stats["mean_dhar"].min(), team_stats["mean_dhar"].max()
    y_min, y_max = team_stats["arrival_rate"].min(), team_stats["arrival_rate"].max()

<<<<<<< HEAD
    # Add padding to axis limits
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    fig, ax = plt.subplots(figsize=(12, 9))
    logo_dir = "./../Logos"  # Adjust path as needed
    
    # Set axis limits FIRST
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Plot team logos with UNIFORM sizing (no stretching)
=======
    fig, ax = plt.subplots(figsize=(12, 9))
    logo_dir = "../Logos"

>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
    for _, r in team_stats.iterrows():
        team = r["defensive_team"]
        logo_file = TEAM_LOGO_MAP.get(team)

        if logo_file:
            path = os.path.join(logo_dir, logo_file)
            if os.path.exists(path):
<<<<<<< HEAD
                try:
                    # Load image
                    img = Image.open(path).convert("RGBA")
                    
                    # KEY: Resize to fixed HEIGHT, maintain aspect ratio
                    target_height = 50  # pixels
                    aspect_ratio = img.width / img.height
                    target_width = int(target_height * aspect_ratio)
                    
                    # Resize with maintained aspect ratio (no stretching)
                    img = img.resize((target_width, target_height), Image.LANCZOS)
                    
                    # Add to plot
                    imagebox = OffsetImage(img, zoom=0.4)  # Adjust zoom for final size
                    ab = AnnotationBbox(
                        imagebox,
                        (r["mean_dhar"], r["arrival_rate"]),
                        frameon=False,
                        box_alignment=(0.5, 0.5)
                    )
                    ax.add_artist(ab)
                except Exception as e:
                    print(f"Warning: Could not load logo for {team}: {e}")
                    # Fallback to text
                    ax.text(
                        r["mean_dhar"],
                        r["arrival_rate"],
                        team,
                        fontsize=8,
                        ha="center",
                        va="center"
                    )

    # Reference lines
    ax.axvline(mean_dhar, linestyle=":", color="black", linewidth=1.5, alpha=0.7)
    ax.axhline(mean_arrival, linestyle=":", color="black", linewidth=1.5, alpha=0.7)

    # Get current plot limits (after setting them)
    x_plot_min, x_plot_max = ax.get_xlim()
    y_plot_min, y_plot_max = ax.get_ylim()

    # --------------------------------------------------------
    # QUADRANT LABELS - Positioned at top/bottom of each quadrant
    # --------------------------------------------------------
    # Calculate centers of each quadrant
    left_x_center = (x_plot_min + mean_dhar) / 2
    right_x_center = (mean_dhar + x_plot_max) / 2
    
    # Position labels at top/bottom with offset from edge
    label_offset_from_edge = (y_plot_max - y_plot_min) * 0.08
    
    # TOP LEFT: Low Stress, High Arrival
    ax.text(
        left_x_center,
        y_plot_max - label_offset_from_edge,
        "Low Expectation\nHigh Execution",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3, edgecolor="none")
    )

    # TOP RIGHT: High Stress, High Arrival
    ax.text(
        right_x_center,
        y_plot_max - label_offset_from_edge,
        "High Expectation\nHigh Execution",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3, edgecolor="none")
    )

    # BOTTOM LEFT: Low Stress, Low Arrival
    ax.text(
        left_x_center,
        y_plot_min + label_offset_from_edge,
        "Low Expectation\nLow Execution",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="orange", alpha=0.3, edgecolor="none")
    )

    # BOTTOM RIGHT: High Stress, Low Arrival
    ax.text(
        right_x_center,
        y_plot_min + label_offset_from_edge,
        "High Expectation\nLow Execution",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.3, edgecolor="none")
    )

    # Labels and title
    ax.set_xlabel("Mean Predicted Arrival Probability (DHAR)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Arrival Rate (Help Arrives in Time)", fontsize=13, fontweight="bold")
    ax.set_title("Team Deep Help Profiles: Expected vs Realized Performance", fontsize=15, fontweight="bold", pad=20)

    # Grid
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("team_dhar_vs_arrival_quadrants.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Save stats
    team_stats.to_csv("team_deep_help_stats.csv", index=False)
    
    print("\n" + "="*90)
    print("✓ Saved: team_dhar_vs_arrival_quadrants.png")
    print("✓ Saved: team_deep_help_stats.csv")
    print("="*90)


if __name__ == "__main__":
    main()
=======
                img = Image.open(path).convert("RGBA")
                img = img.resize((22, 25), Image.LANCZOS)  # FIXED SIZE

                ab = AnnotationBbox(
                    OffsetImage(img, zoom=1.0),
                    (r["mean_dhar"], r["arrival_rate"]),
                    frameon=False,
                    box_alignment=(0.5, 0.5)
                )
                ax.add_artist(ab)

    # Reference lines
    ax.axvline(mean_dhar, linestyle=":", color="black", linewidth=1)
    ax.axhline(mean_arrival, linestyle=":", color="black", linewidth=1)

    # Axis limits
    ax.set_xlim(x_min - 0.02, x_max + 0.02)
    ax.set_ylim(y_min - 0.03, y_max + 0.03)

    # Get plot bounds AFTER limits
    y_plot_min, y_plot_max = ax.get_ylim()
    y_offset = (y_plot_max - y_plot_min) * 0.22

    # --------------------------------------------------------
    # QUADRANT LABELS (TOP / BOTTOM, NOT MIDDLE)
    # --------------------------------------------------------
    ax.text(
        (x_min + mean_dhar) / 2,
        y_plot_max - y_offset,
        "Low Stress\nHigh Arrival",
        ha="center", va="top", fontsize=12, alpha=0.75
    )

    ax.text(
        (mean_dhar + x_max) / 2,
        y_plot_max - y_offset,
        "High Stress\nHigh Arrival",
        ha="center", va="top", fontsize=12, alpha=0.75
    )

    ax.text(
        (x_min + mean_dhar) / 2,
        y_plot_min + y_offset,
        "Low Stress\nLow Arrival",
        ha="center", va="bottom", fontsize=12, alpha=0.75
    )

    ax.text(
        (mean_dhar + x_max) / 2,
        y_plot_min + y_offset,
        "High Stress\nLow Arrival",
        ha="center", va="bottom", fontsize=12, alpha=0.75
    )

    # Labels
    ax.set_xlabel("Mean DHAR (Structural Difficulty)", fontsize=12)
    ax.set_ylabel("Arrival Rate (Help Arrives in Time)", fontsize=12)
    ax.set_title("Team Deep Help Profiles: Structure vs Execution", fontsize=14)

    plt.tight_layout()
    plt.savefig("team_dhar_vs_arrival_quadrants.png", dpi=200)
    plt.show()

    team_stats.to_csv("team_deep_help_stats.csv", index=False)
    print("✓ Saved team_dhar_vs_arrival_quadrants.png")
    print("✓ Saved team_deep_help_stats.csv")


if __name__ == "__main__":
    main()
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
