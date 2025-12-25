import numpy as np
import torch
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import load_data
from transformer_model import TransformerModel

# ============================================================
# Snapshot helper (MUST MATCH TRAINING)
# ============================================================
def label_point(ax, x, y, label, dx, dy, bold=False):
<<<<<<< HEAD
    if bold:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            textcoords="data",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.8),
        )
    else:
        ax.text(
            x + dx,
            y + dy,
            label,
            fontsize=8,
            color="black",
        )
=======
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords="data",
        fontsize=10 if bold else 8,
        fontweight="bold" if bold else "normal",
        ha="left" if dx > 0 else "right",
        va="bottom" if dy > 0 else "top",
        arrowprops=dict(
            arrowstyle="-",
            color="gray",
            linewidth=0.8,
            shrinkA=0,
            shrinkB=4,
        ),
    )
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7

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

    print("[INFO]: Loading model and scaler")

    # --------------------------------------------------------
    # Load model + scaler
    # --------------------------------------------------------
    model = torch.load("./safety_model_best.pth", weights_only=False)
    model.eval()

    scaler = joblib.load("./scaler_best.pkl")

    # --------------------------------------------------------
    # Load ALL DATA — Weeks 1–18
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

    input_df.reset_index(drop=True, inplace=True)
    output_df.reset_index(drop=True, inplace=True)

    # --------------------------------------------------------
    # Ground truth (arrival ≤ 4 yards)
    # --------------------------------------------------------
    output_df["actual"] = (
        np.sqrt(
            output_df["ball_vec_x_clean"]**2 +
            output_df["ball_vec_y_clean"]**2
        ) <= 4
    ).astype(int)

    input_df["actual"] = output_df["actual"].values

    # --------------------------------------------------------
    # Features (MATCH TRAINING)
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
    # Predict DHAR
    # --------------------------------------------------------
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze()
        prob = torch.sigmoid(logits).numpy()

    input_df["pred_prob"] = prob

    # --------------------------------------------------------
    # Residual = Actual − Expected
    # --------------------------------------------------------
<<<<<<< HEAD
    input_df["arrival_score"] = np.where(
        input_df["actual"] == 1,
        1.0, 0
    )
=======
    input_df["residual"] = input_df["actual"] - input_df["pred_prob"]
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7

    # --------------------------------------------------------
    # Player-level aggregation (≥12 plays)
    # --------------------------------------------------------
    player_df = (
        input_df
        .groupby(["nfl_id", "player_position"])
        .agg(
            mean_dhar=("pred_prob", "mean"),
<<<<<<< HEAD
            mean_arrival_score=("arrival_score", "mean"),
=======
            mean_residual=("residual", "mean"),
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
            num_plays=("unique_play_id", "nunique"),
        )
        .reset_index()
    )

    player_df = player_df[player_df["num_plays"] >= 12]

    # --------------------------------------------------------
    # Attach player names
    # --------------------------------------------------------
    name_map = (
        input_df
        .groupby("nfl_id")["player_name"]
        .first()
        .reset_index()
    )

    player_df = player_df.merge(name_map, on="nfl_id", how="left")

    # --------------------------------------------------------
<<<<<<< HEAD
    # Global reference values (for geometry-aware clustering)
    # --------------------------------------------------------
    league_mean_arrival = player_df["mean_arrival_score"].mean()
    league_std_arrival  = player_df["mean_arrival_score"].std()

    dhar_hi = player_df["mean_dhar"].quantile(0.67)  # EASY
    dhar_lo = player_df["mean_dhar"].quantile(0.33)  # HARD

    # --------------------------------------------------------
    # ARCHETYPES (GEOMETRY-AWARE, FINAL)
    # --------------------------------------------------------

    # Q1: Hard assignments, clearly high arrival rate
    high_difficulty_converters = player_df[
        (player_df["mean_dhar"] <= dhar_lo) &
        (player_df["mean_arrival_score"] >= league_mean_arrival + 0.5 * league_std_arrival)
    ].sort_values("mean_arrival_score", ascending=False).head(8)

    # Q2: Easy assignments, reliably converting
    assignment_executers = player_df[
        (player_df["mean_dhar"] >= dhar_hi) &
        (player_df["mean_arrival_score"] >= league_mean_arrival)
    ].sort_values("mean_arrival_score", ascending=False).head(8)

    # Q3: Easy assignments, failing to convert
    low_difficulty_non_converters = player_df[
        (player_df["mean_dhar"] >= dhar_hi) &
        (player_df["mean_arrival_score"] <= league_mean_arrival - 0.5 * league_std_arrival)
    ].sort_values("mean_arrival_score", ascending=True).head(8)

    # Q4: Hard assignments, low arrival rate (contextual stress)
    strained_defenders = player_df[
        (player_df["mean_dhar"] <= dhar_lo) &
        (player_df["mean_arrival_score"] <= league_mean_arrival)
    ].sort_values("mean_arrival_score", ascending=True).head(8)

=======
    # Difficulty tiers
    # --------------------------------------------------------
    dhar_hi = player_df["mean_dhar"].quantile(0.67)
    dhar_lo = player_df["mean_dhar"].quantile(0.33)

    high_diff = player_df[player_df["mean_dhar"] >= dhar_hi]
    low_diff  = player_df[player_df["mean_dhar"] <= dhar_lo]

    # --------------------------------------------------------
    # ARCHETYPES (SYMMETRIC, CONDITIONAL)
    # --------------------------------------------------------
    elite_df = (
        high_diff
        .sort_values("mean_residual", ascending=False)
        .head(8)
    )

    poor_easy_df = (
        low_diff
        .sort_values("mean_residual", ascending=True)
        .head(8)
    )

    context_over_df = (
        low_diff
        .sort_values("mean_residual", ascending=False)
        .head(8)
    )

    strained_df = (
        high_diff
        .sort_values("mean_residual", ascending=True)
        .head(8)
    )
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7

    # --------------------------------------------------------
    # Pretty print groups
    # --------------------------------------------------------
    def print_group(df, title):
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
        for _, r in df.iterrows():
            print(
                f"{r['player_name']:<25} | "
                f"ID: {int(r['nfl_id']):>6} | "
                f"Pos: {r['player_position']:<2} | "
                f"X (DHAR): {r['mean_dhar']:.3f} | "
<<<<<<< HEAD
                f"Y (Arrival Score): {r['mean_arrival_score']:+.3f} | "
                f"Plays: {int(r['num_plays'])}"
            )

    # Print Groups
    print_group(
        high_difficulty_converters,
        "HIGH-DIFFICULTY CONVERTERS (Hard Assignments, High Arrival Rate)"
    )

    print_group(
        assignment_executers,
        "ASSIGNMENT EXECUTERS (Easy Assignments, High Arrival Rate)"
        )

    print_group(
        low_difficulty_non_converters,
        "LOW-DIFFICULTY NON-CONVERTERS (Easy Assignments, Low Arrival Rate)"
    )

    print_group(
        strained_defenders,
        "STRAINED DEFENDERS (Hard Assignments, Low Arrival Rate)"
    )

=======
                f"Y (Residual): {r['mean_residual']:+.3f} | "
                f"Plays: {int(r['num_plays'])}"
            )

    print_group(elite_df, "HIGH-LEVERAGE DEEP HELPERS (Hard Assignments, Best Relative Outcomes)")
    print_group(poor_easy_df, "LOW-LEVERAGE POOR DEFENDERS (Easy Assignments, Worst Relative Outcomes)")
    print_group(context_over_df, "CONTEXTUAL OVERPERFORMERS (Easy Assignments)")
    print_group(strained_df, "STRAINED DEFENDERS (Hard Assignments)")
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7

    ANCHOR_NAMES = {
    "Daxton Hill",
    "Xavier McKinney",
    "Harrison Smith",
    "Jordan Whitehead",
    }

    SECONDARY_NAMES = {
<<<<<<< HEAD
    "Eddie Jackson", "Julian Love", "James Bradberry",
    "Jabrill Peppers", "Russ Yeast", "Alohi Gilman",
    "Quandre Diggs", "Elijah Hicks", "Nick Scott",
=======
    "Micah Hyde", "Julian Love", "James Bradberry",
    "Tyrann Mathieu", "Russ Yeast", "Alohi Gilman",
    "Patrick Peterson", "Elijah Hicks", "Donovan Wilson",
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
    "Cam Taylor-Britt", "Richie Grant", "DJ Turner II",
    }
    
    label_offsets = {
    # High-Leverage anchors
<<<<<<< HEAD
    "Eddie Jackson":     (0.008,  -0.015),
    "Julian Love":       (0.008,  0.008),
    "James Bradberry":   (0.008,  0.008),

    # Low-Leverage Poor anchors
    "Jabrill Peppers":   (0.008,  0.015),
    "Russ Yeast":       (0.008,  0.008),
    "Alohi Gilman":     (-0.06,  -0.015),

    # Contextual Overperformers anchors
    "Quandre Diggs": (0.015,  0.008),
    "Elijah Hicks":     (-0.06,  -0.015),
    "Nick Scott":   (0.015,  -0.01),

    # Strained anchors
    "Cam Taylor-Britt": (0.008,  0.008),
    "Richie Grant":    (0.008,  0.008),
    "DJ Turner II":    (-0.06,  -0.01),
    
    # Anchor names (PRIMARY)
    "Daxton Hill":        (0.015,  0.015),
    "Xavier McKinney":    (-0.045,  0.02),
    "Harrison Smith":    (0.015,  0.015),
    "Jordan Whitehead":  (0.015,  0.015),
=======
    "Micah Hyde":        (0.020,  0.020),
    "Julian Love":       (0.020,  0.030),
    "James Bradberry":   (0.020,  0.015),

    # Low-Leverage Poor anchors
    "Tyrann Mathieu":   (-0.045, -0.020),
    "Russ Yeast":       (-0.045, -0.030),
    "Alohi Gilman":     (-0.045, -0.015),

    # Contextual Overperformers anchors
    "Patrick Peterson": (-0.045,  0.020),
    "Elijah Hicks":     (-0.045,  0.030),
    "Donovan Wilson":   (-0.045,  0.015),

    # Strained anchors
    "Cam Taylor-Britt": (0.020, -0.030),
    "Richie Grant":    (0.020, -0.025),
    "DJ Turner II":    (0.020, -0.040),
    
    # Anchor names (PRIMARY)
    "Daxton Hill":        (-0.030,  0.035),
    "Xavier McKinney":    (0.025,   0.020),
    "Harrison Smith":    (0.025,  -0.025),
    "Jordan Whitehead":  (-0.030, -0.020),
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
    }

    # --------------------------------------------------------
    # Plot with highlighted archetypes
    # --------------------------------------------------------
    
    plt.figure(figsize=(8, 8))

    plt.scatter(
        player_df["mean_dhar"],
<<<<<<< HEAD
        player_df["mean_arrival_score"],
=======
        player_df["mean_residual"],
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
        s=player_df["num_plays"] * 6,
        alpha=0.30,
        color="gray",
        edgecolor="none",
        label="All defenders",
    )

    def highlight(df, color, label):
        plt.scatter(
            df["mean_dhar"],
<<<<<<< HEAD
            df["mean_arrival_score"],
=======
            df["mean_residual"],
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
            s=df["num_plays"] * 9,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            label=label,
        )

<<<<<<< HEAD
    highlight(high_difficulty_converters, "green", "High-Difficulty Converters")
    highlight(assignment_executers, "blue", "Assignment Executers")
    highlight(low_difficulty_non_converters, "red", "Low-Difficulty Non-Converters")
    highlight(strained_defenders, "orange", "Strained Defenders")

=======
    highlight(elite_df, "green",  "High-Leverage (Hard, Best)")
    highlight(poor_easy_df, "red", "Low-Leverage Poor (Easy, Worst)")
    highlight(context_over_df, "blue", "Contextual Overperformers")
    highlight(strained_df, "orange", "Strained Defenders")
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7

    ax = plt.gca()

    for _, r in player_df.iterrows():
        name = r["player_name"]

        if name not in label_offsets:
            continue

        dx, dy = label_offsets[name]

        label_point(
            ax,
            r["mean_dhar"],
<<<<<<< HEAD
            r["mean_arrival_score"],
=======
            r["mean_residual"],
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
            name,
            dx,
            dy,
            bold=(name in ANCHOR_NAMES)
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(player_df["mean_dhar"].mean(), linestyle=":", linewidth=1)

<<<<<<< HEAD
    plt.xlabel("Mean Predicted Arrival Probability (DHAR)")
    plt.ylabel(
        "Mean Arrival Score\n"
        "(Arrivals = +1, Otherwise = 0)"
    )
    plt.title(
        "Player-Level Deep Help Performance\n"
        "(≥12 Deep-Help Plays, Arrival Success with Contextual Penalties)"
=======
    plt.xlabel("Mean Predicted DHAR (Assignment Difficulty)")
    plt.ylabel("Mean Over / Under Performance (Actual − Expected)")
    plt.title(
        "Player-Level Deep Help Performance\n"
        "(≥12 Deep-Help Plays, Relative to Expectation)"
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7
    )

    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
<<<<<<< HEAD
    plt.savefig("DHAR_player_eval_graph.png")
=======
    plt.show()
>>>>>>> 145cd5862cd1c8da9728275a89e99d48309811a7


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
