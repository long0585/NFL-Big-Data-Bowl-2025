# pass_result_by_ground_truth.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data


def build_pre_snapshot(input_df: pd.DataFrame, frames_before: int) -> pd.DataFrame:
    """Build snapshot of tracking data N frames before the throw"""
    throw_frames = (
        input_df.groupby(["unique_play_id"])["frame_id"]
        .max()
        .reset_index(name="frame_id_throw")
    )

    pre_throw_frames = throw_frames.copy()
    pre_throw_frames["frame_id_pre"] = np.maximum(
        pre_throw_frames["frame_id_throw"] - frames_before,
        1,
    )

    snapshot_pre_throw = input_df.merge(
        pre_throw_frames[["unique_play_id", "frame_id_pre"]],
        left_on=["unique_play_id", "frame_id"],
        right_on=["unique_play_id", "frame_id_pre"],
        how="inner",
    ).drop(columns=["frame_id_pre"])

    return snapshot_pre_throw


def save_probability_table_to_png(prob_table: pd.DataFrame, out_path: str) -> None:
    """Render a single probability table (with an Occurrences column) to a PNG file."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 4.5))
    ax.axis("off")

    ax.set_title("The Effect of Help Arrival on Pass Outcome")
    t = ax.table(
        cellText=prob_table.values,
        rowLabels=prob_table.index.tolist(),
        colLabels=["Completion", "Incompletion", "Interception", "Occurrences"],
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    # ----------------------------
    # Load ALL data (weeks 1-18)
    # ----------------------------
    all_input = []
    all_output = []
    for week in range(1, 19):
        print(f"Loading week {week}...")
        week_input, week_output = load_data.load_data(week)
        all_input.append(week_input)
        all_output.append(week_output)

    input_df = pd.concat(all_input, ignore_index=True)
    output_df = pd.concat(all_output, ignore_index=True)

    # ----------------------------
    # Snapshots (match safety_model.py)
    # ----------------------------
    input_df = build_pre_snapshot(input_df, frames_before=2)
    output_df = build_pre_snapshot(output_df, frames_before=0)

    # ----------------------------
    # Filters (match safety_model.py)
    # ----------------------------
    input_df = input_df[input_df["team_coverage_type"] != "COVER_0_MAN"]
    output_df = output_df[output_df["team_coverage_type"] != "COVER_0_MAN"]

    input_df = input_df[input_df["team_coverage_type"] != "COVER_6_ZONE"]
    output_df = output_df[output_df["team_coverage_type"] != "COVER_6_ZONE"]

    input_df = input_df[input_df["helper"] == 1]
    output_df = output_df[output_df["helper"] == 1]

    input_df = input_df[input_df["pass_length"] >= 15]
    output_df = output_df[output_df["pass_length"] >= 15]

    input_df = input_df.sort_values(by="unique_play_id").reset_index(drop=True)
    output_df = output_df.sort_values(by="unique_play_id").reset_index(drop=True)

    # ----------------------------
    # Alignment check
    # ----------------------------
    X_check = input_df[["unique_play_id", "nfl_id"]].reset_index(drop=True)
    y_check = output_df[["unique_play_id", "nfl_id"]].reset_index(drop=True)
    if not (X_check == y_check).all().all():
        raise ValueError("[ERROR]: Input and output not aligned")

    # ----------------------------
    # Ground truth label (match safety_model.py)
    # ----------------------------
    output_df["distance_to_ball"] = (
        output_df["ball_vec_x_clean"] ** 2 + output_df["ball_vec_y_clean"] ** 2
    ) ** 0.5
    output_df["gets_to_ball_in_time"] = (output_df["distance_to_ball"] <= 4).astype(int)

    if "pass_result" not in output_df.columns:
        raise KeyError("[ERROR]: output_df is missing 'pass_result' column.")

    # Keep the three results you care about (and drop NaNs)
    df = output_df[["gets_to_ball_in_time", "pass_result"]].dropna()
    df = df[df["pass_result"].isin(["C", "I", "IN"])]

    # ----------------------------
    # Probability table by ground truth
    # ----------------------------
    col_order = ["C", "I", "IN"]

    counts = pd.crosstab(df["gets_to_ball_in_time"], df["pass_result"]) \
        .reindex(columns=col_order) \
        .fillna(0) \
        .astype(int)

    n_by_y = counts.sum(axis=1)

    # Proportions (as %) for each y
    props_pct = (
        counts.div(n_by_y, axis=0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        * 100.0
    )

    # Add Occurrences column
    prob_table = props_pct.copy()
    prob_table["Occurrences"] = n_by_y

    # Add Overall row
    overall_counts = df["pass_result"].value_counts().reindex(col_order).fillna(0).astype(int)
    overall_props = (overall_counts / overall_counts.sum() * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    overall_row = overall_props.copy()
    overall_row["Occurrences"] = int(overall_counts.sum())
    overall_row.name = "Overall"
    prob_table = pd.concat([prob_table, overall_row.to_frame().T])

    # Format percentages as strings
    prob_table_fmt = prob_table.copy()
    for c in col_order:
        prob_table_fmt[c] = prob_table_fmt[c].map(lambda x: f"{float(x):.1f}%")
    prob_table_fmt["Occurrences"] = prob_table_fmt["Occurrences"].astype(int)

    # Label rows
    label_map = {
        0: "distance_to_ball > 4yd",
        1: "distance_to_ball <= 4yd",
        "Overall": "Overall",
    }
    prob_table_fmt.index = [label_map.get(i, str(i)) for i in prob_table_fmt.index]

    print("\n==============================")
    print("PROPORTIONS by ground truth (y) and pass_result")
    print("(rows sum to 100.0% except Occurrences)")
    print("==============================")
    print(prob_table_fmt.to_string())

    out_png = "pass_result_by_ground_truth_table.png"
    save_probability_table_to_png(prob_table_fmt, out_png)
    print(f"\n[INFO]: Saved probability table PNG -> {out_png}")


if __name__ == "__main__":
    main()