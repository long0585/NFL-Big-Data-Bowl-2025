# safety_model_with_field_heatmaps.py

import os
import numpy as np
import torch
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import load_data
from transformer_model import TransformerModel  # needed for torch.load deserialization


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
# Field drawing (prefer draw_scene.create_football_field, fallback if not available)
# ============================================================
def _fallback_create_football_field(ax):
    """Simple NFL field. Uses x in [0,120], y in [0,53.3]."""
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect("equal", adjustable="box")

    # Field outline
    ax.add_patch(plt.Rectangle((0, 0), 120, 53.3, fill=False, linewidth=2))

    # Endzones
    ax.add_patch(plt.Rectangle((0, 0), 10, 53.3, fill=False, linewidth=2))
    ax.add_patch(plt.Rectangle((110, 0), 10, 53.3, fill=False, linewidth=2))

    # Yard lines every 5 yards
    for x in range(10, 111, 5):
        lw = 2 if (x % 10 == 0) else 1
        ax.plot([x, x], [0, 53.3], linewidth=lw)

    # Remove axes clutter
    ax.set_xticks([])
    ax.set_yticks([])


def create_football_field(ax):
    """Use draw_scene.py's field if available; otherwise draw a fallback field."""
    try:
        # draw_scene.py should already have create_football_field used for animations
        from draw_scene import create_football_field as cs_create_football_field
        return cs_create_football_field(ax)
    except Exception:
        return _fallback_create_football_field(ax)


# ============================================================
# Heatmap helpers (on-field)
# ============================================================
def _pick_xy_columns(df: pd.DataFrame):
    """Pick defender position columns for field plotting."""
    candidates = [("x_clean", "y_clean"), ("x", "y"), ("x_coord", "y_coord"), ("x_position", "y_position")]
    for xc, yc in candidates:
        if xc in df.columns and yc in df.columns:
            return xc, yc
    raise KeyError(
        "Could not find defender field position columns. "
        "Expected one of: (x_clean,y_clean), (x,y), (x_coord,y_coord), (x_position,y_position)."
    )


def _hist2d_on_field(ax, x, y, bins=40, alpha=0.55, log_scale=True):
    """Draw 2D histogram heatmap on the field (x:[0,120], y:[0,53.3])."""
    x_range = (0.0, 120.0)
    y_range = (0.0, 53.3)

    H, xedges, yedges = np.histogram2d(
        x.astype(float),
        y.astype(float),
        bins=bins,
        range=[x_range, y_range],
    )

    if H.max() == 0:
        return None

    norm = None
    if log_scale:
        norm = LogNorm(vmin=1, vmax=max(1, int(H.max())))

    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        alpha=alpha,
        norm=norm,
    )
    return im


def plot_field_heatmaps_by_coverage(
    df: pd.DataFrame,
    bins: int = 40,
    save_dir: str = "./coverage_field_heatmaps",
    coverage_filter: str | None = None,
    actual_filter: int | None = None,
):
    os.makedirs(save_dir, exist_ok=True)

    xcol, ycol = _pick_xy_columns(df)

    # Optional filters
    if coverage_filter is not None:
        df = df[df["team_coverage_type"] == coverage_filter].copy()
    if actual_filter is not None:
        df = df[df["actual"] == int(actual_filter)].copy()

    if df.empty:
        print("[WARNING]: No rows after filtering. Nothing to plot.")
        return

    cov_order = df["team_coverage_type"].value_counts().index.tolist()

    for cov in cov_order:
        d = df[df["team_coverage_type"] == cov].copy()

        if actual_filter is None:
            fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
            panels = [(1, "actual = 1 (made it)"), (0, "actual = 0 (late)")]
            axes_list = axes
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
            panels = [(int(actual_filter), f"actual = {int(actual_filter)}")]
            axes_list = [ax]

        for ax, (cls, title) in zip(axes_list, panels):
            create_football_field(ax)
            s = d[d["actual"] == cls]

            im = _hist2d_on_field(
                ax,
                s[xcol].values,
                s[ycol].values,
                bins=bins,
                alpha=0.55,
                log_scale=True,
            )

            ax.set_title(f"Coverage: {cov} | {title} | n={len(s)}")

            if im is not None:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fname = cov.replace("/", "_").replace(" ", "_")
        outpath = os.path.join(save_dir, f"field_heatmap_{fname}.png")
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"[INFO]: Saved field heatmaps to: {os.path.abspath(save_dir)}")
    print(f"[INFO]: Used defender position columns: {xcol}, {ycol}")


# ============================================================
# Main
# ============================================================
def main():
    # ---------------- CONFIG ----------------
    FRAMES_BEFORE_SNAPSHOT = 2   # ~0.2s pre-throw
    HEATMAP_BINS = 40
    HEATMAP_SAVE_DIR = "./coverage_field_heatmaps"
    COVERAGE_FILTER = None       # e.g. "COVER_3_ZONE"
    ACTUAL_FILTER = None         # 0 or 1, or None for both panels

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
    input_df = build_pre_snapshot(input_df, frames_before=FRAMES_BEFORE_SNAPSHOT)  # ~0.2s pre-throw
    output_df = build_pre_snapshot(output_df, frames_before=0)                     # throw frame

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
    # Ground truth (same as your model script)
    # --------------------------------------------------------
    output_df["distance_to_ball"] = np.sqrt(
        output_df["ball_vec_x_clean"]**2 +
        output_df["ball_vec_y_clean"]**2
    )
    output_df["actual"] = (output_df["distance_to_ball"] <= 4).astype(int)

    # Attach actual to input snapshot rows
    input_df["actual"] = output_df["actual"].values

    # --------------------------------------------------------
    # Features + Predict (kept from your script)
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
        prob = torch.sigmoid(logits).numpy()

    input_df["pred_prob"] = prob
    input_df["predicted"] = (prob > threshold).astype(int)

    # --------------------------------------------------------
    # FIELD HEATMAPS (by coverage type, actual=1 vs actual=0)
    # --------------------------------------------------------
    plot_field_heatmaps_by_coverage(
        input_df,
        bins=HEATMAP_BINS,
        save_dir=HEATMAP_SAVE_DIR,
        coverage_filter=COVERAGE_FILTER,
        actual_filter=ACTUAL_FILTER,
    )

    # --------------------------------------------------------
    # Optional: keep your df_00 debug print
    # --------------------------------------------------------
    df_00 = input_df[
        (input_df["actual"] == 0) &
        (input_df["predicted"] == 0) &
        (input_df["pass_length"] >= 25)
    ].copy()

    df_00 = (
        df_00
        .sort_values(by=["pred_prob", "pass_length"], ascending=[True, False])
        .head(30)
    )

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
            f"Pos: {row.get('player_position', 'NA'):>3} | "
            f"P(arrive): {row['pred_prob']:.3f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()