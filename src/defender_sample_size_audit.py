import numpy as np
import pandas as pd
import load_data

# ============================================================
# Snapshot helper (MATCHES TRAINING)
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

    print("[INFO]: Loading weeks 1–18")

    all_input = []

    for week in range(1, 19):
        print(f"[INFO]: Loading week {week}")
        inp, _ = load_data.load_data(week)
        all_input.append(inp)

    input_df = pd.concat(all_input, ignore_index=True)

    # --------------------------------------------------------
    # Snapshot timing (pre-throw)
    # --------------------------------------------------------
    input_df = build_pre_snapshot(input_df, frames_before=2)

    # --------------------------------------------------------
    # Filters (MATCH TRAINING + PLAYER ANALYSIS INTENT)
    # --------------------------------------------------------
    input_df = input_df[
        (input_df["team_coverage_type"] != "COVER_0_MAN") &
        (input_df["team_coverage_type"] != "COVER_6_ZONE") &
        (input_df["helper"] == 1) &
        (input_df["pass_length"] >= 15)
    ].copy()

    # --------------------------------------------------------
    # Count deep-help plays per defender
    # --------------------------------------------------------
    defender_counts = (
        input_df
        .groupby(["nfl_id", "player_position"])
        .agg(
            num_plays=("unique_play_id", "nunique")
        )
        .reset_index()
        .sort_values("num_plays", ascending=False)
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEFENDER DEEP-HELP SAMPLE SIZE AUDIT (PASS ≥ 15 YARDS)")
    print("=" * 80)

    print(f"Total defenders: {defender_counts.shape[0]}")
    print(f"Total deep-help plays: {defender_counts['num_plays'].sum()}")

    thresholds = [5, 10, 15, 20, 30]

    print("\nDefenders with at least N plays:")
    for t in thresholds:
        count = (defender_counts["num_plays"] >= t).sum()
        print(f"  ≥ {t:>2} plays: {count:>4} defenders")

    print("\nPercentiles of plays per defender:")
    for p in [50, 75, 90, 95]:
        val = np.percentile(defender_counts["num_plays"], p)
        print(f"  {p:>2}th percentile: {val:.1f} plays")

    # --------------------------------------------------------
    # Show top defenders by volume
    # --------------------------------------------------------
    print("\nTop 20 defenders by deep-help volume:")
    print(
        defender_counts
        .head(20)
        .to_string(index=False)
    )

    print("\n[INFO]: Sample size audit complete")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
