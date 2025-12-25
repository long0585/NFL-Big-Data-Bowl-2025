import numpy as np
import pandas as pd

def cover_1_help_safety(input_df, output_df):
    """The cover 1 help safety is defined as the defender that has the largest
    x_clean position when the play begins.

    Returns updated (input_df, output_df) with:
      - helper = 1 for the chosen (unique_play_id, nfl_id)
      - helper = 0 for everyone else in that play (for ALL frames)
    """
    input_out = input_df.copy()
    output_out = output_df.copy()

    # Default to 0
    input_out["helper"] = 0
    output_out["helper"] = 0

    # Get first frame snapshot
    snap = input_out[input_out["frame_id"] == 1]

    # Filter for coverage defenders that are included in output
    snap = snap[
        (snap["player_role"] == "Defensive Coverage") &
        (snap["player_to_predict"] == True)
    ]

    # If no candidates, keep helper=0 for all
    if snap.empty:
        return input_out, output_out

    # Pick the deepest (max x_clean) defender per play
    idx = snap.groupby("unique_play_id")["x_clean"].idxmax()
    helper_pairs = (
        snap.loc[idx, ["unique_play_id", "nfl_id"]]
        .drop_duplicates(subset=["unique_play_id"])
        .rename(columns={"nfl_id": "helper_nfl_id"})
    )

    # Apply helper=1 to ALL frames in BOTH dfs for that (play, player)
    input_out = input_out.merge(helper_pairs, on="unique_play_id", how="left")
    output_out = output_out.merge(helper_pairs, on="unique_play_id", how="left")

    input_out.loc[input_out["nfl_id"] == input_out["helper_nfl_id"], "helper"] = 1
    output_out.loc[output_out["nfl_id"] == output_out["helper_nfl_id"], "helper"] = 1

    # Drop helper column
    input_out = input_out.drop(columns=["helper_nfl_id"])
    output_out = output_out.drop(columns=["helper_nfl_id"])

    input_out["helper"] = input_out["helper"].fillna(0).astype(int)
    output_out["helper"] = output_out["helper"].fillna(0).astype(int)

    return input_out, output_out

def cover_2_4_6_help_safety(input_df, output_df):
    """The cover 2/4/6 help safety is defined as the defender on the same side of 
    the ball's landing location that has the largest x_clean position when the 
    play begins.

    Returns updated (input_df, output_df) with:
      - helper = 1 for the chosen (unique_play_id, nfl_id)
      - helper = 0 for everyone else in that play (for ALL frames)
    """
    input_out = input_df.copy()
    output_out = output_df.copy()

    # Default everyone to 0
    input_out["helper"] = 0
    output_out["helper"] = 0

    # Field is 53.3 yards wide
    MID_Y = 53.3 / 2

    # Get first frame snapshot
    snap = input_out[input_out["frame_id"] == 1]

    # Filter for coverage defenders that are included in output
    snap = snap[
        (snap["player_role"] == "Defensive Coverage") &
        (snap["player_to_predict"] == True)
    ]

    # Keep helper=0 for all if empty
    if snap.empty:
        return input_out, output_out

    # Determine which half the ball is landing on for each play (based on ball_land_y_clean)
    # True: top half >= MID_Y, False: bottom half < MID_Y
    land_side = (
        snap[["unique_play_id", "ball_land_y_clean"]]
        .drop_duplicates(subset=["unique_play_id"])
        .assign(land_top=lambda d: d["ball_land_y_clean"] >= MID_Y)
    )

    snap = snap.merge(land_side[["unique_play_id", "land_top"]], on="unique_play_id", how="left")
    snap["player_top"] = snap["y_clean"] >= MID_Y

    # Keep safeties on the same half as the landing spot
    same_half = snap[snap["player_top"] == snap["land_top"]]

    # If somehow no safety is on that half for a play, keep all helper=0
    if same_half.empty:
        return input_out, output_out

    # Pick deepest safety per play
    idx = same_half.groupby("unique_play_id")["x_clean"].idxmax()
    helper_pairs = (
        same_half.loc[idx, ["unique_play_id", "nfl_id"]]
        .drop_duplicates(subset=["unique_play_id"])
        .rename(columns={"nfl_id": "helper_nfl_id"})
    )

    # Apply helper=1 to ALL frames in BOTH dfs for that (play, player)
    input_out = input_out.merge(helper_pairs, on="unique_play_id", how="left")
    output_out = output_out.merge(helper_pairs, on="unique_play_id", how="left")

    input_out.loc[input_out["nfl_id"] == input_out["helper_nfl_id"], "helper"] = 1
    output_out.loc[output_out["nfl_id"] == output_out["helper_nfl_id"], "helper"] = 1

    # Drop helper column
    input_out = input_out.drop(columns=["helper_nfl_id"])
    output_out = output_out.drop(columns=["helper_nfl_id"])

    input_out["helper"] = input_out["helper"].fillna(0).astype(int)
    output_out["helper"] = output_out["helper"].fillna(0).astype(int)

    return input_out, output_out

def cover_3_help_safety(input_df, output_df):
    """The cover 3 help safety is defined as the defender on the same third of 
    the ball's landing location that has the largest x_clean position when the 
    play begins.

    Returns updated (input_df, output_df) with:
      - helper = 1 for the chosen (unique_play_id, nfl_id)
      - helper = 0 for everyone else in that play (for ALL frames)
    """
    input_out = input_df.copy()
    output_out = output_df.copy()

    # Default everyone to 0
    input_out["helper"] = 0
    output_out["helper"] = 0

    # Field width is 53.3 yards; split into thirds along y
    FIELD_Y = 53.3
    T1 = FIELD_Y / 3.0
    T2 = 2.0 * FIELD_Y / 3.0

    def y_third(y):
        # 0 = bottom, 1 = middle, 2 = top
        if pd.isna(y):
            return np.nan
        if y < T1:
            return 0
        elif y < T2:
            return 1
        else:
            return 2

    # Get first frame snapshot
    snap = input_out[input_out["frame_id"] == 1]

    # Filter for coverage defenders that are included in output
    snap = snap[
        (snap["player_role"] == "Defensive Coverage") &
        (snap["player_to_predict"] == True)
    ]

    # If no candidates, keep helper=0 for all
    if snap.empty:
        return input_out, output_out

    # Determine landing third per play
    land_third = (
        snap[["unique_play_id", "ball_land_y_clean"]]
        .drop_duplicates(subset=["unique_play_id"])
        .assign(land_third=lambda d: d["ball_land_y_clean"].apply(y_third))
    )

    snap = snap.merge(land_third[["unique_play_id", "land_third"]], on="unique_play_id", how="left")
    snap["player_third"] = snap["y_clean"].apply(y_third)

    # Keep safeties on the same third as landing
    same_third = snap[snap["player_third"] == snap["land_third"]]

    # If no deep player in that third, keep helper=0 for all
    if same_third.empty:
        return input_out, output_out

    # Pick deepest safety per play
    idx = same_third.groupby("unique_play_id")["x_clean"].idxmax()
    helper_pairs = (
        same_third.loc[idx, ["unique_play_id", "nfl_id"]]
        .drop_duplicates(subset=["unique_play_id"])
        .rename(columns={"nfl_id": "helper_nfl_id"})
    )

    # Apply helper=1 to ALL frames in BOTH dfs for that (play, player)
    input_out = input_out.merge(helper_pairs, on="unique_play_id", how="left")
    output_out = output_out.merge(helper_pairs, on="unique_play_id", how="left")

    input_out.loc[input_out["nfl_id"] == input_out["helper_nfl_id"], "helper"] = 1
    output_out.loc[output_out["nfl_id"] == output_out["helper_nfl_id"], "helper"] = 1

    # Drop helper column
    input_out = input_out.drop(columns=["helper_nfl_id"])
    output_out = output_out.drop(columns=["helper_nfl_id"])

    input_out["helper"] = input_out["helper"].fillna(0).astype(int)
    output_out["helper"] = output_out["helper"].fillna(0).astype(int)

    return input_out, output_out

def find_closest_db(input_df, output_df):
    """Assigns helper=1 to the closest db to the ball's landing location at 
    the time that the play begins. Used in determining the help defender for
    prevent defense.

    Returns updated (input_df, output_df) with:
      - helper = 1 for the chosen (unique_play_id, nfl_id)
      - helper = 0 for everyone else in that play (for ALL frames)
    """
    input_out = input_df.copy()
    output_out = output_df.copy()

    input_out["helper"] = 0
    output_out["helper"] = 0

    # Potential candidates for help defender
    candidates = input_df.copy()
    candidates = candidates[
        (candidates["player_role"] == "Defensive Coverage") &
        (candidates["player_to_predict"] == True)
    ].copy()

    # Leave helper=0 if somehow empty
    if candidates.empty:
        return input_out, output_out

    # Distance calculation
    candidates["distance_to_ball"] = np.sqrt(
        (candidates["x_clean"] - candidates["ball_land_x_clean"]) ** 2 +
        (candidates["y_clean"] - candidates["ball_land_y_clean"]) ** 2
    )

    # Closest DB per play
    idx = candidates.groupby("unique_play_id")["distance_to_ball"].idxmin()
    helper_pairs = (
        candidates.loc[idx, ["unique_play_id", "nfl_id"]]
        .drop_duplicates(subset=["unique_play_id"])
        .rename(columns={"nfl_id": "helper_nfl_id"})
        .reset_index(drop=True)
    )

    # Apply helper=1 to ALL frames in BOTH dfs
    input_out = input_out.merge(helper_pairs, on="unique_play_id", how="left")
    output_out = output_out.merge(helper_pairs, on="unique_play_id", how="left")

    input_out.loc[input_out["nfl_id"] == input_out["helper_nfl_id"], "helper"] = 1
    output_out.loc[output_out["nfl_id"] == output_out["helper_nfl_id"], "helper"] = 1

    # Drop helper column
    input_out = input_out.drop(columns=["helper_nfl_id"])
    output_out = output_out.drop(columns=["helper_nfl_id"])

    input_out["helper"] = input_out["helper"].fillna(0).astype(int)
    output_out["helper"] = output_out["helper"].fillna(0).astype(int)

    return input_out, output_out