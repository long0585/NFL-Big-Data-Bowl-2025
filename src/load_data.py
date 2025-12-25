"""
Creates pandas dataframes that append supplementary data to both input and output
data, and also adds/edits data fields as described below:
Standardizes play direction to be left-to-right for both input and output dfs
    New fields: x_clean, y_clean, o_clean (input only), dir_clean (input only),
                ball_land_x_clean, ball_land_y_clean, s_clean, a_clean
Adds velocity indicators to both input and output dfs
    New fields: v_x and v_y
Adds the following fields to the output df:
    New fields: player_side, play_direction (only used to calculate x_clean, y_clean)

Usage: python3 ./load_data.py [week_num]
"""
import numpy as np
import pandas as pd
import sys
import time
import help_safety_logic

def build_pre_snapshot(input_df: pd.DataFrame, frames_before: int) -> pd.DataFrame:
    """Build snapshot of tracking data two frames before the throw
    """
    # Last frame per (game_id, play_id)
    throw_frames = (
        input_df
        .groupby(['unique_play_id'])["frame_id"]
        .max()
        .reset_index(name="frame_id_throw")
    )

    # Pre-throw frame: 2 frames before, floored at 1
    pre_throw_frames = throw_frames.copy()
    pre_throw_frames["frame_id_pre"] = np.maximum(
        pre_throw_frames["frame_id_throw"] - frames_before,
        1,
    )

    # Snapshot at pre-throw frame
    snapshot_pre_throw = input_df.merge(
        pre_throw_frames[['unique_play_id', "frame_id_pre"]],
        left_on=['unique_play_id', "frame_id"],
        right_on=['unique_play_id', "frame_id_pre"],
        how="inner",
    )

    snapshot_pre_throw = snapshot_pre_throw.drop(columns=["frame_id_pre"])

    return snapshot_pre_throw

def assign_helper_field(input_df, output_df):
  input_out = input_df.copy()
  output_out = output_df.copy()

  # Remove cover 0 man
  input_out = input_out[input_out['team_coverage_type'] != 'COVER_0_MAN']
  output_out = output_out[output_out['team_coverage_type'] != 'COVER_0_MAN']

  # Set helper=0 as a default
  input_out['helper'] = 0
  output_out['helper'] = 0

  # Coverage type is constant per play; build a per-play lookup
  play_cov = (
      input_out[['unique_play_id', 'team_coverage_type']]
      .drop_duplicates(subset=['unique_play_id'])
      .reset_index(drop=True)
  )

  def _play_ids(mask):
      return play_cov.loc[mask, 'unique_play_id'].tolist()

  # Matching this way handles COVER_2_MAN / COVER_2_ZONE 
  cov1_ids = _play_ids(play_cov['team_coverage_type'].astype(str).str.contains('COVER_1', na=False))
  cov2_ids = _play_ids(play_cov['team_coverage_type'].astype(str).str.contains('COVER_2', na=False))
  cov3_ids = _play_ids(play_cov['team_coverage_type'].astype(str).str.contains('COVER_3', na=False))
  cov4_ids = _play_ids(play_cov['team_coverage_type'].astype(str).str.contains('COVER_4', na=False))
  cov6_ids = _play_ids(play_cov['team_coverage_type'].astype(str).str.contains('COVER_6', na=False))
  prev_ids = _play_ids(play_cov['team_coverage_type'].astype(str).str.contains('PREVENT', na=False))

  helper_pairs_all = []

  # --- COVER 1 ---
  if cov1_ids:
      inp = input_out[input_out['unique_play_id'].isin(cov1_ids)]
      out = output_out[output_out['unique_play_id'].isin(cov1_ids)]
      inp_h, _ = help_safety_logic.cover_1_help_safety(inp, out)
      helper_pairs_all.append(
          inp_h.loc[inp_h['helper'] == 1, ['unique_play_id', 'nfl_id']].drop_duplicates().assign(helper=1)
      )

  # --- COVER 2 ---
  if cov2_ids:
      inp = input_out[input_out['unique_play_id'].isin(cov2_ids)]
      out = output_out[output_out['unique_play_id'].isin(cov2_ids)]
      inp_h, _ = help_safety_logic.cover_2_4_6_help_safety(inp, out)
      helper_pairs_all.append(
          inp_h.loc[inp_h['helper'] == 1, ['unique_play_id', 'nfl_id']].drop_duplicates().assign(helper=1)
      )

  # --- COVER 3 ---
  if cov3_ids:
      inp = input_out[input_out['unique_play_id'].isin(cov3_ids)]
      out = output_out[output_out['unique_play_id'].isin(cov3_ids)]
      inp_h, _ = help_safety_logic.cover_3_help_safety(inp, out)
      helper_pairs_all.append(
          inp_h.loc[inp_h['helper'] == 1, ['unique_play_id', 'nfl_id']].drop_duplicates().assign(helper=1)
      )
  # --- COVER 4 ---
  if cov4_ids:
      inp = input_out[input_out['unique_play_id'].isin(cov4_ids)]
      out = output_out[output_out['unique_play_id'].isin(cov4_ids)]
      inp_h, _ = help_safety_logic.cover_2_4_6_help_safety(inp, out)
      helper_pairs_all.append(
          inp_h.loc[inp_h['helper'] == 1, ['unique_play_id', 'nfl_id']].drop_duplicates().assign(helper=1)
      )

  # --- COVER 6 ---
  if cov6_ids:
      inp = input_out[input_out['unique_play_id'].isin(cov6_ids)]
      out = output_out[output_out['unique_play_id'].isin(cov6_ids)]
      inp_h, _ = help_safety_logic.cover_2_4_6_help_safety(inp, out)
      helper_pairs_all.append(
          inp_h.loc[inp_h['helper'] == 1, ['unique_play_id', 'nfl_id']].drop_duplicates().assign(helper=1)
      )

  # --- PREVENT ---
  if prev_ids:
      inp = input_out[input_out['unique_play_id'].isin(prev_ids)]
      out = output_out[output_out['unique_play_id'].isin(prev_ids)]
      inp_h, _ = help_safety_logic.find_closest_db(inp, out)
      helper_pairs_all.append(
          inp_h.loc[inp_h['helper'] == 1, ['unique_play_id', 'nfl_id']].drop_duplicates().assign(helper=1)
      )

  # Any remaining plays, revert to closest DB
  assigned_ids = set(cov1_ids + cov2_ids + cov3_ids + cov4_ids + cov6_ids + prev_ids)
  other_ids = [pid for pid in play_cov['unique_play_id'].tolist() if pid not in assigned_ids]
  if other_ids:
      inp = input_out[input_out['unique_play_id'].isin(other_ids)]
      out = output_out[output_out['unique_play_id'].isin(other_ids)]
      inp_h, _ = help_safety_logic.find_closest_db(inp, out)
      helper_pairs_all.append(
          inp_h.loc[inp_h['helper'] == 1, ['unique_play_id', 'nfl_id']].drop_duplicates().assign(helper=1)
      )

  # Merge helper pairs back onto all frames
  if helper_pairs_all:
      helper_pairs = (
          pd.concat(helper_pairs_all, ignore_index=True)
          .drop_duplicates(subset=['unique_play_id', 'nfl_id'])
          .reset_index(drop=True)
      )

      input_out = input_out.drop(columns=['helper'], errors='ignore').merge(
          helper_pairs, on=['unique_play_id', 'nfl_id'], how='left'
      )
      output_out = output_out.drop(columns=['helper'], errors='ignore').merge(
          helper_pairs, on=['unique_play_id', 'nfl_id'], how='left'
      )

      input_out['helper'] = input_out['helper'].fillna(0).astype(int)
      output_out['helper'] = output_out['helper'].fillna(0).astype(int)

  return input_out, output_out

def add_velocity_to_output(input_df, output_df, framerate=0.1):
    """
    Adds velocity vectors (v_x, v_y) to output_df. Useful for visualization.
    Helper function for clean_data.

    For the first output frame of each unique play, we use the last available 
    input frame for that same player as the previous position.
    """

    # Get last input frame per player/play to serve as prev for first output frame
    last_input = (
        input_df
        .sort_values(['unique_play_id', "nfl_id", "frame_id"])
        .groupby(['unique_play_id', "nfl_id"], as_index=False)
        .tail(1)
        [['unique_play_id', "nfl_id", "frame_id", "x_clean", "y_clean"]]
        .rename(columns={
            "frame_id": "frame_id_prev",
            "x_clean": "x_prev_input",
            "y_clean": "y_prev_input",
        })
    )

    out = output_df.copy()
    out = out.sort_values(['unique_play_id', "nfl_id", "frame_id"])

    # Previous position within output sequence
    out["x_prev_out"] = out.groupby(['unique_play_id', "nfl_id"])["x_clean"].shift(1)
    out["y_prev_out"] = out.groupby(['unique_play_id', "nfl_id"])["y_clean"].shift(1)

    # Merge in last input positions to fill first-output-frame prevs
    out = out.merge(last_input, on=['unique_play_id', "nfl_id"], how="left")

    # Choose prev from output if available, else from last input
    x_prev = out["x_prev_out"].fillna(out["x_prev_input"])
    y_prev = out["y_prev_out"].fillna(out["y_prev_input"])

    # Velocity components
    out["v_x"] = (out["x_clean"] - x_prev) / framerate
    out["v_y"] = (out["y_clean"] - y_prev) / framerate

    # Remove helper columns
    out = out.drop(columns=["x_prev_out", "y_prev_out", "frame_id_prev", "x_prev_input", "y_prev_input"])

    return out

def clean_data(input_df, output_df):

  """
  Flip tracking data so that all plays run from left to right. The new x, y, s, a, dis, o, and dir data
  will be stored in new columns with the suffix "_clean" even if the variables do not change from their original value.

  :param df: the aggregate dataframe created using the aggregate_data() method

  :return df: the aggregate dataframe with the new columns such that all plays run left to right
  """

  # Fix orientation and rotation so that 0º is facing left-to-right
  input_df["o_clean"] = (-(input_df["o"] - 90)) % 360
  input_df["dir_clean"] = (-(input_df["dir"] - 90)) % 360  

  # Set x,y locations so that a play always moves left-to-right
  input_df["x_clean"] = np.where(
      input_df["play_direction"] == "left",
      120 - input_df["x"],
      input_df[
          "x"
      ],  # 120 because the endzones (10 yds each) are included in the ["x"] values
  )

  input_df["ball_land_x_clean"] = np.where(
      input_df["play_direction"] == "left",
      120 - input_df["ball_land_x"],
      input_df[
          "ball_land_x"
      ],  # 120 because the endzones (10 yds each) are included in the ["x"] values
  )
  input_df["y_clean"] = input_df["y"]
  input_df["ball_land_y_clean"] = input_df["ball_land_y"]
  input_df["s_clean"] = input_df["s"]
  input_df["a_clean"] = input_df["a"]

  input_df["o_clean"] = np.where(
      input_df["play_direction"] == "left", 180 - input_df["o_clean"], input_df["o_clean"]
  )

  input_df["o_clean"] = (input_df["o_clean"] + 360) % 360  # remove negative angles

  input_df["dir_clean"] = np.where(
      input_df["play_direction"] == "left", 180 - input_df["dir_clean"], input_df["dir_clean"]
  )

  input_df["dir_clean"] = (input_df["dir_clean"] + 360) % 360  # remove negative angles

  # Calculate velocity components of input dataframe
  input_df["dir_radians"] = np.radians(input_df["dir_clean"])
  input_df["v_x"] = input_df["s_clean"] * np.cos(input_df["dir_radians"])
  input_df["v_y"] = input_df["s_clean"] * np.sin(input_df["dir_radians"])

  # Handle post-throw changes in x/y axis.
  # Play direction/player side is not given in output_df, so infer it from input_df.
  # Build a mapping from play_id to play_direction using input_df
  play_direction_map = (
        input_df[["unique_play_id", "play_direction"]]
        .drop_duplicates(subset=["unique_play_id"])
        .set_index("unique_play_id")["play_direction"]
        )
  player_side_map = (
      input_df[["nfl_id", "player_side"]]
      .drop_duplicates(subset=["nfl_id"])
      .set_index("nfl_id")["player_side"]
      )
  ball_land_map_x = (
      input_df[["unique_play_id", "ball_land_x_clean"]]
      .drop_duplicates(subset=["unique_play_id"])
      .set_index("unique_play_id")["ball_land_x_clean"]
      )
  ball_land_map_y = (
      input_df[["unique_play_id", "ball_land_y_clean"]]
      .drop_duplicates(subset=["unique_play_id"])
      .set_index("unique_play_id")["ball_land_y_clean"]
      )
  pass_length_map = (
      input_df[["unique_play_id", "pass_length"]]
      .drop_duplicates(subset=["unique_play_id"])
      .set_index("unique_play_id")["pass_length"]
      )

  # Map play_direction and player_side into output_df
  output_df["play_direction"] = output_df["unique_play_id"].map(play_direction_map)
  output_df["player_side"] = output_df["nfl_id"].map(player_side_map)
  output_df["ball_land_x_clean"] = output_df["unique_play_id"].map(ball_land_map_x)
  output_df["ball_land_y_clean"] = output_df["unique_play_id"].map(ball_land_map_y)
  output_df['pass_length'] = output_df['unique_play_id'].map(pass_length_map)

  # Compute cleaned coordinates for post-throw data using inferred play_direction
  output_df["x_clean"] = np.where(
      output_df["play_direction"] == "left",
      120 - output_df["x"],
      output_df["x"],
  )
  output_df["y_clean"] = output_df["y"]
  
  # Quantify per-role data (used in building target/defender pairs)
  input_df["player_targeted"] = (
      input_df["player_role"] == "Targeted Receiver"
  )
  input_df["is_defender"] = (
      input_df["player_role"] == "Defensive Coverage"
  )
  input_df["is_passer"] = (
      input_df["player_role"] == "Passer"
  )

  # Avoid double-counting for every frame
  input_df_unique = input_df[['unique_play_id', 'nfl_id', 'player_position', 'player_role']].drop_duplicates()
  # Allow 'player_position' field in output_df
  output_df = output_df.merge(
    input_df_unique[['unique_play_id', 'nfl_id', 'player_position', 'player_role']], 
    on=['unique_play_id', 'nfl_id'], 
    how='left'
  )
  # Add velocity to output 
  output_df = add_velocity_to_output(input_df, output_df)
  
  # Compute vector from player location to ball landing position
  input_df['ball_vec_x_clean'] = input_df['ball_land_x_clean'] - input_df['x_clean']
  input_df['ball_vec_y_clean'] = input_df['ball_land_y_clean'] - input_df['y_clean']
  output_df['ball_vec_x_clean'] = output_df['ball_land_x_clean'] - output_df['x_clean']
  output_df['ball_vec_y_clean'] = output_df['ball_land_y_clean'] - output_df['y_clean']

  # Assign "helper" field to input and output dfs
  input_df, output_df = assign_helper_field(input_df, output_df)

  return input_df, output_df

def load_data(week):
    """
    Returns
    -------
    input_df : pandas.DataFrame
        Concatenated input & supplementary data for all weeks.
    output_df : pandas.DataFrame
        Concatenated output & supplementary data for all weeks.
    """

    start_time = time.time()
    print("[INFO]: Reading input data from week " + str(week))
    input_file = f"./../data/train/input_2023_w{week:02d}.csv"
    input_df = pd.read_csv(input_file)
    # Read output
    print("[INFO]: Reading output data from week " + str(week))
    output_file = f"./../data/train/output_2023_w{week:02d}.csv"
    output_df = pd.read_csv(output_file)
    # Read supplementary
    print("[INFO]: Reading supplementary data")
    supplementary = pd.read_csv("./../data/supplementary_data.csv")

    # Create unique per-game id
    input_df["unique_play_id"] = (
    input_df['game_id'].astype(str) + "_" +
    input_df['play_id'].astype(str))

    output_df["unique_play_id"] = (
    output_df['game_id'].astype(str) + "_" +
    output_df['play_id'].astype(str))

    supplementary["unique_play_id"] = (
    supplementary['game_id'].astype(str) + "_" +
    supplementary['play_id'].astype(str))

    # Join with input/output dataframes based on both game_id and play_id 
    input_df = input_df.merge(
        supplementary,
        on=["unique_play_id"],
        how="left",
    )

    output_df = output_df.merge(
        supplementary,
        on=["unique_play_id"],
        how="left",
    )
  
    end_time = time.time()
    input_df, output_df = clean_data(input_df, output_df)
    print("[INFO]: Loaded data in " + str(end_time - start_time) + " seconds")
    
    return input_df, output_df