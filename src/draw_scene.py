"""
Animate a play on a 2d version of an NFL field.
Usage:
python3 ./draw_scene.py play_id_to_visualize week_num_of_play
"""
from IPython.display import HTML
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import pandas as pd
import sys
import load_data
import manual_test
from transformer_model import TransformerModel

animation_fps = 10

def create_football_field(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6.33))
    else:
        fig = None  # When passing ax, no new figure is created

    # Field outline and endzones
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='lightgreen', zorder=0)
    ax.add_patch(rect)

    # Endzones
    ax.add_patch(patches.Rectangle((0, 0), 10, 53.3, linewidth=0.1, facecolor='blue', alpha=0.2, zorder=0))
    ax.add_patch(patches.Rectangle((110, 0), 10, 53.3, linewidth=0.1, facecolor='blue', alpha=0.2, zorder=0))

    # Field lines every 10 yards
    for x in range(10, 120, 10):
        ax.plot([x, x], [0, 53.3], color='white', zorder=1)

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.axis('off')

    return fig, ax

def animate_play(prethrow_data, postthrow_data, predict, actual):
    """Animates the play on a 2D football field, as well as the model prediction
    and actual output values. The animation is output to the play_sample.gif file."""
    print("[INFO]: Animating play")
    # Combine pre- and post-throw data so model runs continuosuly
    prethrow_frame_ids = sorted(prethrow_data['frame_id'].unique())
    postthrow_frame_ids = sorted(postthrow_data['frame_id'].unique())
    postthrow_frame_ids = [frame + len(prethrow_frame_ids) for frame in postthrow_frame_ids]
    
    frame_ids = prethrow_frame_ids + postthrow_frame_ids
    fig, ax = create_football_field()

    # Save a separate snapshot of the training frame for that play
    snapshot_frame_id = prethrow_frame_ids[-2]
    
    # Handle legend creation
    legend_handles = [
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor='blue', markeredgecolor='blue',
                      markersize=8, label='Offensive player (non-targeted)'),
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor='purple', markeredgecolor='purple',
                      markersize=8, label='Targeted offensive receiver'),
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor='white', markeredgecolor='black',
                      markersize=8, label='Defensive player (non-helper)'),
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor='gold', markeredgecolor='gold',
                      markersize=8, label='Identified help defender'),
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor='brown', markeredgecolor='brown',
                      markersize=8, label='Incomplete ball: landing location'),
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor='green', markeredgecolor='green',
                      markersize=8, label='Completed ball: landing location'),
        mlines.Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor='red', markeredgecolor='red',
                      markersize=8, label='Intercepted ball: landing location'),
    ]
    # Adjust to make room for legend
    fig.subplots_adjust(right=0.8)

    # Legend on right side
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        framealpha=0.9,
        fontsize=8
    )

    def update(frame_id):

        ax.clear()
        create_football_field(ax)
        if frame_id <= len(prethrow_frame_ids):
            play_data = prethrow_data
        else:
            play_data = postthrow_data

        frame_id_clean = frame_id if frame_id <= len(prethrow_frame_ids) else frame_id - len(prethrow_frame_ids)
        frame_data = play_data[play_data['frame_id'] == frame_id_clean]
        # Separate into offensive and defensive players
        offense_data = frame_data[((frame_data['player_side'] == 'Offense') & 
                                  (frame_data['player_role'] != 'Targeted Receiver'))]
        defense_data = frame_data[((frame_data['player_side'] == 'Defense') &
                                   (frame_data['helper'] == 0))]
        # Indicate targeted receiver
        targeted_receiver_data = frame_data[frame_data['player_role'] == 'Targeted Receiver']
        # Indicate help defender
        helper_data = frame_data[frame_data['helper'] == 1]

        ax.scatter(offense_data['x_clean'], offense_data['y_clean'], color='blue', s=50, label='Offense')
        ax.scatter(defense_data['x_clean'], defense_data['y_clean'], color='white', s=50, label='Defense')
        ax.scatter(targeted_receiver_data['x_clean'], targeted_receiver_data['y_clean'], color='purple', s=50, label='Targeted Receiver')
        ax.scatter(helper_data['x_clean'], helper_data['y_clean'], color='gold', s=50, label='Helper')
        # Arbitrarily picking row 0, doesn't matter
        pass_result = prethrow_data["pass_result"].iloc[0]
        if pass_result == "C":
            ax.scatter(targeted_receiver_data['ball_land_x_clean'], targeted_receiver_data['ball_land_y_clean'], color='green', s=50, label='ball landing location')
        elif pass_result == "IN":
            ax.scatter(targeted_receiver_data['ball_land_x_clean'], targeted_receiver_data['ball_land_y_clean'], color='red', s=50, label='ball landing location')
        elif pass_result == "I":
            ax.scatter(targeted_receiver_data['ball_land_x_clean'], targeted_receiver_data['ball_land_y_clean'], color='brown', s=50, label='ball landing location')

        # Arrows represent velocity, not direction player is facing.
        # Orientation arrows face the same direction as velocity anyway so we can't
        # get a true "facing direction" arrow
        for i in range(len(frame_data)):
            dx = frame_data.iloc[i]['v_x']
            dy = frame_data.iloc[i]['v_y']
            ax.arrow(frame_data.iloc[i]['x_clean'], frame_data.iloc[i]['y_clean'], dx, dy,
                    head_width=0.25, head_length=0.5, fc='red', ec='red')
        ax.set_title(f"Unique play id: {frame_data['unique_play_id'].values[0]}. Coverage: {frame_data['team_coverage_type'].values[0]}. Frame: {frame_id}\nPrediction: {predict}. Actual: {actual}")
        if (frame_id == snapshot_frame_id):
            ax.set_title(f"Unique play id: {frame_data['unique_play_id'].values[0]}. Coverage: {frame_data['team_coverage_type'].values[0]}. Frame: {frame_id}. Frames till catch: {len(postthrow_frame_ids) + 2}\nPrediction: {predict}. Actual: {actual}")
            fig.savefig("play_snapshot.png", dpi=200, bbox_inches="tight")
            print(f"[INFO]: Saved snapshot at prethrow frame {snapshot_frame_id}")


    ani = animation.FuncAnimation(fig, update, frames=frame_ids, repeat=False)
    ani.save("play_sample.gif", writer='pillow', fps=animation_fps)

    return HTML(ani.to_jshtml())

def load_play_data(gameid, playid, week):
    print("[INFO]: Loading data")
    input_df, output_df = load_data.load_data(week)
    input_unique_id = (str(gameid) + "_" + str(playid))
    output_unique_id = (str(gameid) + "_" + str(playid))
    input_play = input_df[(input_df["unique_play_id"] == input_unique_id)]
    output_play = output_df[(output_df["unique_play_id"] == output_unique_id)]
    if not input_play.empty and not output_play.empty:
        return input_play, output_play
    raise Exception("play_id not found for provided game_id and week_num")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("Usage: python3 ./draw_scene.py game_id play_id week_num")
    game_id = int(sys.argv[1])
    play_id = int(sys.argv[2])
    week_num = int(sys.argv[3])
    input_play, output_play = load_play_data(game_id, play_id, week_num)
    predict, actual = manual_test.run_manual_test(input_play, output_play)
    animate_play(input_play, output_play, predict, actual)