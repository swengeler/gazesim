import os
import re
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from src.data.utils import iterate_directories, parse_run_info, filter_by_screen_ts

"""
data_root = os.getenv("GAZESIM_ROOT")

same = []
total_frames = 0
for run_dir in iterate_directories(data_root, track_names=["flat", "wave"]):
    # load the video and find out the number of frames
    video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
    w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))

    # load the screen_timestamps dataframe and compare #frames and #columns
    df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))

    same.append(int(num_frames) == len(df_screen.index))
    total_frames += int(num_frames)
    print("{}\n#frames: {}\n#rows: {}\n".format(run_dir, int(num_frames), len(df_screen.index)))

same = np.array(same)
print("#same: {}/{}".format(same.sum(), len(same)))
print("total #frames: {}".format(total_frames))
"""


def add_frames(data, df_screen_ts):
    data["frame"].extend(list(df_screen_ts["frame"].values))
    return data


def add_subject(data, run_dir, num_frames):
    subject_number = parse_run_info(run_dir)["subject"]
    if subject_number is None:
        raise ValueError("Subject number could not be read for '{}'.".format(run_dir))

    data["subject"].extend([subject_number] * num_frames)
    return data


def add_run(data, run_dir, num_frames):
    run_number = parse_run_info(run_dir)["run"]
    if run_number is None:
        raise ValueError("Run number could not be read for '{}'.".format(run_dir))

    data["run"].extend([run_number] * num_frames)
    return data


def add_track_name(data, run_dir, num_frames):
    track_name = parse_run_info(run_dir)["track_name"]
    if track_name is None:
        raise ValueError("Track name could not be read for '{}'.".format(run_dir))

    data["track_name"].extend([track_name] * num_frames)
    return data


def add_rgb_available(data, run_dir, num_frames):
    video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
    w, h = (video_capture.get(i) for i in range(3, 5))

    if w == 800 and h == 600:
        rgb_available = [1] * num_frames
    else:
        rgb_available = [0] * num_frames

    data["rgb_available"].extend(rgb_available)
    return data


def add_gaze_measurement_available(data, df_screen_ts, df_gaze):
    # TODO: eventually clear this up
    # the main question is whether we e.g. want to include a half-frame "buffer" at the start and end that
    # would count towards that frame... probably, but then the question is whether how the GT is actually
    # computed (e.g. mean/median/only one measurement) changes anything about that
    # => SOLUTION FOR NOW: change name to "measurement", since that is certainly accurate
    # => might then just have to add more columns for any "weird" GT computations that deviates if there are any
    # => seem like it would be especially relevant for the control GT (maybe time-lag for GT computation there)

    df_screen_ts, df_gaze = filter_by_screen_ts(df_screen_ts, df_gaze, buffer=((1 / 60) / 2))
    gaze_measurement_available = df_screen_ts["frame"].isin(df_gaze["frame"]).astype(int)

    data["gaze_measurement_available"].extend(gaze_measurement_available)
    return data


def add_control_measurement_available(data, df_screen_ts, df_drone):
    df_screen_ts, df_drone = filter_by_screen_ts(df_screen_ts, df_drone, buffer=((1 / 60) / 2))
    gaze_measurement_available = df_screen_ts["frame"].isin(df_drone["frame"]).astype(int)

    data["control_measurement_available"].extend(gaze_measurement_available)
    return data


def add_lap_info(data, df_screen_ts, df_lap_info, df_trajectory):
    df_temp = df_screen_ts.copy()
    df_temp["valid_lap"] = -1
    df_temp["lap_index"] = -1
    df_temp["expected_trajectory"] = -1

    for _, row in df_lap_info.iterrows():
        ts_index = df_screen_ts["ts"].between(row["ts_start"], row["ts_end"])
        df_temp.loc[ts_index, "valid_lap"] = row["is_valid"]
        df_temp.loc[ts_index, "lap_index"] = row["lap"]

    if df_trajectory is not None:
        for _, row in df_trajectory.iterrows():
            lap_start = df_lap_info["ts_start"].loc[df_lap_info["lap"] == row["lap"]].values[0]
            lap_end = df_lap_info["ts_end"].loc[df_lap_info["lap"] == row["lap"]].values[0]
            if row["expected_trajectory"] == 1:
                df_temp.loc[df_screen_ts["ts"].between(lap_start, lap_end), "expected_trajectory"] = 1
            else:
                df_temp.loc[df_screen_ts["ts"].between(lap_start, lap_end), "expected_trajectory"] = 0

    data["valid_lap"].extend(df_temp["valid_lap"].values)
    data["lap_index"].extend(df_temp["lap_index"].values)
    data["expected_trajectory"].extend(df_temp["expected_trajectory"].values)
    return data


def add_turn_info(data, df_screen_ts, df_lap_info):
    df_temp = df_screen_ts.copy()
    df_temp["left_turn"] = 0
    df_temp["right_turn"] = 0
    df_temp["left_half"] = 0
    df_temp["right_half"] = 0

    # TODO: for this to work properly, either need to implement some fancy stuff to
    #  "wrap around" for the right half or redefine laps to start and end in the middle
    gates_list = [(3, 4, 5), (8, 9, 10), (2, 3, 4, 5, 6), (7, 8, 9, 10, 1)][:2]
    label_list = ["left_turn", "right_turn", "left_half", "right_half"][:2]
    for _, row in df_lap_info.iterrows():
        gate_id = [int(i) for i in row["gate_id"][1:-1].strip().split()]
        gate_ts = [float(i) for i in row["gate_timestamps"][1:-1].strip().split()]

        for turn_gates, turn_label in zip(gates_list, label_list):
            ts_low = None
            ts_high = None
            for gid, gts in zip(gate_id, gate_ts):
                if ts_low is None and turn_gates[0] <= gid <= turn_gates[1]:
                    ts_low = gts
                if turn_gates[1] <= gid <= turn_gates[2]:
                    ts_high = gts
            if ts_low is not None and ts_high is not None:
                df_temp.loc[df_screen_ts["ts"].between(ts_low, ts_high), turn_label] = 1

    data["left_turn"].extend(df_temp["left_turn"].values)
    data["right_turn"].extend(df_temp["right_turn"].values)
    data["left_half"].extend(df_temp["left_half"].values)
    data["right_half"].extend(df_temp["right_half"].values)
    return data


def main(args):
    # TODO: basically use the same loop as above and record all the stuff for each frame...
    #  1. (tight/loose) left/right turn?
    #  2. gaze_gt_available (based on gaze_on_surface.csv)
    #  3. control_gt_available (based on drone.csv)
    #  4. rgb_available (based on whether screen.mp4 exists and is right size)
    #  5. valid_lap
    #  6. expected_trajectory
    #  7. lap index (probably)
    #  8. obviously subject and run...
    #  9. track name (index both flat and wave tracks)
    #  => taking the last two, should "build" path from that info instead of using rel_run_path

    data = {
        "frame": [],
        "subject": [],
        "run": [],
        "track_name": [],
        "rgb_available": [],
        "gaze_measurement_available": [],
        "control_measurement_available": [],
        "valid_lap": [],
        "lap_index": [],
        "expected_trajectory": [],
        "left_turn": [],
        "right_turn": [],
        "left_half": [],
        "right_half": []
    }

    for run_dir in tqdm(iterate_directories(args.data_root)):
        # load relevant dataframes
        df_screen_ts = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))
        df_gaze = pd.read_csv(os.path.join(run_dir, "gaze_on_surface.csv"))
        df_drone = pd.read_csv(os.path.join(run_dir, "drone.csv"))
        df_lap_info = pd.read_csv(os.path.join(run_dir, "laptimes.csv"))
        df_trajectory = None
        if os.path.exists(os.path.join(run_dir, "expected_trajectory.csv")):
            # TODO: need to add this info for "wave" tracks as well
            #  - would maybe be good to have "horizontal" and "vertical" view of trajectory and judge that?
            pd.read_csv(os.path.join(run_dir, "expected_trajectory.csv"))

        num_frames = len(df_screen_ts.index)

        # add frames, subject number, run number and track name
        add_frames(data, df_screen_ts)
        add_subject(data, run_dir, num_frames)
        add_run(data, run_dir, num_frames)
        add_track_name(data, run_dir, num_frames)

        # add information about whether the RGB video and the ground-truth types are available
        add_rgb_available(data, run_dir, num_frames)
        add_gaze_measurement_available(data, df_screen_ts, df_gaze)
        add_control_measurement_available(data, df_screen_ts, df_drone)

        # add information about the lap/trajectory as a whole
        add_lap_info(data, df_screen_ts, df_lap_info, df_trajectory)

        # add information about curve in the horizontal (or whatever you want to call it...)
        add_turn_info(data, df_screen_ts, df_lap_info)

        # TODO: probably add info about e.g. upward/downward trajectory for "wave" track

    for key in data:
        print("{}: {}".format(key, len(data[key])))

    df_data = pd.DataFrame(data)
    df_data.to_csv(os.path.join(args.data_root, "index.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-e", "--exclude", type=str, nargs="+", default=[],
                        help="Properties to exclude from indexing.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    main(arguments)
