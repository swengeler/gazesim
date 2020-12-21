import os
import pandas as pd
import cv2

from tqdm import tqdm
from src.data.utils import run_info_to_path


COLUMN_DICT = {
    "ts": "time-since-start [s]",
    "PositionX": "position_x [m]",
    "PositionY": "position_y [m]",
    "PositionZ": "position_z [m]",
    "VelocityX": "velocity_x [m/s]",
    "VelocityY": "velocity_y [m/s]",
    "VelocityZ": "velocity_z [m/s]",
    "AccX": "acceleration_x [m/s/s]",
    "AccY": "acceleration_y [m/s/s]",
    "AccZ": "acceleration_z [m/s/s]",
    "rot_w_quat": "rotation_w [quaternion]",
    "rot_x_quat": "rotation_x [quaternion]",
    "rot_y_quat": "rotation_y [quaternion]",
    "rot_z_quat": "rotation_z [quaternion]",
}


def extract_lap_trajectory(args):
    # load the correct drone.csv and laptimes.csv
    run_dir = os.path.join(args.data_root, run_info_to_path(args.subject, args.run, args.track_name))
    inpath_drone = os.path.join(run_dir, "drone.csv")
    inpath_laps = os.path.join(run_dir, "laptimes.csv")
    inpath_exp_traj = os.path.join(run_dir, "expected_trajectory.csv")

    df_drone = pd.read_csv(inpath_drone)
    df_laps = pd.read_csv(inpath_laps)

    # select the data from the drone dataframe by the timestamps for the lap index in laptimes
    time_stamps = df_laps.loc[df_laps["lap"] == args.lap_index, ["ts_start", "ts_end"]].values[0]
    df_traj = df_drone[df_drone["ts"].between(time_stamps[0], time_stamps[1])]

    # select columns and use new column headers
    df_traj = df_traj[[co for co in COLUMN_DICT]]
    df_traj = df_traj.rename(COLUMN_DICT, axis=1)

    # adjust time stamps to start at 0
    df_traj["time-since-start [s]"] = df_traj["time-since-start [s]"] - df_traj["time-since-start [s]"].min()

    # save the data to specified path
    file_name = "trajectory_s{:03d}_r{:02d}_{}_li{:02d}.csv".format(args.subject, args.run, args.track_name, args.lap_index)
    df_traj.to_csv(os.path.join(args.save_path, file_name), index=False)


def extract_lap_video(args):
    # TODO:
    run_dir = os.path.join(args.data_root, run_info_to_path(args.subject, args.run, args.track_name))
    inpath_video = os.path.join(run_dir, "screen.mp4")
    inpath_ts = os.path.join(run_dir, "screen_timestamps.csv")
    inpath_laps = os.path.join(run_dir, "laptimes.csv")
    file_name = "video_s{:03d}_r{:02d}_{}_li{:02d}.mp4".format(args.subject, args.run, args.track_name, args.lap_index)

    # load screen_timestamps.csv and laptimes.csv
    df_ts = pd.read_csv(inpath_ts)
    df_laps = pd.read_csv(inpath_laps)

    # select only those frames that are within the specified lap's timestamps
    time_stamps = df_laps.loc[df_laps["lap"] == args.lap_index, ["ts_start", "ts_end"]].values[0]
    df_ts = df_ts[df_ts["ts"].between(time_stamps[0], time_stamps[1])]

    # read the screen video for those frames and save it in a similar way to the trajectories above
    video_capture = cv2.VideoCapture(inpath_video)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))

    video_writer = cv2.VideoWriter(
        os.path.join(args.save_path, file_name),
        int(fourcc),
        fps,
        (int(w), int(h)),
        True
    )

    for _, row in tqdm(df_ts.iterrows(), total=len(df_ts.index)):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
        frame = video_capture.read()[1]
        video_writer.write(frame)

    video_writer.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-dr", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-md", "--mode", type=str, default="trajectory", choices=["trajectory", "video"])
    parser.add_argument("-s", "--subject", type=int)
    parser.add_argument("-r", "--run", type=int)
    parser.add_argument("-li", "--lap_index", type=int)
    parser.add_argument("-tn", "--track_name", type=str, default="flat")
    parser.add_argument("-sp", "--save_path", type=str, default=os.path.join(os.getenv("HOME"), "Downloads"))

    arguments = parser.parse_args()

    if arguments.mode == "trajectory":
        extract_lap_trajectory(arguments)
    elif arguments.mode == "video":
        extract_lap_video(arguments)
