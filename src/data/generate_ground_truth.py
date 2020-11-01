import os
import numpy as np
import pandas as pd
import cv2

from typing import Type
from tqdm import tqdm
from time import time
from src.data.utils import iterate_directories, generate_gaussian_heatmap, filter_by_screen_ts, parse_run_info


class GroundTruthGenerator:

    def __init__(self, config):
        self.run_dir_list = iterate_directories(config["data_root"], track_names=config["track_name"])

    def get_gt_info(self, run_dir, subject, run):
        raise NotImplementedError()

    def compute_gt(self, run_dir):
        raise NotImplementedError()

    def generate(self):
        for rd in self.run_dir_list:
            self.compute_gt(rd)


class MovingWindowFrameMeanGT(GroundTruthGenerator):

    NAME = "moving_window_frame_mean_gt"

    def __init__(self, config):
        super().__init__(config)

        # make sure the input is correct
        assert config["mw_size"] > 0 and config["mw_size"] % 2 == 1

        self._half_window_size = int((config["mw_size"] - 1) / 2)

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        gaze_gt_path = os.path.join(index_dir, "gaze_gt.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(gaze_gt_path):
            df_gaze_gt = pd.read_csv(gaze_gt_path)
        else:
            df_gaze_gt = df_frame_index.copy()
            df_gaze_gt = df_gaze_gt[["frame", "subject", "run"]]

        if self.__class__.NAME not in df_gaze_gt.columns:
            df_gaze_gt[self.__class__.NAME] = -1

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_gaze_gt["subject"] == subject) & (df_gaze_gt["run"] == run)

        return df_gaze_gt, gaze_gt_path, match_index

    def compute_gt(self, run_dir):
        start = time()

        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_gaze_gt, gaze_gt_path, match_index = self.get_gt_info(run_dir, subject, run)

        # initiate video capture and writer
        video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))

        if not (w == 800 and h == 600):
            print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
            return

        if not int(num_frames) == match_index.sum():
            print("WARNING: Number of frames in video and registered in main index is different for directory '{}'.".format(run_dir))
            return

        # writer is only initialised after making sure that everything else works
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, f"{self.__class__.NAME}.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

        # load data frames with the timestamps and positions for gaze and the frames/timestamps for the video
        df_gaze = pd.read_csv(os.path.join(run_dir, "gaze_on_surface.csv"))
        df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_gaze = df_gaze[["ts", "frame", "norm_x_su", "norm_y_su"]]
        df_gaze.columns = ["ts", "frame", "x", "y"]
        df_gaze["x"] = df_gaze["x"] * 800
        df_gaze["y"] = (1.0 - df_gaze["y"]) * 600

        # filter by screen timestamps
        df_screen, df_gaze = filter_by_screen_ts(df_screen, df_gaze)

        # since the measurements are close together and to reduce computational load,
        # compute the mean measurement for each frame
        # TODO: think about whether this is a good way to deal with this issue
        df_gaze = df_gaze[["frame", "x", "y"]]
        df_gaze = df_gaze.groupby("frame").mean()
        df_gaze["frame"] = df_gaze.index

        # create new dataframe to write frame-level information into
        df_gaze_gt.loc[match_index, self.__class__.NAME] = df_screen["frame"].isin(df_gaze["frame"]).astype(int).values

        # loop through all frames and compute the ground truth (where possible)
        for frame_idx in tqdm(df_screen["frame"], disable=False):
            if frame_idx >= num_frames:
                print("Number of frames in CSV file exceeds number of frames in video!")
                break

            # compute the range of frames to use for the ground truth
            frame_low = frame_idx - self._half_window_size
            frame_high = frame_idx + self._half_window_size

            # create the heatmap
            current_frame_data = df_gaze[df_gaze["frame"].between(frame_low, frame_high)]
            current_mu = current_frame_data[["x", "y"]].values
            heatmap = generate_gaussian_heatmap(mu=current_mu, down_scale_factor=10)

            if heatmap.max() > 0.0:
                heatmap /= heatmap.max()

            heatmap = (heatmap * 255).astype("uint8")
            heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)

            # save the resulting frame
            video_writer.write(heatmap)

        video_writer.release()

        # save gaze gt index to CSV with updated data
        df_gaze_gt.to_csv(gaze_gt_path, index=False)

        print("Saved moving window ground-truth for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class DroneControlFrameMeanGT(GroundTruthGenerator):

    NAME = "drone_control_frame_mean_gt"

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        control_gt_path = os.path.join(index_dir, "control_gt.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(control_gt_path):
            df_control_gt = pd.read_csv(control_gt_path)
        else:
            df_control_gt = df_frame_index.copy()
            df_control_gt = df_control_gt[["frame", "subject", "run"]]

        if self.__class__.NAME not in df_control_gt.columns:
            df_control_gt[self.__class__.NAME] = -1
            df_control_gt["Thrust"] = np.nan
            df_control_gt["Roll"] = np.nan
            df_control_gt["Pitch"] = np.nan
            df_control_gt["Yaw"] = np.nan

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_control_gt["subject"] == subject) & (df_control_gt["run"] == run)

        return df_control_gt, control_gt_path, match_index

    def compute_gt(self, run_dir):
        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_control_gt, control_gt_path, match_index = self.get_gt_info(run_dir, subject, run)

        # define paths
        df_drone_path = os.path.join(run_dir, "drone.csv")
        df_screen_path = os.path.join(run_dir, "screen_timestamps.csv")
        df_screen_info_path = os.path.join(run_dir, "screen_frame_info.csv")

        # load dataframes
        df_drone = pd.read_csv(df_drone_path)
        df_screen = pd.read_csv(df_screen_path)

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_drone = df_drone[["ts", "Throttle", "Roll", "Pitch", "Yaw"]]
        df_drone["frame"] = -1

        # filter by screen timestamps
        df_screen, df_drone = filter_by_screen_ts(df_screen, df_drone)

        # compute the mean measurement for each frame
        df_drone = df_drone[["frame", "Throttle", "Roll", "Pitch", "Yaw"]]
        df_drone = df_drone.groupby("frame").mean()
        df_drone["frame"] = df_drone.index
        df_drone = df_drone.reset_index(drop=True)
        df_drone = df_drone[["frame", "Throttle", "Roll", "Pitch", "Yaw"]]

        # add information about control GT being available to frame-wise screen info
        df_control_gt.loc[match_index, self.__class__.NAME] = df_screen["frame"].isin(df_drone["frame"]).astype(int).values
        for _, row in tqdm(df_drone.iterrows(), disable=False, total=len(df_drone.index)):
            df_control_gt.loc[match_index & (df_control_gt["frame"] == row["frame"]),
                              ["Throttle", "Roll", "Pitch", "Yaw"]] = row[["Throttle", "Roll", "Pitch", "Yaw"]].values

        # save control gt to CSV with updated data
        df_control_gt.to_csv(control_gt_path, index=False)


def resolve_gt_class(ground_truth_type: str) -> Type[GroundTruthGenerator]:
    if ground_truth_type == "moving_window_frame_mean_gt":
        return MovingWindowFrameMeanGT
    elif ground_truth_type == "drone_control_frame_mean_gt":
        return DroneControlFrameMeanGT
    return GroundTruthGenerator


def main(args):
    config = vars(args)

    generator = resolve_gt_class(config["ground_truth_type"])(config)
    generator.generate()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser()

    # general arguments
    PARSER.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    PARSER.add_argument("-tn", "--track_name", type=str, nargs="+", default=["flat", "wave"], choices=["flat", "wave"],
                        help="The method to use to compute the ground-truth.")
    PARSER.add_argument("-gtt", "--ground_truth_type", type=str, default="moving_window_frame_mean_gt",
                        choices=["moving_window_frame_mean_gt", "drone_control_frame_mean_gt"],
                        help="The method to use to compute the ground-truth.")

    # arguments only used for moving_window
    PARSER.add_argument("--mw_size", type=int, default=25,
                        help="Size of the temporal window in frames from which the "
                             "ground-truth for the current frame should be computed.")

    # parse the arguments
    ARGS = PARSER.parse_args()

    # generate the GT
    main(ARGS)

