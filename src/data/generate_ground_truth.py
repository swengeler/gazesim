import os
import numpy as np
import pandas as pd
import cv2

from typing import Type
from tqdm import tqdm
from time import time
from src.data.utils import iterate_directories, generate_gaussian_heatmap, filter_by_screen_ts, parse_run_info, pair


class GroundTruthGenerator:

    def __init__(self, config):
        self.run_dir_list = iterate_directories(config["data_root"], track_names=config["track_name"])
        if config["directory_index"] is not None:
            self.run_dir_list = self.run_dir_list[int(config["directory_index"][0]):config["directory_index"][1]]
        """
        for r_idx, r in enumerate(self.run_dir_list):
            print(r_idx, ":", r)
        exit()
        """

    def get_gt_info(self, run_dir, subject, run):
        raise NotImplementedError()

    def compute_gt(self, run_dir):
        raise NotImplementedError()

    def generate(self):
        for rd in tqdm(self.run_dir_list, disable=True):
            self.compute_gt(rd)


class MovingWindowFrameMeanGT(GroundTruthGenerator):

    NAME = "moving_window_frame_mean_gt"

    def __init__(self, config):
        super().__init__(config)

        # make sure the input is correct
        assert config["mw_size"] > 0 and config["mw_size"] % 2 == 1

        self._half_window_size = int((config["mw_size"] - 1) / 2)
        self._skip_existing = config["skip_existing"]

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        gaze_gt_path = os.path.join(index_dir, "test_gaze_gt.csv")

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

        # save gaze gt index to CSV with updated data
        df_gaze_gt.to_csv(gaze_gt_path, index=False)

        if self._skip_existing and os.path.exists(os.path.join(run_dir, f"{self.__class__.NAME}.mp4")):
            print("INFO: Video already exists for '{}'.".format(run_dir))
            return

        # writer is only initialised after making sure that everything else works
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, f"{self.__class__.NAME}.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

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

        print("Saved moving window ground-truth for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class RandomGazeGT(GroundTruthGenerator):

    NAME = "random_gaze_gt"

    def __init__(self, config):
        super().__init__(config)

        # make sure the input is correct
        assert config["mw_size"] > 0 and config["mw_size"] % 2 == 1

        self._half_window_size = int((config["mw_size"] - 1) / 2)
        self._skip_existing = config["skip_existing"]

        np.random.seed(config["random_seed"])

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        gaze_gt_path = os.path.join(index_dir, "test_gaze_gt.csv")

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
            print("WARNING: Number of frames in video and registered in main index "
                  "is different for directory '{}'.".format(run_dir))
            return

        # moved this here to be able to exit as early as possible if video already exists
        # this "ground-truth" type is available for all frames
        df_gaze_gt.loc[match_index, self.__class__.NAME] = 1

        # save gaze gt index to CSV with updated data
        df_gaze_gt.to_csv(gaze_gt_path, index=False)

        if self._skip_existing and os.path.exists(os.path.join(run_dir, f"{self.__class__.NAME}.mp4")):
            print("INFO: Video already exists for '{}'.".format(run_dir))
            return

        # load data frames with the timestamps and positions for gaze and the frames/timestamps for the video
        df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))
        df_screen["x"] = w / 2
        df_screen["y"] = h / 2

        # initial position
        next_pos = np.random.multivariate_normal(np.array([h / 2, w / 2]), np.array([[h, 0.0], [0.0, w]]))
        while not (0.0 <= next_pos[0] < h and 0.0 <= next_pos[1] < w):
            next_pos = np.random.multivariate_normal(np.array([h / 2, w / 2]), np.array([[h, 0.0], [0.0, w]]))
        df_screen.loc[df_screen.index[0], "x"] = next_pos[1]
        df_screen.loc[df_screen.index[0], "y"] = next_pos[0]
        prev_pos = next_pos

        # iteratively adding positions for every frame
        for i in tqdm(range(1, len(df_screen.index)), disable=False):
            next_pos = np.random.multivariate_normal(prev_pos, np.array([[h / 8, 0.0], [0.0, w / 8]]))
            while not (0.0 <= next_pos[0] < h and 0.0 <= next_pos[1] < w):
                next_pos = np.random.multivariate_normal(prev_pos, np.array([[h / 8, 0.0], [0.0, w / 8]]))

            df_screen.loc[df_screen.index[i], "x"] = next_pos[1]
            df_screen.loc[df_screen.index[i], "y"] = next_pos[0]

            prev_pos = next_pos

        df_screen["x"] = df_screen["x"].astype(int)
        df_screen["y"] = df_screen["y"].astype(int)

        # writer is only initialised after making sure that everything else works
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, f"{self.__class__.NAME}.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

        # loop through all frames and "create" the ground truth
        for frame_idx in tqdm(df_screen["frame"], disable=False):
            if frame_idx >= num_frames:
                print("Number of frames in CSV file exceeds number of frames in video!")
                break

            # compute the range of frames to use for the ground truth
            frame_low = frame_idx - self._half_window_size
            frame_high = frame_idx + self._half_window_size

            # create the heatmap
            current_frame_data = df_screen[df_screen["frame"].between(frame_low, frame_high)]
            current_mu = current_frame_data[["x", "y"]].values
            heatmap = generate_gaussian_heatmap(mu=current_mu, down_scale_factor=10)

            if heatmap.max() > 0.0:
                heatmap /= heatmap.max()

            heatmap = (heatmap * 255).astype("uint8")
            heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)

            # save the resulting frame
            video_writer.write(heatmap)

        video_writer.release()

        print("Saved random gaze ground-truth for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class OpticalFlowFarneback(GroundTruthGenerator):

    NAME = "optical_flow_farneback"

    def __init__(self, config):
        super().__init__(config)

    def get_gt_info(self, run_dir, subject, run):
        # probably just ignore this, since anything where RGB is available is pretty much good?
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
        # TODO: probably not needed
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # initiate video capture and writer
        video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))
        num_frames = int(num_frames)

        if not (w == 800 and h == 600):
            print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
            return

        # writer is only initialised after making sure that everything else works
        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, f"{self.__class__.NAME}.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

        # loop through all frames and compute the optical flow
        _, previous_frame = video_capture.read()
        video_writer.write(np.zeros_like(previous_frame))
        hsv_representation = np.zeros_like(previous_frame)
        hsv_representation[..., 1] = 255
        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        # TODO: should there be some way to indicate whether optical flow is available? e.g. in frame_index?
        for frame_idx in tqdm(range(1, num_frames), disable=False):
            _, current_frame = video_capture.read()
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            optical_flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
            hsv_representation[..., 0] = angle * 180 / np.pi / 2
            hsv_representation[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb_representation = cv2.cvtColor(hsv_representation, cv2.COLOR_HSV2BGR)

            # save the resulting frame
            video_writer.write(rgb_representation)

            previous_frame = current_frame

        video_writer.release()

        print("Saved optical flow for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


class DroneControlFrameMeanGT(GroundTruthGenerator):

    NAME = "drone_control_frame_mean_gt"

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        control_gt_path = os.path.join(index_dir, "control_gt.csv")
        control_measurements_path = os.path.join(index_dir, f"{self.__class__.NAME}.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(control_gt_path):
            df_control_gt = pd.read_csv(control_gt_path)
        else:
            df_control_gt = df_frame_index.copy()
            df_control_gt = df_control_gt[["frame", "subject", "run"]]

        oc = ["throttle_norm [0,1]", "roll_norm [-1,1]", "pitch_norm [-1,1]", "yaw_norm [-1,1]"]
        c = ["throttle", "roll", "pitch", "yaw"]
        if os.path.exists(control_measurements_path):
            df_control_measurements = pd.read_csv(control_measurements_path)
            df_control_measurements = df_control_measurements.rename(
                {c: rc for c, rc in zip(oc, c)}, axis=1)
        else:
            df_control_measurements = df_frame_index.copy()
            df_control_measurements = df_control_measurements[["frame"]]
            df_control_measurements.columns = [c[0]]
            df_control_measurements[c[0]] = np.nan
            for col in c:
                if col not in df_control_measurements.columns:
                    df_control_measurements[col] = np.nan

        if self.__class__.NAME not in df_control_gt.columns:
            df_control_gt[self.__class__.NAME] = -1
            for col in c:
                if col not in df_control_measurements.columns:
                    df_control_measurements[col] = np.nan

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_control_gt["subject"] == subject) & (df_control_gt["run"] == run)

        return df_control_gt, df_control_measurements, control_gt_path, control_measurements_path, match_index, c, oc

    def compute_gt(self, run_dir):
        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_control_gt, df_control_measurements, control_gt_path, control_measurements_path, match_index, c, oc = \
            self.get_gt_info(run_dir, subject, run)

        # define paths
        df_drone_path = os.path.join(run_dir, "drone.csv")
        df_screen_path = os.path.join(run_dir, "screen_timestamps.csv")

        # load dataframes
        df_drone = pd.read_csv(df_drone_path)
        df_screen = pd.read_csv(df_screen_path)

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_drone = df_drone.rename({c: rc for c, rc in zip(oc, c)}, axis=1)
        df_drone = df_drone[(["ts"] + c)]
        df_drone["frame"] = -1

        # filter by screen timestamps
        df_screen, df_drone = filter_by_screen_ts(df_screen, df_drone)

        # compute the mean measurement for each frame
        df_drone = df_drone[(["frame"] + c)]
        df_drone = df_drone.groupby("frame").mean()
        df_drone["frame"] = df_drone.index
        df_drone = df_drone.reset_index(drop=True)
        df_drone = df_drone[(["frame"] + c)]

        # add information about control GT being available to frame-wise screen info
        df_control_gt.loc[match_index, self.__class__.NAME] = df_screen["frame"].isin(df_drone["frame"]).astype(int).values
        df_control_measurements_columns = df_control_measurements.copy()[c]
        df_drone = df_drone.set_index("frame")
        for (_, row), f in tqdm(zip(df_drone.iterrows(), df_drone.index), disable=False, total=len(df_drone.index)):
            df_control_measurements_columns.iloc[match_index & (df_control_gt["frame"] == f)] = row.values
        df_control_measurements[c] = df_control_measurements_columns[c]

        # save control gt to CSV with updated data
        df_control_gt.to_csv(control_gt_path, index=False)
        df_control_measurements.to_csv(control_measurements_path, index=False)


class DroneStateFrameMean(GroundTruthGenerator):

    # TODO: maybe rename this script, since this isn't technically used as ground-truth?

    NAME = "drone_state_frame_mean"

    def get_gt_info(self, run_dir, subject, run):
        # get the path to the index directory
        index_dir = os.path.join(run_dir, os.pardir, os.pardir, "index")
        frame_index_path = os.path.join(index_dir, "frame_index.csv")
        state_path = os.path.join(index_dir, "state.csv")

        df_frame_index = pd.read_csv(frame_index_path)

        if os.path.exists(state_path):
            df_state = pd.read_csv(state_path)
        else:
            df_state = df_frame_index.copy()
            df_state = df_state[["frame", "subject", "run"]]

        columns = ["DroneVelocityX", "DroneVelocityY", "DroneVelocityZ", "DroneAccelerationX", "DroneAccelerationY",
                   "DroneAccelerationZ", "DroneAngularX", "DroneAngularY", "DroneAngularZ"]
        if columns[0] not in df_state.columns:
            for col in columns:
                df_state[col] = np.nan

        # in principle, need only subject and run to identify where to put the new info...
        # e.g. track name is more of a property to filter on...
        match_index = (df_state["subject"] == subject) & (df_state["run"] == run)

        return df_state, state_path, match_index, columns

    def compute_gt(self, run_dir):
        # get info about the current run
        run_info = parse_run_info(run_dir)
        subject = run_info["subject"]
        run = run_info["run"]

        # get the ground-truth info
        df_state, state_path, match_index, columns = self.get_gt_info(run_dir, subject, run)

        # define paths
        df_drone_path = os.path.join(run_dir, "drone.csv")
        df_screen_path = os.path.join(run_dir, "screen_timestamps.csv")

        # load dataframes
        df_drone = pd.read_csv(df_drone_path)
        df_screen = pd.read_csv(df_screen_path)

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_drone = df_drone[(["ts"] + columns)]
        df_drone["frame"] = -1

        # filter by screen timestamps
        df_screen, df_drone = filter_by_screen_ts(df_screen, df_drone)

        # compute the mean measurement for each frame
        df_drone = df_drone[(["frame"] + columns)]
        df_drone = df_drone.groupby("frame").mean()
        df_drone["frame"] = df_drone.index
        df_drone = df_drone.reset_index(drop=True)
        df_drone = df_drone[(["frame"] + columns)]

        # add information about control GT being available to frame-wise screen info
        df_state_columns = df_state.copy()[columns]
        df_drone = df_drone.set_index("frame")
        for (_, row), f in tqdm(zip(df_drone.iterrows(), df_drone.index), disable=False, total=len(df_drone.index)):
            df_state_columns.iloc[match_index & (df_state["frame"] == f)] = row.values
        df_state[columns] = df_state_columns[columns]

        # save control gt to CSV with updated data
        df_state.to_csv(state_path, index=False)


def resolve_gt_class(ground_truth_type: str) -> Type[GroundTruthGenerator]:
    if ground_truth_type == "moving_window_frame_mean_gt":
        return MovingWindowFrameMeanGT
    elif ground_truth_type == "drone_control_frame_mean_gt":
        return DroneControlFrameMeanGT
    elif ground_truth_type == "random_gaze_gt":
        return RandomGazeGT
    elif ground_truth_type == "drone_state_frame_mean":
        return DroneStateFrameMean
    elif ground_truth_type == "optical_flow":
        return OpticalFlowFarneback
    return GroundTruthGenerator


def main(args):
    config = vars(args)

    generator = resolve_gt_class(config["ground_truth_type"])(config)
    generator.generate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-tn", "--track_name", type=str, nargs="+", default=["flat", "wave"], choices=["flat", "wave"],
                        help="The method to use to compute the ground-truth.")
    parser.add_argument("-gtt", "--ground_truth_type", type=str, default="moving_window_frame_mean_gt",
                        choices=["moving_window_frame_mean_gt", "drone_control_frame_mean_gt", "random_gaze_gt",
                                 "drone_state_frame_mean", "optical_flow"],
                        help="The method to use to compute the ground-truth.")
    parser.add_argument("-di", "--directory_index", type=pair, default=None)
    parser.add_argument("-rs", "--random_seed", type=int, default=127,
                        help="The random seed.")
    parser.add_argument("-se", "--skip_existing", action="store_true")

    # arguments only used for moving_window
    parser.add_argument("--mw_size", type=int, default=25,
                        help="Size of the temporal window in frames from which the "
                             "ground-truth for the current frame should be computed.")

    # parse the arguments
    arguments = parser.parse_args()

    # generate the GT
    main(arguments)

