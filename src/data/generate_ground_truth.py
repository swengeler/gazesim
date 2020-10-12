import os
import threading
import queue
import numpy as np
import pandas as pd
import cv2
import tables

from src.data.utils import iterate_directories, generate_gaussian_heatmap
from tqdm import tqdm
from time import time


class GroundTruthGenerator(threading.Thread):

    def __init__(self, q, *args, **kwargs):
        super(GroundTruthGenerator, self).__init__(*args, **kwargs)
        self.q = q

    def compute_gt(self, run_dir):
        raise NotImplementedError()

    def run(self):
        while True:
            try:
                run_dir = self.q.get(timeout=3)  # 3s timeout
            except queue.Empty:
                return
            self.compute_gt(run_dir)
            self.q.task_done()


class MovingWindowGT(GroundTruthGenerator):

    def __init__(self, *args, window_size=25, **kwargs):
        super(MovingWindowGT, self).__init__(*args, **kwargs)

        # make sure the input is correct
        assert window_size > 0 and window_size % 2 == 1

        self._half_window_size = int((window_size - 1) / 2)

    def compute_gt(self, run_dir):
        start = time()

        # initiate video capture and writer
        video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))

        if not (w == 800 and h == 600):
            print("WARNING: Screen video does not have the correct dimensions for directory '{}'.".format(run_dir))
            return

        # load data frames with the timestamps and positions for gaze and the frames/timestamps for the video
        df_gaze = pd.read_csv(os.path.join(run_dir, "gaze_on_surface.csv"))
        df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))

        # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
        df_gaze = df_gaze[["ts", "frame", "norm_x_su", "norm_y_su"]]
        df_gaze.columns = ["ts", "frame", "x", "y"]
        df_gaze["x"] = df_gaze["x"] * 800
        df_gaze["y"] = (1.0 - df_gaze["y"]) * 600

        # use only those timestamps that can be matched to the "screen" video
        first_screen_ts = df_screen["ts"].iloc[0]
        last_screen_ts = df_screen["ts"].iloc[-1]
        df_gaze = df_gaze[(first_screen_ts <= df_gaze["ts"]) & (df_gaze["ts"] <= last_screen_ts)]

        # compute timestamp windows around each frame to "sort" the gaze measurements into
        frame_ts_prev = df_screen["ts"].values[:-1]
        frame_ts_next = df_screen["ts"].values[1:]
        frame_ts_midpoint = ((frame_ts_prev + frame_ts_next) / 2).tolist()
        frame_ts_midpoint.insert(0, first_screen_ts)
        frame_ts_midpoint.append(last_screen_ts)

        # update the gaze dataframe with the "screen" frames
        # TODO: maybe should just put this in a separate column and save in the CSV file?
        for frame_idx, (ts_prev, ts_next) in enumerate(zip(frame_ts_midpoint[:-1], frame_ts_midpoint[1:])):
            df_gaze.loc[(ts_prev <= df_gaze["ts"]) & (df_gaze["ts"] < ts_next), "frame"] = frame_idx

        # since the measurements are close together and to reduce computational load,
        # compute the mean measurement for each frame
        # TODO: think about whether this is a good way to deal with this issue
        df_gaze = df_gaze[["frame", "x", "y"]]
        df_gaze = df_gaze.groupby("frame").mean()
        df_gaze["frame"] = df_gaze.index

        # create new dataframe to write frame-level information into
        df_frame_info = df_screen.copy()
        df_frame_info = df_frame_info[["frame"]]
        df_frame_info["gt_available"] = df_frame_info["frame"].isin(df_gaze["frame"]).astype(int)

        video_writer = cv2.VideoWriter(
            os.path.join(run_dir, "moving_window_gt.mp4"),
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

        # file = tables.open_file(os.path.join(run_dir, "moving_window_gt.h5"), mode="w")
        # atom = tables.UInt8Atom()
        # array = file.create_earray(file.root, "data", atom, (0, h, w))

        # loop through all frames and compute the ground truth (where possible)
        # ground_truth_list = []
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

            # ground_truth_list.append(heatmap.astype("float16"))
            # array.append(np.expand_dims((heatmap * 255).astype("uint8"), 0))

            heatmap = (heatmap * 255).astype("uint8")
            heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)

            # load the frame and modify it
            """
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, image = video_capture.read()
            if success:
                if heatmap.max() > 0.0:
                    heatmap /= heatmap.max()
                heatmap = cv2.applyColorMap((heatmap * 255).astype("uint8"), cv2.COLORMAP_JET)
                image = cv2.addWeighted(image, 1.0, heatmap, 0.3, 0)
            """
            video_writer.write(heatmap)

        video_writer.release()
        # np.save(os.path.join(run_dir, "moving_window_gt"), np.stack(ground_truth_list, axis=0))
        # file.close()

        # save screen dataframe to CSV, updated with additional column
        df_frame_info.to_csv(os.path.join(run_dir, "screen_frame_info.csv"), index=False)

        print("Saved moving window ground-truth for directory '{}' after {:.2f}s.".format(run_dir, time() - start))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-n", "--num_workers", type=int, default=4,
                        help="The number of workers to use for generating the ground-truth.")
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-t", "--ground_truth_type", type=str, default="moving_window", choices=["moving_window"],
                        help="The method to use to compute the ground-truth.")

    # arguments only used for moving_window
    parser.add_argument("--mw_size", type=int, default=25,
                        help="Size of the temporal window in frames from which the "
                             "ground-truth for the current frame should be computed.")

    # parse the arguments
    arguments = parser.parse_args()

    # generate the ground-truth
    task_queue = queue.Queue()
    for rd in iterate_directories(arguments.data_root):
        task_queue.put_nowait(rd)

    try:
        if arguments.ground_truth_type == "moving_window":
            for _ in range(arguments.num_workers):
                MovingWindowGT(task_queue, window_size=arguments.mw_size).start()
        else:
            print("No ground-truth type specified. Not generating any ground-truth.")
    except KeyboardInterrupt as e:
        print(e)

    task_queue.join()
