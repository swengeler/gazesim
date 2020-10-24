import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm


def turn_only_data(data_root, split="train", video_name="screen", gt_name="moving_window_gt", convert_range=False):
    cap_dict = {}
    for turn_label in ["turn_left", "turn_right"]:
        df_index = pd.read_csv(os.path.join(data_root, f"{turn_label}_{split}.csv"))

        # compute per-channel mean of data and ground-truth (maybe not for GT?)
        data_mean = None
        for i, row in tqdm(df_index.iterrows(), disable=False):
            full_run_path = os.path.join(data_root, row["rel_run_path"])
            if full_run_path not in cap_dict:
                cap_dict[full_run_path] = {
                    "data": cv2.VideoCapture(os.path.join(full_run_path, f"{video_name}.mp4")),
                    # "label": cv2.VideoCapture(os.path.join(full_run_path, f"{gt_name}.mp4"))
                }

            cap_dict[full_run_path]["data"].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            # cap_dict[full_run_path]["label"].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])

            data = cv2.cvtColor(cap_dict[full_run_path]["data"].read()[1], cv2.COLOR_BGR2RGB)
            data = data.astype("float64")
            if convert_range:
                data /= 255.0
            # success, label = cap_dict[full_run_path]["label"].read()

            if data_mean is None:
                data_mean = np.mean(data, axis=(0, 1))
            else:
                data_mean += np.mean(data, axis=(0, 1))

        data_mean /= len(df_index.index)
        # label_mean /= len(df_index.index)
        print("Mean colour channel values for '{}' data ({} split): {}".format(turn_label, split, data_mean))

        # compute per-channel standard deviation
        data_std = None
        for i, row in tqdm(df_index.iterrows(), disable=False):
            full_run_path = os.path.join(data_root, row["rel_run_path"])

            cap_dict[full_run_path]["data"].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            # cap_dict[full_run_path]["label"].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])

            data = cv2.cvtColor(cap_dict[full_run_path]["data"].read()[1], cv2.COLOR_BGR2RGB)
            data = data.astype("float64")
            if convert_range:
                data /= 255.0
            # success, label = cap_dict[full_run_path]["label"].read()

            if data_std is None:
                data_std = np.mean((data - data_mean) ** 2, axis=(0, 1))
            else:
                data_std += np.mean((data - data_mean) ** 2, axis=(0, 1))

        data_std /= len(df_index.index)
        # label_std /= len(df_index.index)
        print("Standard deviation of colour channel values for '{}' data ({} split): {}\n".format(turn_label, split, data_std))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-t", "--ground_truth_type", type=str, default="moving_window_gt", choices=["moving_window_gt"],
                        help="The method to use to compute the ground-truth.")
    parser.add_argument("-v", "--video_name", type=str, default="screen",
                        help="Video (masked or others).")
    parser.add_argument("-s", "--split", type=str, default="train", choices=["train", "val", "test"],
                        help="The split of the data to compute statistics for.")
    parser.add_argument("-c", "--convert_range", action="store_true",
                        help="Whether to convert from [0, 255] to [0, 1] range before computing the statistics.")
    # TODO: for now no selection for different subset of data (only left and right turn)

    # parse the arguments
    arguments = parser.parse_args()

    # compute the statistics
    turn_only_data(
        data_root=arguments.data_root,
        split=arguments.split,
        video_name=arguments.video_name,
        gt_name=arguments.ground_truth_type,
        convert_range=arguments.convert_range
    )
