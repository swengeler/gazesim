import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm


def turn_only_data(data_root, split="train", gt_name="moving_window_gt"):
    cap_dict = {}
    for turn_label in ["turn_left", "turn_right"]:
        df_index = pd.read_csv(os.path.join(data_root, f"{turn_label}_{split}.csv"))

        # compute per-channel mean of data and ground-truth (maybe not for GT?)
        data_mean = None
        for i, row in tqdm(df_index.iterrows(), disable=False):
            full_run_path = os.path.join(data_root, row["rel_run_path"])
            if full_run_path not in cap_dict:
                cap_dict[full_run_path] = {
                    "data": cv2.VideoCapture(os.path.join(full_run_path, "screen.mp4")),
                    # "label": cv2.VideoCapture(os.path.join(full_run_path, f"{gt_name}.mp4"))
                }

            cap_dict[full_run_path]["data"].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            # cap_dict[full_run_path]["label"].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])

            success, data = cap_dict[full_run_path]["data"].read()
            data = data.astype("float64")
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

            success, data = cap_dict[full_run_path]["data"].read()
            data = data.astype("float64")
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
    parser.add_argument("-t", "--ground_truth_type", type=str, default="moving_window", choices=["moving_window"],
                        help="The method to use to compute the ground-truth.")  # TODO: probably remove this...
    parser.add_argument("-s", "--split", type=str, default="train", choices=["train", "val", "test"],
                        help="The split of the data to compute statistics for.")
    # TODO: for now no selection for different subset of data (only left and right turn)

    # parse the arguments
    arguments = parser.parse_args()

    # compute the statistics
    turn_only_data(arguments.data_root, arguments.split, f"{arguments.ground_truth_type}_gt")
