import os
import cv2
import numpy as np
import pandas as pd
import json

from tqdm import tqdm
from src.data.utils import run_info_to_path


# def turn_only_data(data_root, split="train", video_name="screen", gt_name="moving_window_gt", convert_range=False):
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
        data_std = np.sqrt(data_std)
        # label_std /= len(df_index.index)
        print("Standard deviation of colour channel values for '{}' data ({} split): {}\n".format(turn_label, split, data_std))


def compute_statistics(config):
    # 1. load split index dataframe
    # 2. filter by training data only
    # 3. loop through remaining dataset and load frames
    df_split = pd.read_csv(config["split"] + ".csv")
    df_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
    df_index = df_index[df_split["split"] == "train"]

    cap_dict = {}

    # compute the mean first (required for std)
    data_mean = None
    for i, row in tqdm(df_index.iterrows(), disable=False, total=len(df_index.index)):
        full_run_path = os.path.join(config["data_root"],
                                     run_info_to_path(row["subject"], row["run"], row["track_name"]))
        if full_run_path not in cap_dict:
            cap_dict[full_run_path] = cv2.VideoCapture(os.path.join(full_run_path,
                                                                    "{}.mp4".format(config["video_name"])))

        cap_dict[full_run_path].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])

        data = cv2.cvtColor(cap_dict[full_run_path].read()[1], cv2.COLOR_BGR2RGB)
        data = data.astype("float64")
        if not config["original_range"]:
            data /= 255.0

        if data_mean is None:
            data_mean = np.mean(data, axis=(0, 1))
        else:
            data_mean += np.mean(data, axis=(0, 1))
    data_mean /= len(df_index.index)

    # compute the standard deviation, basically using the same loop
    data_std = None
    for i, row in tqdm(df_index.iterrows(), disable=False, total=len(df_index.index)):
        full_run_path = os.path.join(config["data_root"],
                                     run_info_to_path(row["subject"], row["run"], row["track_name"]))

        cap_dict[full_run_path].set(cv2.CAP_PROP_POS_FRAMES, row["frame"])

        data = cv2.cvtColor(cap_dict[full_run_path].read()[1], cv2.COLOR_BGR2RGB)
        data = data.astype("float64")
        if not config["original_range"]:
            data /= 255.0

        if data_std is None:
            data_std = np.mean((data - data_mean) ** 2, axis=(0, 1))
        else:
            data_std += np.mean((data - data_mean) ** 2, axis=(0, 1))
    data_std /= len(df_index.index)
    data_std = np.sqrt(data_std)

    # save to the JSON file
    with open(config["split"] + "_info.json", "r") as f:
        split_info = json.load(f)
    if "mean" not in split_info:
        split_info["mean"] = {}
        split_info["std"] = {}
    split_info["mean"][config["video_name"]] = data_mean.tolist()
    split_info["std"][config["video_name"]] = data_std.tolist()
    with open(config["split"] + "_info.json", "w") as f:
        json.dump(split_info, f)


def parse_config(args):
    config = vars(args)
    try:
        split_index = int(config["split"])
        config["split"] = os.path.join(config["data_root"], "splits", "split{:03d}".format(split_index))
    except ValueError:
        if config["split"].endswith(".json"):
            config["split"] = os.path.abspath(config["split"])[:-5]
        elif config["split"].endswith(".csv"):
            config["split"] = os.path.abspath(config["split"])[:-4]
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-v", "--video_name", type=str, default="screen",
                        help="Video (masked or others).")
    parser.add_argument("-s", "--split", default=0,
                        help="The split of the data to compute statistics for (on the training set). "
                             "Can either be the path to a file or an index.")
    parser.add_argument("-or", "--original_range", action="store_true",
                        help="Whether to convert from [0, 255] to [0, 1] range before computing the statistics.")
    # TODO: change so that input is a split index file (stored in subfolder splits/) and once done computing
    #  statistics for it, update the split info file with the statistics
    #  => should probably still have the different video names as keys?

    # parse the arguments
    arguments = parser.parse_args()

    # compute the statistics
    """
    turn_only_data(
        data_root=arguments.data_root,
        split=arguments.split,
        video_name=arguments.video_name,
        gt_name=arguments.ground_truth_type,
        convert_range=arguments.convert_range
    )
    """
    compute_statistics(parse_config(arguments))
