import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn
from src.data.utils import filter_by_screen_ts


def generate_mean_mask(args):
    # load the indexing dataframe from the root directory
    df_index = pd.read_csv(args.index_file)

    # define the dimensions of the output map and creating a scaled down heatmap
    sigma = np.array([[200, 0], [0, 200]])
    width = 800
    height = 600
    down_scale_factor = 5

    width_small = int(np.round(width / down_scale_factor))
    height_small = int(np.round(height / down_scale_factor))

    grid = np.mgrid[0:width_small, 0:height_small]
    grid = grid.transpose((1, 2, 0))

    # create the array to accumulate values in
    values_accumulated = np.zeros((width_small, height_small))

    # loop through the frames
    df_dict = {}
    for _, row in tqdm(df_index.iterrows(), total=len(df_index.index)):
        # load the dataframes if they haven't been loaded yet
        if row["rel_run_path"] not in df_dict:
            df_screen = pd.read_csv(os.path.join(args.data_root, row["rel_run_path"], "screen_timestamps.csv"))
            df_gaze = pd.read_csv(os.path.join(args.data_root, row["rel_run_path"], "gaze_on_surface.csv"))

            # filter out the frames that cannot be found in the index
            indexed_frames = df_index[df_index["rel_run_path"] == row["rel_run_path"]]["frame"]
            df_screen = df_screen[df_screen["frame"].isin(indexed_frames)]

            # select only the necessary data from the gaze dataframe and compute the on-screen coordinates
            df_gaze = df_gaze[["ts", "frame", "norm_x_su", "norm_y_su"]]
            df_gaze.columns = ["ts", "frame", "x", "y"]
            df_gaze["x"] = df_gaze["x"] * 800
            df_gaze["y"] = (1.0 - df_gaze["y"]) * 600

            # filter by screen timestamps
            df_screen, df_gaze = filter_by_screen_ts(df_screen, df_gaze)

            df_dict[row["rel_run_path"]] = {"screen": df_screen, "gaze": df_gaze}

        df_gaze = df_dict[row["rel_run_path"]]["gaze"]
        current_gaze = df_gaze[df_gaze["frame"] == row["frame"]]
        for _, gaze_row in current_gaze.iterrows():
            mu = np.array([gaze_row["x"], gaze_row["y"]])
            gaussian = mvn(mean=(mu / down_scale_factor), cov=(sigma / (down_scale_factor ** 2)))
            values_current = gaussian.pdf(grid)
            values_accumulated += values_current

    values_accumulated = cv2.resize(values_accumulated.copy(), (height, width), interpolation=cv2.INTER_CUBIC)
    values_accumulated /= len(df_index.index)
    mean_mask = values_accumulated.transpose((1, 0))

    if mean_mask.max() > 0.0:
        mean_mask /= mean_mask.max()
    mean_mask = (mean_mask * 255).astype("uint8")
    mean_mask = np.repeat(mean_mask[:, :, np.newaxis], 3, axis=2)
    cv2.imwrite(os.path.join(args.data_root, f"mean_mask_{os.path.basename(args.index_file)[:-4]}.png"), mean_mask)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-if", "--index_file", type=str, default=None,
                        help="CSV file that indexes the frames from which to take the gaze measurements.")
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="File name to save the mask under.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    generate_mean_mask(arguments)
