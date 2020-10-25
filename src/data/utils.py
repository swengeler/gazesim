import os
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn

########################################
# FUNCTIONS RELATED TO THE FILE SYSTEM #
########################################


def iterate_directories(data_root, track_names=None):
    if track_names is None:
        track_names = ["flat"]

    directories = []
    if re.search(r"/\d\d_", data_root):
        directories.append(data_root)
    elif re.search(r"/s0\d\d", data_root):
        for run in sorted(os.listdir(data_root)):
            run_dir = os.path.join(data_root, run)
            if os.path.isdir(run_dir) and run[3:] in track_names:
                directories.append(run_dir)
    else:
        for subject in sorted(os.listdir(data_root)):
            subject_dir = os.path.join(data_root, subject)
            if os.path.isdir(subject_dir) and subject.startswith("s"):
                for run in sorted(os.listdir(subject_dir)):
                    run_dir = os.path.join(subject_dir, run)
                    if os.path.isdir(run_dir) and run[3:] in track_names:
                        directories.append(run_dir)

    return directories


def parse_run_info(run_dir):
    if run_dir[-1] == "/":
        run_dir = run_dir[:-1]

    # return the subject number, run number and track name
    result = re.search(r"/s0\d\d", run_dir)
    subject_number = None if result is None else int(result[0][2:])

    result = re.search(r"/s0\d\d/\d\d_", run_dir)
    run_number = None if result is None else int(result[0][6:8])

    result = re.search(r"/s0\d\d/\d\d_.+", run_dir)
    track_name = None if result is None else result[0][9:]

    info = {
        "subject": subject_number,
        "run": run_number,
        "track_name": track_name
    }

    return info


#################################
# FUNCTIONS RELATED TO PLOTTING #
#################################


def generate_gaussian_heatmap(width=800,
                              height=600,
                              mu=np.array([400, 300]),
                              sigma=np.array([[200, 0], [0, 200]]),
                              down_scale_factor=5,
                              combine="max",
                              plot=False,
                              show_progress=False):
    mu_list = []
    if isinstance(mu, np.ndarray):
        if len(mu.shape) == 1:
            mu_list.append(mu)
        elif len(mu.shape) == 2:
            for m in mu:
                mu_list.append(m)
    elif isinstance(mu, list):
        mu_list = mu

    sigma_list = []
    if isinstance(sigma, np.ndarray):
        if len(sigma.shape) == 2:
            sigma_list.append(sigma)
        elif len(sigma.shape) == 3:
            for s in sigma:
                sigma_list.append(s)
    elif isinstance(sigma, list):
        sigma_list = sigma

    if len(mu_list) > 1 and len(sigma_list) == 1:
        sigma_list = [sigma_list[0] for _ in range(len(mu_list))]
    if len(mu_list) == 1 and len(sigma_list) > 1:
        mu_list = [mu_list[0] for _ in range(len(sigma_list))]

    width_small = int(np.round(width / down_scale_factor))
    height_small = int(np.round(height / down_scale_factor))

    grid = np.mgrid[0:width_small, 0:height_small]
    grid = grid.transpose((1, 2, 0))

    values_accumulated = np.zeros((width_small, height_small))
    for m, s in tqdm(zip(mu_list, sigma_list), disable=(not show_progress)):
        gaussian = mvn(mean=(m / down_scale_factor), cov=(s / (down_scale_factor ** 2)))
        values_current = gaussian.pdf(grid)
        values_accumulated = np.maximum(values_accumulated, values_current)

    values_accumulated = cv2.resize(values_accumulated, (height, width), interpolation=cv2.INTER_CUBIC)
    if not values_accumulated.sum() == 0.0:
        values_accumulated /= values_accumulated.sum()
    values_accumulated = values_accumulated.transpose((1, 0))

    if plot:
        plt.imshow(values_accumulated, cmap="jet", interpolation="nearest")
        plt.show()
        """
        values_accumulated = cv2.applyColorMap((values_accumulated / values_accumulated.max() * 255).astype("uint8"),
                                               cv2.COLORMAP_JET)
        cv2.imshow("Heatmap", values_accumulated)
        cv2.waitKey(0)
        """

    return values_accumulated


##########################################################
# FUNCTIONS RELATED TO STANDARD OPERATIONS ON DATAFRAMES #
##########################################################


def filter_by_screen_ts(df_screen, df_other, buffer=0.0):
    # use only those timestamps that can be matched to the "screen" video
    first_screen_ts = df_screen["ts"].iloc[0] - buffer
    last_screen_ts = df_screen["ts"].iloc[-1] + buffer
    df_other = df_other[(first_screen_ts <= df_other["ts"]) & (df_other["ts"] <= last_screen_ts)].copy()

    # compute timestamp windows around each frame to "sort" the gaze measurements into
    frame_ts_prev = df_screen["ts"].values[:-1]
    frame_ts_next = df_screen["ts"].values[1:]
    frame_ts_midpoint = ((frame_ts_prev + frame_ts_next) / 2).tolist()
    frame_ts_midpoint.insert(0, first_screen_ts)
    frame_ts_midpoint.append(last_screen_ts)

    # update the gaze dataframe with the "screen" frames
    # TODO: maybe should just put this in a separate column and save in the CSV file?
    # TODO: CAN'T USE FRAME_IDX HERE!!!!! NEED THE ACTUAL FRAME!!!!!!!!
    for frame_idx, ts_prev, ts_next in zip(df_screen["frame"], frame_ts_midpoint[:-1], frame_ts_midpoint[1:]):
        df_other.loc[(ts_prev <= df_other["ts"]) & (df_other["ts"] < ts_next), "frame"] = frame_idx

    return df_screen, df_other
