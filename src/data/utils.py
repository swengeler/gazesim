import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from scipy.stats import multivariate_normal as mvn

########################################
# FUNCTIONS RELATED TO THE FILE SYSTEM #
########################################


def iterate_directories(data_root, track_name="flat"):
    # loop over the subjects
    directories = []
    for subject in os.listdir(data_root):
        subject_dir = os.path.join(data_root, subject)
        if os.path.isdir(subject_dir) and subject.startswith("s"):
            # loop over the experiments for each subject
            for run in os.listdir(subject_dir):
                run_dir = os.path.join(subject_dir, run)
                if os.path.isdir(run_dir) and track_name in run:
                    directories.append(run_dir)
    return sorted(directories)


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
        gaussian = mvn(mean=m / down_scale_factor, cov=s / (down_scale_factor ** 2))
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