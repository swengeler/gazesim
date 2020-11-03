import os
import numpy as np
import cv2

from tqdm import tqdm
from src.data.utils import iterate_directories


def handle_single_video(config, run_dir, gt_video_path):
    rgb_video_path = os.path.join(run_dir, "screen.mp4")
    hard_video_path = os.path.join(run_dir, "hard_mask_{}.mp4".format(config["ground_truth_name"]))
    soft_video_path = os.path.join(run_dir, "soft_mask_{}.mp4".format(config["ground_truth_name"]))
    mean_video_path = os.path.join(run_dir, "{}.mp4".format(config["mean_mask_name"]))

    # load the mean mask
    mean_mask = None
    if config["mean_mask_path"] is not None:
        mean_mask = cv2.imread(config["mean_mask_path"]).astype("float64")[:, :, 0]
        mean_mask /= 255.0
        if mean_mask.max() > 0.0:
            mean_mask /= mean_mask.max()

    # initialise the capture and writer objects
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    gt_cap = cv2.VideoCapture(gt_video_path)

    w, h, fps, fourcc, num_frames = (rgb_cap.get(i) for i in range(3, 8))
    hard_video_writer, soft_video_writer, mean_video_writer = None, None, None
    if "hard_mask" in config["output_mode"]:
        hard_video_writer = cv2.VideoWriter(hard_video_path, int(fourcc), fps, (int(w), int(h)), True)
    if "soft_mask" in config["output_mode"]:
        soft_video_writer = cv2.VideoWriter(soft_video_path, int(fourcc), fps, (int(w), int(h)), True)
    if "mean_mask" in config["output_mode"]:
        mean_video_writer = cv2.VideoWriter(mean_video_path, int(fourcc), fps, (int(w), int(h)), True)

    if num_frames != gt_cap.get(7):
        print("RGB and GT video do not have the same number of frames for {}.".format(run_dir))
        return

    # loop through all frames
    sml = config["soft_masking_lambda"]
    for frame_idx in tqdm(range(int(num_frames))):
        rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        gt_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        rgb_frame = rgb_cap.read()[1].astype("float64")
        gt_frame = gt_cap.read()[1].astype("float64")[:, :, 0]
        gt_frame /= 255.0
        if gt_frame.max() > 0.0:
            gt_frame /= gt_frame.max()

        if "hard_mask" in config["output_mode"]:
            hard_masked = rgb_frame * gt_frame[:, :, np.newaxis]
            hard_masked = np.round(hard_masked).astype("uint8")
            hard_video_writer.write(hard_masked)

        if "soft_mask" in config["output_mode"]:
            soft_masked = sml * rgb_frame + (1 - sml) * rgb_frame * gt_frame[:, :, np.newaxis]
            soft_masked = np.round(soft_masked).astype("uint8")
            soft_video_writer.write(soft_masked)

        if "mean_mask" in config["output_mode"]:
            mean_masked = rgb_frame * mean_mask[:, :, np.newaxis]
            mean_masked = np.round(mean_masked).astype("uint8")
            mean_video_writer.write(mean_masked)

    if "hard_mask" in config["output_mode"]:
        hard_video_writer.release()
    if "soft_mask" in config["output_mode"]:
        soft_video_writer.release()
    if "mean_mask" in config["output_mode"]:
        mean_video_writer.release()

    print("Finished writing {} for '{}' to '{}'.".format(
        " and ".join(config["output_mode"]), config["ground_truth_name"], run_dir))


def main(config):
    for run_dir in iterate_directories(config["data_root"], config["track_name"]):
        # need ground-truth to be there
        gt_video_path = os.path.join(run_dir, "{}.mp4".format(config["ground_truth_name"]))

        # check if required file exists
        if os.path.exists(gt_video_path):
            handle_single_video(config, run_dir, gt_video_path)
        else:
            print("Skipping '{}' because no video for '{}' exists.".format(run_dir, config["ground_truth_name"]))


def parse_config(args):
    config = vars(args)
    config["data_root"] = os.path.abspath(config["data_root"])
    config["mean_mask_path"] = None if args.mean_mask_path is None else os.path.abspath(config["mean_mask_path"])
    config["mean_mask_name"] = ""
    if config["mean_mask_path"] is not None:
        config["mean_mask_name"] = os.path.splitext(os.path.basename(config["mean_mask_path"]))[0]
    if "all" in config["output_mode"]:
        config["output_mode"] = ["soft_mask", "hard_mask", "mean_mask"]
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-tn", "--track_name", type=str, nargs="+", default=["flat", "wave"],
                        help="The name of the track.")
    parser.add_argument("-gtn", "--ground_truth_name", type=str, default="moving_window_frame_mean_gt",
                        help="The name of the ground-truth video.")
    parser.add_argument("-om", "--output_mode", type=str, nargs="+", default=["soft_mask", "hard_mask"],
                        choices=["all", "soft_mask", "hard_mask", "mean_mask"],
                        help="Which mask type to generate.")
    parser.add_argument("-l", "--soft_masking_lambda", type=float, default=0.2,
                        help="Lambda for soft masking.")
    parser.add_argument("-mmp", "--mean_mask_path", type=str, default=None,
                        help="File path to the mean mask to use for masking.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    main(parse_config(arguments))

