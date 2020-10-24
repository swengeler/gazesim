import os
import re
import numpy as np
import cv2

from tqdm import tqdm
from src.data.utils import iterate_directories


def handle_single_video(args, run_dir, gt_video_path):
    rgb_video_path = os.path.join(run_dir, "screen.mp4")
    hard_video_path = os.path.join(run_dir, f"hard_mask_{args.ground_truth_name}.mp4")
    soft_video_path = os.path.join(run_dir, f"soft_mask_{args.ground_truth_name}.mp4")
    mean_video_path = os.path.join(run_dir, f"mean_mask_{args.ground_truth_name}.mp4")

    # load the mean mask
    mean_mask = cv2.imread(args.mean_mask_path).astype("float64")[:, :, 0]
    mean_mask /= 255.0
    if mean_mask.max() > 0.0:
        mean_mask /= mean_mask.max()

    # initialise the capture and writer objects
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    gt_cap = cv2.VideoCapture(gt_video_path)

    w, h, fps, fourcc, num_frames = (rgb_cap.get(i) for i in range(3, 8))
    hard_video_writer, soft_video_writer, mean_video_writer = None, None, None
    if args.output_mode in ["all", "hard_mask"]:
        hard_video_writer = cv2.VideoWriter(hard_video_path, int(fourcc), fps, (int(w), int(h)), True)
    if args.output_mode in ["all", "soft_mask"]:
        soft_video_writer = cv2.VideoWriter(soft_video_path, int(fourcc), fps, (int(w), int(h)), True)
    if args.output_mode in ["all", "mean_mask"]:
        mean_video_writer = cv2.VideoWriter(mean_video_path, int(fourcc), fps, (int(w), int(h)), True)

    if num_frames != gt_cap.get(7):
        print("RGB and GT video do not have the same number of frames for {}.".format(run_dir))
        return

    # loop through all frames
    for frame_idx in tqdm(range(int(num_frames))):
        rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        gt_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        rgb_frame = rgb_cap.read()[1].astype("float64")
        gt_frame = gt_cap.read()[1].astype("float64")[:, :, 0]
        gt_frame /= 255.0
        if gt_frame.max() > 0.0:
            gt_frame /= gt_frame.max()

        if args.output_mode in ["all", "hard_mask"]:
            hard_masked = rgb_frame * gt_frame[:, :, np.newaxis]
            hard_masked = np.round(hard_masked).astype("uint8")
            hard_video_writer.write(hard_masked)

        if args.output_mode in ["all", "soft_mask"]:
            soft_masked = args.soft_masking_lambda * rgb_frame + (1 - args.soft_masking_lambda) * rgb_frame * gt_frame[:, :, np.newaxis]
            soft_masked = np.round(soft_masked).astype("uint8")
            soft_video_writer.write(soft_masked)

        if args.output_mode in ["all", "mean_mask"]:
            mean_masked = rgb_frame * mean_mask[:, :, np.newaxis]
            mean_masked = np.round(mean_masked).astype("uint8")
            mean_video_writer.write(mean_masked)

    if args.output_mode in ["all", "hard_mask"]:
        hard_video_writer.release()
    if args.output_mode in ["all", "soft_mask"]:
        soft_video_writer.release()
    if args.output_mode in ["all", "mean_mask"]:
        mean_video_writer.release()


def main(args):
    args.data_root = os.path.abspath(args.data_root)
    # args.output_mode = ["soft_mask", "hard_mask", "mean_mask"] if args.output_mode == "all" else [args.output_mode]

    for run_dir in iterate_directories(args.data_root):
        # need ground-truth to be there
        gt_video_path = os.path.join(run_dir, f"{args.ground_truth_name}.mp4")

        # check if required file exists
        if os.path.exists(gt_video_path):
            handle_single_video(args, run_dir, gt_video_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-tn", "--track_name", type=str, default="flat",
                        help="The name of the track.")
    parser.add_argument("-gtn", "--ground_truth_name", type=str, default="moving_window_gt",
                        help="The name of the ground-truth video.")
    parser.add_argument("-om", "--output_mode", type=str, default="all",
                        choices=["all", "soft_mask", "hard_mask", "mean_mask"],
                        help="Which mask type to generate.")
    parser.add_argument("-l", "--soft_masking_lambda", type=float, default=0.2,
                        help="Lambda for soft masking.")
    parser.add_argument("-mmp", "--mean_mask_path", type=str, default=None,
                        help="File path to the mean mask to use for masking.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    main(arguments)

