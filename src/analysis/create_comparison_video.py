import os
import re
import numpy as np
import pandas as pd
import cv2
import torch

from tqdm import tqdm
from src.models.utils import image_softmax
from src.models.resnet import ResNet18BaseModelSimple
from src.data.datasets import get_dataset
from src.data.utils import filter_by_screen_ts

# TODO options for which "frames" to include
# - include only those with GT given
# - include only those on valid lap, expected trajectory, left/right turn etc.
# - split non-adjacent sections into separately videos/clips (maybe a bit overkill?)


def handle_single_video(args, run_dir, frame_info_path):
    # create the directory structure to save the video
    rel_run_dir = os.path.relpath(run_dir, os.path.abspath(os.path.join(run_dir, os.pardir, os.pardir)))
    log_dir = os.path.abspath(os.path.join(os.path.dirname(args.model_path), os.pardir))
    save_dir = os.path.join(log_dir, "visualisations", rel_run_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # model info/parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_info = torch.load(args.model_path, map_location=device)

    # create model
    # TODO: should work with other model classes
    model = ResNet18BaseModelSimple(transfer_weights=False)
    model.load_state_dict(model_info["model_state_dict"])
    model = model.to(device)
    model.eval()

    # load the dataframe with the info
    df_frame_info = pd.read_csv(frame_info_path)

    # create the dataset
    video_dataset = get_dataset(run_dir, data_type="single_video", resize_height=300, use_pims=args.use_pims)
    video_dataset.return_original = True

    # determine size and where to position things based on the output mode
    output_size = (400, 300)
    positions = {}
    df_gaze = None
    if args.output_mode == "overlay_maps":
        output_size = (400 * 2, 300 + 100)
        positions["ground_truth"] = (10, 40)
        positions["prediction"] = (10, 75)
        positions["valid_lap"] = (310, 40)
        positions["expected_trajectory"] = (310, 75)
        positions["turn_left"] = (610, 40)
        positions["turn_right"] = (610, 75)
    elif args.output_mode == "overlay_all":
        output_size = (400, 300 + 300)
        positions["ground_truth"] = (10, 40)
        positions["prediction"] = (10, 75)
        positions["valid_lap"] = (10, 140)
        positions["expected_trajectory"] = (10, 175)
        positions["turn_left"] = (10, 240)
        positions["turn_right"] = (10, 275)
    elif args.output_mode == "overlay_none":
        output_size = (400 * 3, 300 + 100)
        positions["ground_truth"] = (410, 75)
        positions["prediction"] = (810, 75)
        positions["valid_lap"] = (10, 40)
        positions["expected_trajectory"] = (10, 75)
        positions["turn_left"] = (410, 40)
        positions["turn_right"] = (810, 40)
    elif args.output_mode == "overlay_raw":
        output_size = (400, 300 + 100)
        positions["ground_truth"] = (10, 40)
        positions["prediction"] = (10, 75)

        df_gaze = pd.read_csv(os.path.join(run_dir, "gaze_on_surface.csv"))
        df_gaze = df_gaze[["ts", "frame", "norm_x_su", "norm_y_su"]]
        df_gaze.columns = ["ts", "frame", "x", "y"]
        df_gaze["x"] = (df_gaze["x"] * 800) / 2
        df_gaze["y"] = ((1.0 - df_gaze["y"]) * 600) / 2

        df_screen = pd.read_csv(os.path.join(run_dir, "screen_timestamps.csv"))
        df_screen, df_gaze = filter_by_screen_ts(df_screen, df_gaze)

    # extract FPS and codec information and create the video writer
    if run_dir not in video_dataset.cap_dict:
        video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
    else:
        video_capture = video_dataset.cap_dict[os.path.join(run_dir)]["data"]
    fps, fourcc = (video_capture.get(i) for i in range(5, 7))
    video_name = f"{args.ground_truth_name}_comparison_{args.output_mode}.mp4"
    video_writer = cv2.VideoWriter(os.path.join(save_dir, video_name), int(fourcc), fps, output_size, True)

    # loop through all frames in frame_info
    for _, row in tqdm(df_frame_info.iterrows(), total=len(df_frame_info.index)):
        # return original frame and predict label
        frame, label, frame_original = video_dataset[row["frame"]]
        frame, label = frame.unsqueeze(0).to(device), label.unsqueeze(0).to(device) if label is not None else label
        predicted_label = model(frame)
        predicted_label = image_softmax(predicted_label)

        # downscale original frame to match the size of the labels
        frame = cv2.resize(frame_original, (400, 300))
        if args.use_pims:
            # TODO: think again about whether the frames loaded with OpenCV
            #  in the datasets need to be converted from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # should always match but check just to be sure => should maybe print out if it doesn't match
        if label is None or row["gt_available"] == 0:
            label = torch.zeros(predicted_label.size())

        # convert to numpy for easier processing
        label = label.cpu().detach().numpy()
        predicted_label = predicted_label.cpu().detach().numpy()

        # stack greyscale labels to become RGB
        label = np.repeat(np.squeeze(label)[np.newaxis, :, :], 3, axis=0).transpose((1, 2, 0))
        predicted_label = np.repeat(np.squeeze(predicted_label)[np.newaxis, :, :], 3, axis=0).transpose((1, 2, 0))

        # normalise using the maximum of the maxima of either label and convert to [0, 255] scale
        norm_max = max([label.max(), predicted_label.max()])
        if norm_max != 0:
            label /= norm_max
            predicted_label /= norm_max
        label = (label * 255).astype("uint8")
        predicted_label = (predicted_label * 255).astype("uint8")

        # set all but one colour channel for GT and predicted labels to 0
        label[:, :, 1:] = 0
        predicted_label[:, :, :-1] = 0

        # stack the frames/labels for the video
        temp = np.zeros(output_size[::-1] + (3,), dtype="uint8")
        if args.output_mode == "overlay_maps":
            combined_labels = cv2.addWeighted(label, 0.5, predicted_label, 0.5, 0)
            new_frame = np.hstack((frame, combined_labels))
            temp[100:, :, :] = new_frame
        elif args.output_mode == "overlay_all":
            frame = np.repeat(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, axis=2)
            combined_labels = cv2.addWeighted(label, 0.5, predicted_label, 0.5, 0)
            new_frame = cv2.addWeighted(frame, 0.5, combined_labels, 0.5, 0)
            temp[300:, :, :] = new_frame
        elif args.output_mode == "overlay_none":
            new_frame = np.hstack((frame, label, predicted_label))
            temp[100:, :, :] = new_frame
        elif args.output_mode == "overlay_raw":
            frame = np.repeat(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, axis=2)
            combined_labels = cv2.addWeighted(label, 0.5, predicted_label, 0.5, 0)
            new_frame = cv2.addWeighted(frame, 0.3, combined_labels, 0.7, 0)

            current_gaze = df_gaze[df_gaze["frame"] == row["frame"]]
            for _, gaze_row in current_gaze.iterrows():
                cv2.circle(new_frame, (int(gaze_row["x"]), int(gaze_row["y"])), 4, (0, 255, 0), -1)

            temp[100:, :, :] = new_frame
        new_frame = temp

        # add the other information we want to display
        cv2.putText(new_frame, "Ground-truth", positions["ground_truth"], cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)
        cv2.putText(new_frame, "Prediction", positions["prediction"], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
        if args.output_mode != "overlay_raw":
            cv2.putText(new_frame, "Valid lap", positions["valid_lap"], cv2.FONT_HERSHEY_DUPLEX, 0.8,
                        (255, 255, 255) if row["valid_lap"] == 1 else (50, 50, 50), 1)
            cv2.putText(new_frame, "Expected trajectory", positions["expected_trajectory"], cv2.FONT_HERSHEY_DUPLEX, 0.8,
                        (255, 255, 255) if row["expected_trajectory"] == 1 else (50, 50, 50), 1)
            cv2.putText(new_frame, "Left turn", positions["turn_left"], cv2.FONT_HERSHEY_DUPLEX, 0.8,
                        (255, 255, 255) if row["turn_left"] == 1 else (50, 50, 50), 1)
            cv2.putText(new_frame, "Right turn", positions["turn_right"], cv2.FONT_HERSHEY_DUPLEX, 0.8,
                        (255, 255, 255) if row["turn_right"] == 1 else (50, 50, 50), 1)

        # write the newly created frame to file
        video_writer.write(new_frame)

        if row["frame"] >= 500:
            break


def handle_single_run(args, run_dir):
    # need screen_frame_info.csv for information about valid lap etc.
    # need ground-truth to be there as well
    gt_video_path = os.path.join(run_dir, f"{args.ground_truth_name}.mp4")
    df_frame_info_path = os.path.join(run_dir, "screen_frame_info.csv")

    # check if required files exist
    if os.path.exists(gt_video_path) and os.path.exists(df_frame_info_path):
        handle_single_video(args, run_dir, df_frame_info_path)


def main(args):
    args.data_root = os.path.abspath(args.data_root)
    # loop through directory structure and create plots for every run/video that has the necessary information
    # check if data_root is already a subject or run directory
    if re.search(r"/s0\d\d", args.data_root):
        if re.search(r"/\d\d_", args.data_root):
            handle_single_run(args, args.data_root)
        else:
            for run in sorted(os.listdir(args.data_root)):
                run_dir = os.path.join(args.data_root, run)
                if os.path.isdir(run_dir) and args.track_name in run_dir:
                    handle_single_run(args, run_dir)
    else:
        for subject in sorted(os.listdir(args.data_root)):
            subject_dir = os.path.join(args.data_root, subject)
            if os.path.isdir(subject_dir):
                for run in sorted(os.listdir(subject_dir)):
                    run_dir = os.path.join(subject_dir, run)
                    if os.path.isdir(run_dir) and args.track_name in run_dir:
                        handle_single_run(args, run_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-tn", "--track_name", type=str, default="flat",
                        help="The name of the track.")
    parser.add_argument("-gtn", "--ground_truth_name", type=str, default="moving_window_gt",
                        help="The name of the ground-truth video.")
    parser.add_argument("-om", "--output_mode", type=str, default="overlay_maps",
                        choices=["overlay_maps", "overlay_all", "overlay_none", "overlay_raw"],
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("--use_pims", action="store_true",
                        help="Whether to use PIMS (PyAV) instead of OpenCV for reading frames.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    main(arguments)

