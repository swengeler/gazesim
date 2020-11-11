import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import cv2
import torch

from tqdm import tqdm
from src.models.utils import image_softmax
from src.models.c3d import C3DRegressor
from src.data.old_datasets import get_dataset
from src.data.utils import parse_run_info, find_contiguous_sequences, resolve_split_index_path, run_info_to_path
from src.training.config import parse_config as parse_train_config
from src.training.helpers import to_device, resolve_model_class, resolve_dataset_class, resolve_optimiser_class
from src.training.helpers import resolve_losses, resolve_output_processing_func, resolve_logger_class

# TODO options for which "frames" to include
# - include only those with GT given
# - include only those on valid lap, expected trajectory, left/right turn etc.
# - split non-adjacent sections into separately videos/clips (maybe a bit overkill?)

# TODO: need new way to extract clips from index files, maybe:
# - supply index file and only allow validation and test data (set which to take or both)
# - if specified also filter by subject, run, lap
# - extract clips using the existing function for that
# - how do datasets have to be modified? => subindex?
# - everything should be similarly flexible, based on config? model checkpoint?
style.use("ggplot")


def create_frame(fig, ax, control_gt, control_prediction, input_images):
    # plot in numpy and convert to opencv-compatible image
    error = control_gt - control_prediction
    x = np.arange(len(control_gt))
    x_labels = ["throttle", "roll", "pitch", "yaw"]
    width = 0.3

    ax.bar(x - width, control_gt, width, label="GT", color="#55a868")
    ax.bar(x, control_prediction, width, label="pred", color="#4c72b0")
    ax.bar(x + width, error, width, label="error", color="#c44e52")
    ax.set_ylim(-1, 1)
    # ax.set_xlim(x[0] - 2 * width, x[-1] + 2 * width)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.axhline(c="k", lw=0.2)
    ax.legend()
    # fig.tight_layout()

    fig.canvas.draw()
    plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # stack the frames/labels for the video
    frame = np.hstack(input_images + [plot_image])
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    plt.cla()

    return frame


def test(config):
    # load frame_index and split_index
    frame_index = pd.read_csv(os.path.join(config["data_root"], "index", "frame_index.csv"))
    split_index = pd.read_csv(config["split_config"] + ".csv")
    # TODO: maybe check that the split index actually contains data that can be used properly

    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(config["gpu"])
                          if use_cuda and config["gpu"] < torch.cuda.device_count() else "cpu")

    # define paths
    log_dir = os.path.abspath(os.path.join(os.path.dirname(config["model_load_path"]), os.pardir))
    save_dir = os.path.join(log_dir, "visualisations")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_path = os.path.join(log_dir, "config.json")

    # load the config
    with open(config_path, "r") as f:
        train_config = json.load(f)
    train_config["data_root"] = config["data_root"]
    train_config["gpu"] = config["gpu"]
    train_config["model_load_path"] = config["model_load_path"]
    train_config = parse_train_config(train_config)
    train_config["split_config"] = config["split_config"]

    # load the model
    model_info = train_config["model_info"]
    model = resolve_model_class(train_config["model_name"])(train_config)
    model.load_state_dict(model_info["model_state_dict"])
    model = model.to(device)
    model.eval()

    # TODO: when loading train_config, should replace data_root

    # TODO: apply filter (can include split, subject, run I guess, not sure that lap would make sense)
    #  => actually, if we wanted to just extract a single video for one lap, it would probably make sense...
    # but if the default is that everything is used... then there is a bit of an issue for labeling complete sequences
    # could either ignore that if it is the case, complain/skip if there is overlap in a single sequence or just
    # use a more "rigorous" structure, where we always filter by split => probably the latter

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    for split in config["split"]:
        current_frame_index = frame_index.loc[split_index["split"] == split]
        current_dataset = resolve_dataset_class(train_config["dataset_name"])(train_config, split=split)
        sequences = find_contiguous_sequences(current_frame_index, new_index=True)

        video_writer_dict = {}
        # TODO: maybe would be better to additionally sort by subject/run somehow and take care of all of the
        #  sequences for an individual run_dir at once
        run_dirs = [run_info_to_path(current_frame_index["subject"].iloc[si],
                                     current_frame_index["run"].iloc[si],
                                     current_frame_index["track_name"].iloc[si])
                    for si, _ in sequences]
        if len(config["subjects"]) > 0:
            check = [current_frame_index["subject"].iloc[si] in config["subjects"] for si, _ in sequences]
            sequences = [sequences[i] for i in range(len(sequences)) if check[i]]
            run_dirs = [run_dirs[i] for i in range(len(run_dirs)) if check[i]]
        if len(config["runs"]) > 0:
            check = [current_frame_index["run"].iloc[si] in config["runs"] for si, _ in sequences]
            sequences = [sequences[i] for i in range(len(sequences)) if check[i]]
            run_dirs = [run_dirs[i] for i in range(len(run_dirs)) if check[i]]

        run_dir = run_dirs[0]
        for (start_index, end_index), current_run_dir in tqdm(zip(sequences, run_dirs), disable=False, total=len(sequences)):
            # TODO: can't use frame_index here (which has the original indexing)
            """
            new_run_dir = run_info_to_path(current_frame_index["subject"].iloc[start_index],
                                           current_frame_index["run"].iloc[start_index],
                                           current_frame_index["track_name"].iloc[start_index])
            """
            if current_run_dir != run_dir:
                video_writer_dict[run_dir].release()
                run_dir = current_run_dir

            if run_dir not in video_writer_dict:
                video_dir = os.path.join(save_dir, run_dir)
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)
                video_capture = cv2.VideoCapture(os.path.join(config["data_root"], run_dir, "screen.mp4"))
                fps, fourcc = (video_capture.get(i) for i in range(5, 7))
                video_name = f"control_comparison_{split}.mp4"
                video_writer = cv2.VideoWriter(os.path.join(video_dir, video_name), int(fourcc), fps, (1600, 600), True)
                video_writer_dict[run_dir] = video_writer

            for index in tqdm(range(start_index, end_index), disable=True):
                # read the current data sample
                sample = to_device(current_dataset[index], device, make_batch=True)

                # compute the loss
                prediction = model(sample)
                prediction["output_control"] = resolve_output_processing_func("output_control")(prediction["output_control"])
                """
                individual_losses = torch.nn.functional.mse_loss(prediction["output_control"],
                                                                 sample["output_control"],
                                                                 reduction="none")
                """
                # TODO: maybe plot the difference in some way as well...

                # get the values as numpy arrays
                control_gt = sample["output_control"].cpu().detach().numpy().reshape(-1)
                control_prediction = prediction["output_control"].cpu().detach().numpy().reshape(-1)

                # get the input images (for now there will only be one)
                input_images = []
                for key in sorted(sample["original"]):
                    if key.startswith("input_image"):
                        input_images.append(sample["original"][key])

                # create the new frame and write it
                frame = create_frame(fig, ax, control_gt, control_prediction, input_images)
                for _ in range(config["slow_down_factor"]):
                    video_writer_dict[run_dir].write(frame)

        for _, vr in video_writer_dict.items():
            vr.release()


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
    model = C3DRegressor()
    model.load_state_dict(model_info["model_state_dict"])
    model = model.to(device)
    model.eval()

    # load the dataframe with the info
    df_frame_info = pd.read_csv(frame_info_path)

    # create the dataset
    root_dir = os.path.join(run_dir, os.pardir, os.pardir)
    run_info = parse_run_info(run_dir)
    sub_index = (0, -1)
    run_split = "train"
    for split in ["train", "val", "test"]:
        df_test = pd.read_csv(os.path.join(root_dir, f"turn_left_drone_control_gt_{split}.csv"))
        if len(df_test[(df_test["subject"] == run_info["subject"]) & (df_test["run"] == run_info["run"])].index) != 0:
            # found the right file, get the start and end index
            temp = df_test[(df_test["subject"] == run_info["subject"]) & (df_test["run"] == run_info["run"])]

            start_index = temp.index[0]
            end_index = temp.index[-1] + 1

            sub_index = (start_index, end_index)
            run_split = split

            break

    video_dataset = get_dataset(root_dir, split=run_split, data_type="turn_left_drone_control_gt",
                                resize_height=(122, 122), use_pims=args.use_pims, sub_index=sub_index)
    video_dataset.return_original = True

    # determine size and where to position things based on the output mode
    output_size = (800 * 2, 600)

    # extract FPS and codec information and create the video writer
    if run_dir not in video_dataset.cap_dict:
        video_capture = cv2.VideoCapture(os.path.join(run_dir, "screen.mp4"))
    else:
        video_capture = video_dataset.cap_dict[os.path.join(run_dir)]["data"]
    fps, fourcc = (video_capture.get(i) for i in range(5, 7))
    video_name = f"{args.model_path[-11:-3]}_drone_control_gt_comparison_test.mp4"
    video_writer = cv2.VideoWriter(os.path.join(save_dir, video_name), int(fourcc), fps, output_size, True)

    # loop through all frames in frame_info
    for _, row in tqdm(df_frame_info.iterrows(), total=len(df_frame_info.index)):
        # TODO: ideally need some way to identify the sequences that are actually usable
        #  => might be better to not even use screen_info (should be removed anyway) and instead maybe access
        #     the sequences information used for the creation of the dataset anyway
        #  of course this presupposes that full sequences are used for one dataset...

        # filter frames if specified
        if args.filter is not None and row[args.filter] != 1:
            continue

        if len(video_dataset.df_index.loc[video_dataset.df_index["frame"] == row["frame"]]) == 0:
            continue

        if video_dataset.df_index.loc[video_dataset.df_index["frame"] == row["frame"], "stack_index"].values[0] == -1:
            continue

        # print(row["frame"])
        # print(video_dataset.df_index.loc[video_dataset.df_index["frame"] == row["frame"], "stack_index"].values[0])
        index = video_dataset.df_index.loc[video_dataset.df_index["frame"] == row["frame"], "stack_index"].values[0]
        # dataset_index = video_dataset.df_index.index[video_dataset.df_index["frame"] == row["frame"]].values[0]
        # print(video_dataset.df_index["stack_index"].iloc[dataset_index:(dataset_index + 50)])>

        # return original frame and predict label
        frame, label, frame_original, _ = video_dataset[index]
        frame, label = frame.unsqueeze(0).to(device), label.unsqueeze(0).to(device) if label is not None else label
        predicted_label = model(frame)

        # convert to numpy for easier processing
        label = label.cpu().detach().numpy().reshape(-1)
        predicted_label = predicted_label.cpu().detach().numpy().reshape(-1)

        # downscale original frame to match the size of the labels => no need for that now
        frame = frame_original[-1]
        if args.use_pims:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # plot in numpy and convert to opencv-compatible image
        x = np.arange(len(label))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        rects1 = ax.bar(x - width / 2, label, width, label='GT')
        rects2 = ax.bar(x + width / 2, predicted_label, width, label='pred')
        ax.set_ylim(-1, 1)
        ax.set_xticks(x)
        ax.axhline(c="k", lw=0.2)
        ax.legend()
        fig.tight_layout()

        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # stack the frames/labels for the video
        new_frame = np.hstack((frame, plot_image))
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)

        # write the newly created frame to file
        for _ in range(args.slow_down_factor):
            video_writer.write(new_frame)

        # if row["frame"] >= 500:
        #     break


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


def parse_config(args):
    config = vars(args)
    config["split_config"] = resolve_split_index_path(config["split_config"], config["data_root"])
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-m", "--model_load_path", type=str, default=None,
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-s", "--split", type=str, nargs="+", default=["val"], choices=["train", "val", "test"],
                        help="Splits for which to create videos.")
    parser.add_argument("-sc", "--split_config", type=str, default=0,
                        help="TODO.")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="The GPU to use.")
    parser.add_argument("-sub", "--subjects", type=int, nargs="*", default=[],
                        help="Subjects to use.")
    parser.add_argument("-run", "--runs", type=int, nargs="*", default=[],
                        help="Runs to use.")
    parser.add_argument("-f", "--filter", type=str, default=None, choices=["turn_left", "turn_right"],
                        help="'Property' by which to filter frames (only left/right turn for now).")
    parser.add_argument("-tn", "--track_name", type=str, default="flat",
                        help="The name of the track.")
    parser.add_argument("-gtn", "--ground_truth_name", type=str, default="moving_window_gt",
                        help="The name of the ground-truth video.")
    parser.add_argument("-om", "--output_mode", type=str, default="overlay_maps",
                        choices=["overlay_maps", "overlay_all", "overlay_none", "overlay_raw"],
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-sf", "--slow_down_factor", type=int, default=1,
                        help="Factor by which the output video is slowed down (frames are simply saved multiple times).")
    parser.add_argument("--use_pims", action="store_true",
                        help="Whether to use PIMS (PyAV) instead of OpenCV for reading frames.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    # main(arguments)
    test(parse_config(arguments))

