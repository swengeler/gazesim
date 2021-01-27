import os
import json
import numpy as np
import pandas as pd
import matplotlib.style as style
import cv2
import torch

from tqdm import tqdm
from gazesim.data.utils import find_contiguous_sequences, resolve_split_index_path, run_info_to_path
from gazesim.training.config import parse_config as parse_train_config
from gazesim.training.helpers import resolve_model_class, resolve_dataset_class
from gazesim.training.utils import to_device, to_batch
from gazesim.training.helpers import resolve_output_processing_func
from gazesim.models.utils import image_softmax

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


def main(config):
    # "formatting"
    output_size = (800, 600)  # TODO: change the other stuff too
    positions = {}
    if config["output_mode"] == "overlay_maps":
        output_size = (800 * 2, 600 + 100)
        positions["ground_truth"] = (10, 40)
        positions["prediction"] = (10, 75)
        positions["valid_lap"] = (310, 40)
        positions["expected_trajectory"] = (310, 75)
        positions["turn_left"] = (610, 40)
        positions["turn_right"] = (610, 75)
    elif config["output_mode"] == "overlay_all":
        output_size = (800, 600 + 300)
        positions["ground_truth"] = (10, 40)
        positions["prediction"] = (10, 75)
        positions["valid_lap"] = (10, 140)
        positions["expected_trajectory"] = (10, 175)
        positions["turn_left"] = (10, 240)
        positions["turn_right"] = (10, 275)
    elif config["output_mode"] == "overlay_none":
        output_size = (800 * 3, 600 + 100)
        positions["ground_truth"] = (410, 75)
        positions["prediction"] = (810, 75)
        positions["valid_lap"] = (10, 40)
        positions["expected_trajectory"] = (10, 75)
        positions["turn_left"] = (410, 40)
        positions["turn_right"] = (810, 40)
    elif config["output_mode"] == "overlay_simple":
        output_size = (800, 600 + 100)
        positions["ground_truth"] = (10, 40)
        positions["prediction"] = (10, 75)

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
    train_config["input_video_names"] = [config["video_name"]]

    # load the model
    model_info = train_config["model_info"]
    model = resolve_model_class(train_config["model_name"])(train_config)
    model.load_state_dict(model_info["model_state_dict"])
    model = model.to(device)
    model.eval()

    # TODO: apply filter (can include split, subject, run I guess, not sure that lap would make sense)
    #  => actually, if we wanted to just extract a single video for one lap, it would probably make sense...
    # but if the default is that everything is used... then there is a bit of an issue for labeling complete sequences
    # could either ignore that if it is the case, complain/skip if there is overlap in a single sequence or just
    # use a more "rigorous" structure, where we always filter by split => probably the latter

    for split in config["split"]:
        current_frame_index = frame_index.loc[split_index["split"] == split]
        current_dataset = resolve_dataset_class(train_config["dataset_name"])(train_config, split=split)
        sequences = find_contiguous_sequences(current_frame_index, new_index=True)
        # TODO: do lower FPS stuff somehow => can probably just do [::frame_skip] and then make sure that the
        #  difference between frames is == frame_skip

        video_writer_dict = {}
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
            if current_run_dir != run_dir:
                video_writer_dict[run_dir].release()
                run_dir = current_run_dir

            if run_dir not in video_writer_dict:
                video_dir = os.path.join(save_dir, run_dir)
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)
                # TODO: also input video name
                video_capture = cv2.VideoCapture(os.path.join(config["data_root"], run_dir, "{}.mp4".format(config["video_name"])))
                fps, fourcc = (video_capture.get(i) for i in range(5, 7))
                video_name = "{}_comparison_{}_{}_{}.mp4".format("default_attention_gt", config["video_name"], config["output_mode"], split)
                video_writer = cv2.VideoWriter(os.path.join(video_dir, video_name), int(fourcc), fps, output_size, True)
                video_writer_dict[run_dir] = video_writer

            for index in tqdm(range(start_index, end_index), disable=True):
                # read the current data sample
                sample = to_batch([current_dataset[index]])
                sample = to_device(sample, device)

                # compute the loss
                prediction = model(sample)
                prediction["output_attention"] = image_softmax(resolve_output_processing_func(
                    "output_attention")(prediction["output_attention"]))

                # get the values as numpy arrays
                attention_gt = sample["output_attention"].cpu().detach().numpy().squeeze()
                attention_prediction = prediction["output_attention"].cpu().detach().numpy().squeeze()

                # get the original frame as a numpy array (also convert color for OpenCV)
                frame = sample["original"]["input_image_0"].cpu().detach().numpy().squeeze()
                frame = (frame * 255.0).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # stack greyscale labels to become RGB
                attention_gt = np.repeat(attention_gt[np.newaxis, :, :], 3, axis=0).transpose((1, 2, 0))
                attention_prediction = np.repeat(attention_prediction[np.newaxis, :, :], 3, axis=0).transpose((1, 2, 0))

                # normalise using the maximum of the maxima of either label and convert to [0, 255] scale
                norm_max = max([attention_gt.max(), attention_prediction.max()])
                if norm_max != 0:
                    attention_gt /= norm_max
                    attention_prediction /= norm_max
                attention_gt = (attention_gt * 255).astype("uint8")
                attention_prediction = (attention_prediction * 255).astype("uint8")

                # set all but one colour channel for GT and predicted labels to 0
                attention_gt[:, :, 1:] = 0
                attention_prediction[:, :, :-1] = 0

                # scale the attention maps to the right size
                attention_gt = cv2.resize(attention_gt, (800, 600))
                attention_prediction = cv2.resize(attention_prediction, (800, 600))

                # stack the frames/labels for the video
                temp = np.zeros(output_size[::-1] + (3,), dtype="uint8")
                if config["output_mode"] == "overlay_maps":
                    combined_labels = cv2.addWeighted(attention_gt, 0.5, attention_prediction, 0.5, 0)
                    new_frame = np.hstack((frame, combined_labels))
                    temp[100:, :, :] = new_frame
                elif config["output_mode"] == "overlay_all":
                    frame = np.repeat(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, axis=2)
                    combined_labels = cv2.addWeighted(attention_gt, 0.5, attention_prediction, 0.5, 0)
                    new_frame = cv2.addWeighted(frame, 0.5, combined_labels, 0.5, 0)
                    temp[300:, :, :] = new_frame
                elif config["output_mode"] == "overlay_none":
                    new_frame = np.hstack((frame, attention_gt, attention_prediction))
                    temp[100:, :, :] = new_frame
                elif config["output_mode"] == "overlay_simple":
                    frame = np.repeat(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, axis=2)
                    combined_labels = cv2.addWeighted(attention_gt, 0.5, attention_prediction, 0.5, 0)
                    new_frame = cv2.addWeighted(frame, 0.3, combined_labels, 0.7, 0)
                    temp[100:, :, :] = new_frame
                new_frame = temp

                # add the other information we want to display
                cv2.putText(new_frame, "Ground-truth", positions["ground_truth"], cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)
                cv2.putText(new_frame, "Prediction", positions["prediction"], cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)

                # write the image (possibly multiple times for slowing the video down)
                for _ in range(config["slow_down_factor"]):
                    video_writer_dict[run_dir].write(new_frame)

        for _, vr in video_writer_dict.items():
            vr.release()


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
    parser.add_argument("-vn", "--video_name", type=str, default="screen",
                        help="The name of the input video.")
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
    parser.add_argument("-om", "--output_mode", type=str, default="overlay_maps",
                        choices=["overlay_maps", "overlay_all", "overlay_none", "overlay_simple"],
                        help="The path to the model checkpoint to use for computing the predictions.")
    parser.add_argument("-sf", "--slow_down_factor", type=int, default=1,
                        help="Factor by which the output video is slowed down (frames are simply saved multiple times).")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    main(parse_config(arguments))
