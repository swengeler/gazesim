import os
import cv2
import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.utils import get_indexed_reader, resolve_split_index_path, run_info_to_path


class ImageToAttentionMap(object):

    def __call__(self, sample):
        # for now always expects numpy array (and respective dimension for the colour channels)
        return sample[:, :, 0]


class MakeValidDistribution(object):

    def __call__(self, sample):
        # expects torch tensor with
        pixel_sum = torch.sum(sample, [])
        if pixel_sum > 0.0:
            sample = sample / pixel_sum
        return sample


class ImageDataset(Dataset):
    # maybe have some generic methods to get some frame from some video?
    # maybe some data structures to keep track of videos to load from?
    pass


class ImageToAttentionDataset(Dataset):

    def __init__(self, config, split):
        super().__init__()
        # TODO: before re-implementing these classes, should probably figure out in what way train/val/test splits
        #  are going to be represented and (probably more importantly) how they will be generated...
        #  => for now assume that train/val/test splits are already defined in some split index files
        #     and do not have to be generated dynamically (this could only be done leaving away
        #     normalisation or using global - not training set - statistics)

        # what information is required for this
        # - input video name(s)
        # - output video name
        # - split index file path or index
        # - which split to actually use
        # - do we want/need a subindex? guess I'll leave it out for now...
        # - should the data/label transform be defined in the class itself or outside it?
        #   => considering that we load the dataset statistics alongside the split index, probably inside it
        self.data_root = config["data_root"]
        self.input_names = config["input_video_names"]
        self.output_name = config["ground_truth_name"]
        self.split = split

        frame_index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))
        split_index = pd.read_csv(config["split_config"] + ".csv")
        frame_index["split"] = split_index
        self.index = frame_index[frame_index["split"] == self.split]
        # TODO: should it be possible to filter here as well? right now I'm just doing the filtering in the
        #  creation of the split index files, which makes things cleaner for statistics and mean masks
        #  => I think it's best to leave it at that for now; if I decide to e.g. compute "global" statistics
        #     at some point (which is not conceptually correct I think), then this would be an option to make
        #     things more flexible (without having to generate splits in advance as well)

        # TODO: if there is no specific information for mean/std for a video, should default to something
        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        self.input_transforms = [transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(split_index_info["mean"][i], split_index_info["std"][i])
        ]) for i in self.input_names]
        self.output_transform = transforms.Compose([
            ImageToAttentionMap(),
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            MakeValidDistribution()
        ])

        self.video_readers = {}

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, item):
        # get the information about the current item
        current_row = self.index.iloc[item]
        current_frame_index = current_row["frame"]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_row["subject"],
                                                                        current_row["run"],
                                                                        current_row["track_name"]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_readers:
            self.video_readers[current_run_dir] = {
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.input_names)
            }
            self.video_readers[current_run_dir]["output_attention"] = get_indexed_reader(os.path.join(
                current_run_dir, f"{self.output_name}.mp4"))

        # read the frames
        inputs = [self.video_readers[current_run_dir][f"input_image_{idx}"][current_frame_index]
                  for idx in range(len(self.input_names))]
        output = self.video_readers[current_run_dir]["output_attention"][current_frame_index]

        # start constructing the output dictionary => keep the original frames in there
        out = {"original": {f"input_image_{idx}": i.copy() for idx, i in enumerate(inputs)}}
        out["original"]["output_attention"] = output.copy()

        # apply transforms to the inputs and output
        for idx, i in enumerate(inputs):
            out[f"input_image_{idx}"] = self.input_transforms[idx](i)
        out["output_attention"] = self.output_transform(output)

        # return a dictionary
        return out


class ImageToControlDataset(Dataset):

    def __init__(self, config, split):
        super().__init__()
        self.data_root = config["data_root"]
        self.input_names = config["input_video_names"]
        self.output_name = config["ground_truth_name"]
        self.split = split
        self.output_columns = []

        frame_index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))
        split_index = pd.read_csv(config["split_config"] + ".csv")
        frame_index["split"] = split_index

        # TODO: could have check of GT availability here but probably better to do in generate_splits.py
        ground_truth_index = pd.read_csv(os.path.join(self.data_root, "index", "control_gt.csv"))
        ground_truth = pd.read_csv(os.path.join(self.data_root, "index", f"{self.output_name}.csv"))
        for col in ground_truth:
            frame_index[col] = ground_truth[col]
            self.output_columns.append(col)

        self.index = frame_index[frame_index["split"] == self.split]

        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        self.input_transforms = [transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(split_index_info["mean"][i], split_index_info["std"][i])
        ]) for i in self.input_names]

        self.video_readers = {}

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, item):
        # get the information about the current item
        current_row = self.index.iloc[item]
        current_frame_index = current_row["frame"]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_row["subject"],
                                                                        current_row["run"],
                                                                        current_row["track_name"]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_readers:
            self.video_readers[current_run_dir] = {
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.input_names)
            }

        # read the input frames
        inputs = [self.video_readers[current_run_dir][f"input_image_{idx}"][current_frame_index]
                  for idx in range(len(self.input_names))]

        # extract the control GT from the dataframe
        output = self.index[self.output_columns].iloc[item].values

        # start constructing the output dictionary => keep the original frames in there
        out = {"original": {f"input_image_{idx}": i.copy() for idx, i in enumerate(inputs)}}

        # apply transforms to the inputs and output
        for idx, i in enumerate(inputs):
            out[f"input_image_{idx}"] = self.input_transforms[idx](i)
        out["output_control"] = torch.from_numpy(output).float()

        # return a dictionary
        return out


class ImageAndStateToControlDataset(Dataset):

    def __init__(self, config, split):
        super().__init__()
        self.data_root = config["data_root"]
        self.video_input_names = config["input_video_names"]
        self.state_input_names = config["drone_state_names"]
        self.output_name = config["ground_truth_name"]
        self.output_columns = []
        self.split = split

        frame_index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))
        split_index = pd.read_csv(config["split_config"] + ".csv")
        frame_index["split"] = split_index

        # TODO: could have check of GT availability here but probably better to do in generate_splits.py
        ground_truth_index = pd.read_csv(os.path.join(self.data_root, "index", "control_gt.csv"))
        ground_truth = pd.read_csv(os.path.join(self.data_root, "index", f"{self.output_name}.csv"))
        for col in ground_truth:
            frame_index[col] = ground_truth[col]
            self.output_columns.append(col)

        drone_state = pd.read_csv(os.path.join(self.data_root, "index", "state.csv"))
        for col in self.state_input_names:
            frame_index[col] = drone_state[col]

        self.index = frame_index[frame_index["split"] == self.split]

        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        self.video_input_transforms = [transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(split_index_info["mean"][i], split_index_info["std"][i])
        ]) for i in self.video_input_names]

        self.video_readers = {}

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, item):
        # get the information about the current item
        current_row = self.index.iloc[item]
        current_frame_index = current_row["frame"]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_row["subject"],
                                                                        current_row["run"],
                                                                        current_row["track_name"]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_readers:
            self.video_readers[current_run_dir] = {
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.video_input_names)
            }

        # read the input frames
        video_inputs = [self.video_readers[current_run_dir][f"input_image_{idx}"][current_frame_index]
                        for idx in range(len(self.video_input_names))]

        # read the state input
        state_input = self.index[self.state_input_names].iloc[item].values

        # extract the control GT from the dataframe
        output = self.index[self.output_columns].iloc[item].values

        # start constructing the output dictionary => keep the original frames in there
        out = {"original": {f"input_image_{idx}": np.array(i.copy()) for idx, i in enumerate(video_inputs)}}

        # apply transforms to the inputs and output
        for idx, i in enumerate(video_inputs):
            # TODO: maybe rename these to video_input_X or image_input_X in all classes?
            out[f"input_image_{idx}"] = self.video_input_transforms[idx](i)
        out["input_state"] = torch.from_numpy(state_input).float()
        out["output_control"] = torch.from_numpy(output).float()

        # return a dictionary
        return out


class ImageToAttentionAndControlDataset(Dataset):
    pass


if __name__ == "__main__":
    test_config = {
        "data_root": os.getenv("GAZESIM_ROOT"),
        "input_video_names": ["screen"],
        "ground_truth_name": "drone_control_frame_mean_gt",
        "drone_state_names": ["DroneVelocityX", "DroneVelocityY", "DroneVelocityZ"],
        "resize": 300,
        "split_config": 0
    }

    dataset = ImageAndStateToControlDataset(test_config, split="train")
    print(len(dataset))
    print(dataset.index.columns)

    sample = dataset[0]
    print(sample.keys())
    print(sample["original"].keys())
