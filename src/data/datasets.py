import os
import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm import tqdm
from src.data.utils import get_indexed_reader, resolve_split_index_path, run_info_to_path
from src.data.constants import STATISTICS, HIGH_LEVEL_COMMAND_LABEL


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
        input_statistics = {}
        for i in self.input_names:
            if "mean" in split_index_info and "std" in split_index_info:
                input_statistics[i] = {
                    "mean": split_index_info["mean"][i] if i in split_index_info["mean"] else STATISTICS["mean"][i],
                    "std": split_index_info["std"][i] if i in split_index_info["std"] else STATISTICS["std"][i]
                }
            else:
                input_statistics[i] = {
                    "mean": STATISTICS["mean"][i],
                    "std": STATISTICS["std"][i]
                }
        self.input_transforms = [transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
        ]) for i in self.input_names]
        self.output_transform = transforms.Compose([
            ImageToAttentionMap(),
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            MakeValidDistribution()
        ])

        self.video_readers = {}
        """
        self.index["run_dir"] = None
        unique_run_info = self.index.groupby(["track_name", "subject", "run"]).size().reset_index()
        print("Preparing video readers!")
        for _, row in tqdm(unique_run_info.iterrows()):
            run_dir = os.path.join(self.data_root, run_info_to_path(row["subject"], row["run"], row["track_name"]))
            if run_dir not in self.video_readers:
                self.video_readers[run_dir] = {
                    f"input_image_{idx}": get_indexed_reader(os.path.join(run_dir, f"{i}.mp4"))
                    for idx, i in enumerate(self.input_names)
                }
                self.video_readers[run_dir]["output_attention"] = get_indexed_reader(os.path.join(
                    run_dir, f"{self.output_name}.mp4"))
            self.index.loc[(self.index["track_name"] == row["track_name"])
                           & (self.index["subject"] == row["subject"])
                           & (self.index["run"] == row["run"]), "run_dir"] = run_dir
        """

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, item):
        # get the information about the current item
        current_row = self.index.iloc[item]
        current_frame_index = current_row["frame"]
        # current_run_dir = current_row["run_dir"]
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
        out = {"original": {f"input_image_{idx}": np.array(i.copy()) for idx, i in enumerate(inputs)}}
        out["original"]["output_attention"] = np.array(output.copy())

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
        input_statistics = {}
        for i in self.input_names:
            if "mean" in split_index_info and "std" in split_index_info:
                input_statistics[i] = {
                    "mean": split_index_info["mean"][i] if i in split_index_info["mean"] else STATISTICS["mean"][i],
                    "std": split_index_info["std"][i] if i in split_index_info["std"] else STATISTICS["std"][i]
                }
            else:
                input_statistics[i] = {
                    "mean": STATISTICS["mean"][i],
                    "std": STATISTICS["std"][i]
                }
        self.input_transforms = [transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
        ]) for i in self.input_names]

        self.video_readers = {}
        """
        self.index["run_dir"] = None
        unique_run_info = self.index.groupby(["track_name", "subject", "run"]).size().reset_index()
        for _, row in unique_run_info.iterrows():
            run_dir = os.path.join(self.data_root, run_info_to_path(row["subject"], row["run"], row["track_name"]))
            if run_dir not in self.video_readers:
                self.video_readers[run_dir] = {
                    f"input_image_{idx}": get_indexed_reader(os.path.join(run_dir, f"{i}.mp4"))
                    for idx, i in enumerate(self.input_names)
                }
            self.index.loc[(self.index["track_name"] == row["track_name"])
                           & (self.index["subject"] == row["subject"])
                           & (self.index["run"] == row["run"]), "run_dir"] = run_dir
        """

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, item):
        # get the information about the current item
        current_row = self.index.iloc[item]
        current_frame_index = current_row["frame"]
        # current_run_dir = current_row["run_dir"]
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
        out = {"original": {f"input_image_{idx}": np.array(i.copy()) for idx, i in enumerate(inputs)}}

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

        self.index = frame_index[frame_index["split"] == self.split].copy()

        self.index["label"] = 4
        for idx, (track_name, half) in enumerate([("flat", "left_half"), ("flat", "right_half"),
                                                  ("wave", "left_half"), ("wave", "right_half")]):
            self.index.loc[(self.index["track_name"] == track_name) & (self.index[half] == 1), "label"] = idx

        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        input_statistics = {}
        for i in self.video_input_names:
            if "mean" in split_index_info and "std" in split_index_info:
                input_statistics[i] = {
                    "mean": split_index_info["mean"][i] if i in split_index_info["mean"] else STATISTICS["mean"][i],
                    "std": split_index_info["std"][i] if i in split_index_info["std"] else STATISTICS["std"][i]
                }
            else:
                input_statistics[i] = {
                    "mean": STATISTICS["mean"][i],
                    "std": STATISTICS["std"][i]
                }
        self.video_input_transforms = [transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
        ]) for i in self.video_input_names]

        self.video_readers = {}
        """
        self.index["run_dir"] = None
        unique_run_info = self.index.groupby(["track_name", "subject", "run"]).size().reset_index()
        print("Preparing video readers!")
        for _, row in tqdm(unique_run_info.iterrows()):
            run_dir = os.path.join(self.data_root, run_info_to_path(row["subject"], row["run"], row["track_name"]))
            if run_dir not in self.video_readers:
                self.video_readers[run_dir] = {
                    f"input_image_{idx}": get_indexed_reader(os.path.join(run_dir, f"{i}.mp4"))
                    for idx, i in enumerate(self.video_input_names)
                }
            self.index.loc[(self.index["track_name"] == row["track_name"])
                           & (self.index["subject"] == row["subject"])
                           & (self.index["run"] == row["run"]), "run_dir"] = run_dir
        """

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, item):
        # get the information about the current item
        current_row = self.index.iloc[item]
        current_frame_index = current_row["frame"]
        # current_run_dir = current_row["run_dir"]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_row["subject"],
                                                                        current_row["run"],
                                                                        current_row["track_name"]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_readers:
            # print(os.path.join(current_run_dir, f"{self.video_input_names[1]}.mp4"))
            self.video_readers[current_run_dir] = {
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.video_input_names)
            }

        # read the input frames
        video_inputs = [self.video_readers[current_run_dir][f"input_image_{idx}"][current_frame_index]
                        for idx in range(len(self.video_input_names))]

        # read the state input
        state_input = self.index[self.state_input_names].iloc[item].values

        # determine the "high level command" label for the sample
        high_level_label = self.index["label"].iloc[item]

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
        out["label_high_level"] = high_level_label

        # return a dictionary
        return out


class ImageToAttentionAndControlDataset(Dataset):
    pass


class StackedImageToAttentionDataset(Dataset):

    def __init__(self, config, split):
        super().__init__()
        self.data_root = config["data_root"]
        self.input_names = config["input_video_names"]
        self.output_name = config["ground_truth_name"]
        self.split = split
        self.stack_size = config["stack_size"]

        frame_index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))
        split_index = pd.read_csv(config["split_config"] + ".csv")
        frame_index["split"] = split_index
        self.index = frame_index[frame_index["split"] == self.split].copy().reset_index(drop=True)

        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        input_statistics = {}
        for i in self.input_names:
            if "mean" in split_index_info and "std" in split_index_info:
                input_statistics[i] = {
                    "mean": split_index_info["mean"][i] if i in split_index_info["mean"] else STATISTICS["mean"][i],
                    "std": split_index_info["std"][i] if i in split_index_info["std"] else STATISTICS["std"][i]
                }
            else:
                input_statistics[i] = {
                    "mean": STATISTICS["mean"][i],
                    "std": STATISTICS["std"][i]
                }
        self.input_transforms = [transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
        ]) for i in self.input_names]
        self.output_transform = transforms.Compose([
            ImageToAttentionMap(),
            transforms.ToPILImage(),
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            MakeValidDistribution()
        ])
        # TODO: MIGHT NEED TO ADD ADDITIONAL STREAM/TRANSFORM/WHATEVER FOR
        #  THE CROPPED STREAM OF THE COARSE-REFINE ARCHITECTURE

        self.video_readers = {}
        """
        self.index["run_dir"] = None
        unique_run_info = self.index.groupby(["track_name", "subject", "run"]).size().reset_index()
        for _, row in unique_run_info.iterrows():
            run_dir = os.path.join(self.data_root, run_info_to_path(row["subject"], row["run"], row["track_name"]))
            if run_dir not in self.video_readers:
                self.video_readers[run_dir] = {
                    f"input_stack_{idx}": get_indexed_reader(os.path.join(run_dir, f"{i}.mp4"))
                    for idx, i in enumerate(self.input_names)
                }
                self.video_readers[run_dir]["output_attention"] = get_indexed_reader(os.path.join(
                    run_dir, f"{self.output_name}.mp4"))
            self.index.loc[(self.index["track_name"] == row["track_name"])
                           & (self.index["subject"] == row["subject"])
                           & (self.index["run"] == row["run"]), "run_dir"] = run_dir
        """

        # find contiguous sequences of frames
        sequences = []
        frames = self.index["frame"]
        jumps = (frames - frames.shift()) != 1
        frames = list(frames.index[jumps]) + [frames.index[-1] + 1]
        for i, start_index in enumerate(frames[:-1]):
            # note that the range [start_frame, end_frame) is exclusive
            sequences.append((start_index, frames[i + 1]))

        # need some way to not mess up the indexing, but also cannot remove data
        # maybe use second "index" that skips the (stack_size - 1) first frames of each contiguous sequence?
        self.index["stack_index"] = -1
        df_col_index = self.index.columns.get_loc("stack_index")
        total_num_frame_stacks = 0
        for start_index, end_index in sequences:
            num_frames_seq = end_index - start_index
            num_frame_stacks = num_frames_seq - self.stack_size + 1
            if end_index - start_index < self.stack_size:
                continue

            # assign index over entire dataframe to "valid" frames
            index = np.arange(num_frame_stacks) + total_num_frame_stacks
            # print(len(index))
            # print(end_index - (start_index + self.stack_size - 1))
            # print(len(self.index.iloc[(start_index + self.stack_size - 1):end_index, df_col_index].index))
            self.index.iloc[(start_index + self.stack_size - 1):end_index, df_col_index] = index

            # keep track of the current number of stacks (highest index)
            total_num_frame_stacks += num_frame_stacks

        if "dreyeve_transforms" in config and config["dreyeve_transforms"]:
            # define the following transforms
            # - loaded stack to 112x112
            # - load last frame to 448x448
            # - loaded stack to 256x256 => random crop to 112x112
            self.input_stack_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
            ])
            self.input_last_frame_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
            ])
            self.input_stack_crop_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"]),
                transforms.RandomCrop((112, 112))
            ])

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, item):
        # get the information about the current item
        current_stack_index = self.index.index[self.index["stack_index"] == item].values[0]
        # current_run_dir = self.index["run_dir"].iloc[current_stack_index]
        current_stack_df = self.index.iloc[(current_stack_index - self.stack_size + 1):(current_stack_index + 1)]
        current_frame_index = current_stack_df["frame"].iloc[-1]
        current_run_dir = os.path.join(self.data_root, run_info_to_path(current_stack_df["subject"].iloc[0],
                                                                        current_stack_df["run"].iloc[0],
                                                                        current_stack_df["track_name"].iloc[0]))

        # initialise the video readers if necessary
        if current_run_dir not in self.video_readers:
            # print(os.path.join(current_run_dir, f"{self.video_input_names[1]}.mp4"))
            self.video_readers[current_run_dir] = {
                f"input_stack_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.input_names)
            }
            self.video_readers[current_run_dir]["output_attention"] = get_indexed_reader(os.path.join(
                current_run_dir, f"{self.output_name}.mp4"))

        # read the frames
        inputs = [[] for _ in self.input_names]
        inputs_original = [[] for _ in self.input_names]
        for idx, _ in enumerate(self.input_names):
            for _, row in current_stack_df.iterrows():
                # TODO: need to re-add the thing
                frame = self.video_readers[current_run_dir][f"input_stack_{idx}"][row["frame"]]
                frame_original = frame.copy()

                # apply input transform here already
                frame = self.input_transforms[idx](frame)

                inputs[idx].append(frame)
                inputs_original[idx].append(frame_original)

            # stack the frames
            # TODO: not sure if this is actually the right dimension to stack them in
            inputs[idx] = torch.stack(inputs[idx], 1)
            inputs_original[idx] = np.stack(inputs_original[idx], 0)

        output = self.video_readers[current_run_dir]["output_attention"][current_frame_index]

        # start constructing the output dictionary => keep the original frames in there
        # out = {"original": {f"input_stack_{idx}": np.array(i.copy()) for idx, i in enumerate(inputs)}}
        out = {"original": {}}
        for idx, _ in enumerate(self.input_names):
            out["original"][f"input_stack_{idx}"] = np.array(inputs_original[idx])
        out["original"]["output_attention"] = np.array(output)

        # apply transform only for the output
        for idx, i in enumerate(inputs):
            out[f"input_stack_{idx}"] = i
        out["output_attention"] = self.output_transform(output)

        # return a dictionary
        return out


if __name__ == "__main__":
    test_config = {
        "data_root": os.getenv("GAZESIM_ROOT"),
        "input_video_names": ["screen", "hard_mask_moving_window_frame_mean_gt"],
        "ground_truth_name": "moving_window_frame_mean_gt",  # "drone_control_frame_mean_gt",
        "drone_state_names": ["DroneVelocityX", "DroneVelocityY", "DroneVelocityZ"],
        "resize": 150,
        "split_config": resolve_split_index_path(10, data_root=os.getenv("GAZESIM_ROOT")),
        "stack_size": 4
    }

    dataset = StackedImageToAttentionDataset(test_config, split="train")
    print(len(dataset))
    print(dataset.index.columns)

    sample = dataset[0]
    print(sample.keys())
    print(sample["original"].keys())

    print(sample["input_stack_0"].shape)
    print(sample["input_stack_1"].shape)
    print(sample["output_attention"].shape)
