import os
import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.utils import get_indexed_reader, resolve_split_index_path, run_info_to_path
from src.data.transforms import MakeValidDistribution, ImageToAttentionMap, ManualRandomCrop
from src.data.constants import STATISTICS


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
            if config["no_normalisation"]:
                input_statistics[i] = {"mean": np.array([0, 0, 0]), "std": np.array([1, 1, 1])}
            else:
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
            if config["no_normalisation"]:
                input_statistics[i] = {"mean": np.array([0, 0, 0]), "std": np.array([1, 1, 1])}
            else:
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
            if config["no_normalisation"]:
                input_statistics[i] = {"mean": np.array([0, 0, 0]), "std": np.array([1, 1, 1])}
            else:
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
        self.dreyeve_transforms = config["dreyeve_transforms"]

        frame_index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))
        split_index = pd.read_csv(config["split_config"] + ".csv")
        frame_index["split"] = split_index
        self.index = frame_index[frame_index["split"] == self.split].copy().reset_index(drop=True)

        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        input_statistics = {}
        for i in self.input_names:
            if config["no_normalisation"]:
                input_statistics[i] = {"mean": np.array([0, 0, 0]), "std": np.array([1, 1, 1])}
            else:
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

        self.random_crop = None
        if self.dreyeve_transforms:
            # define the following transforms
            # - loaded stack to 112x112
            # - load last frame to 448x448
            # - loaded stack to 256x256 => random crop to 112x112
            self.random_crop = ManualRandomCrop((256, 256), (112, 112))

            self.input_transforms = {
                "stack": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.input_names],
                "stack_crop": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    self.random_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"]),
                ]) for i in self.input_names],
                "last_frame": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.input_names],
            }

            self.output_transforms = {
                "stack": transforms.Compose([
                    ImageToAttentionMap(),
                    transforms.ToPILImage(),
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    MakeValidDistribution()
                ]),
                "stack_crop": transforms.Compose([
                    ImageToAttentionMap(),
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    self.random_crop,
                    transforms.ToTensor(),
                    MakeValidDistribution(),
                    # TODO: really not sure if this is proper to do (in general and particularly in this case
                    #  - in general: might actually want to learn to predict in such a way that (usually) the
                    #    upscaled output is close-ish to a valid distribution, which it might not even be if
                    #    the small version is already one
                    #  - here specifically: this might actually make problems because there will be cases where
                    #    every value is 0 and maybe cases where very few values will be non-zero => in the latter
                    #    case these values would be adjusted to be very high and I'm unsure if that's a good thing
                    #  - on the other hand: not sure if KL-divergence works (not just conceptually, but also
                    #    in practice) if this isn't properly normalised => if it doesn't work in practice, might
                    #    still want to consider not doing this normalisation when using MSE (if that ever happens)
                ])
            }
        else:
            # just a simple transform for every frame
            self.input_transforms = {
                "stack": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(config["resize"]),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.input_names]
            }
            self.output_transforms = {
                "stack": transforms.Compose([
                    ImageToAttentionMap(),
                    transforms.ToPILImage(),
                    transforms.Resize(config["resize"]),
                    transforms.ToTensor(),
                    MakeValidDistribution(),
                ])
            }
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

    def __len__(self):
        return self.index["stack_index"].max() + 1

    def __getitem__(self, item):
        # get the information about the current item
        # TODO: maybe just change the index of the dataframe? might make this particular step easier
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
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.input_names)
            }
            self.video_readers[current_run_dir]["output_attention"] = get_indexed_reader(os.path.join(
                current_run_dir, f"{self.output_name}.mp4"))

        # prepare new "crop parameters" to get a random crop consistent across different tensors
        if self.random_crop is not None:
            self.random_crop.update()

        # read the frames
        inputs = [{k: [] for k in self.input_transforms} for _ in self.input_names]
        # inputs_original = [{} for _ in self.input_names]
        for idx, _ in enumerate(self.input_names):
            for in_stack_idx, (_, row) in enumerate(current_stack_df.iterrows()):
                frame = self.video_readers[current_run_dir][f"input_image_{idx}"][row["frame"]]
                # frame_original = frame.copy()

                # apply input transform here already
                for k in self.input_transforms:
                    # TODO: for stack_crop, need to store random crop parameters and also apply them to the output
                    #  for that image... => any point in rethinking how transforms are applied? e.g. having custom
                    #  class (e.g. DrEYEveTransform) that applies everything, including the correct random crop
                    #  (see here: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
                    if k != "last_frame":
                        inputs[idx][k].append(self.input_transforms[k][idx](frame))
                    elif in_stack_idx == self.stack_size - 1:
                        inputs[idx][k] = self.input_transforms[k][idx](frame)
                # frame = self.input_transforms[idx](frame)
                # inputs[idx].append(frame)
                # inputs_original[idx].append(frame_original)

            # stack the frames
            for k in self.input_transforms:
                if k != "last_frame":
                    inputs[idx][k] = torch.stack(inputs[idx][k], 1)
            # inputs[idx] = torch.stack(inputs[idx], 1)
            # inputs_original[idx] = np.stack(inputs_original[idx], 0)

        output = self.video_readers[current_run_dir]["output_attention"][current_frame_index]

        # start constructing the output dictionary => keep the original frames in there
        # out = {"original": {f"input_stack_{idx}": np.array(i.copy()) for idx, i in enumerate(inputs)}}
        out = {"original": {}}
        # for idx, _ in enumerate(self.input_names):
        #     out["original"][f"input_stack_{idx}"] = np.array(inputs_original[idx])
        out["original"]["output_attention"] = np.array(output)

        # apply transform only for the output
        for idx, i in enumerate(inputs):
            out[f"input_image_{idx}"] = i
        # out["output_attention"] = {k: self.output_transforms[k](output) for k in self.output_transforms}
        # TODO: this is very ugly right now, need to figure out some way to change it
        out["output_attention"] = self.output_transforms["stack"](output)
        if self.dreyeve_transforms:
            out["output_attention_crop"] = self.output_transforms["stack_crop"](output)

        # return a dictionary
        return out


class StackedImageAndStateToControlDataset(Dataset):

    def __init__(self, config, split):
        super().__init__()
        self.data_root = config["data_root"]
        self.video_input_names = config["input_video_names"]
        self.state_input_names = config["drone_state_names"]
        self.output_name = config["ground_truth_name"]
        self.output_columns = []
        self.split = split
        self.stack_size = config["stack_size"]
        self.dreyeve_transforms = config["dreyeve_transforms"]

        frame_index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))
        split_index = pd.read_csv(config["split_config"] + ".csv")
        frame_index["split"] = split_index

        ground_truth = pd.read_csv(os.path.join(self.data_root, "index", f"{self.output_name}.csv"))
        for col in ground_truth:
            frame_index[col] = ground_truth[col]
            self.output_columns.append(col)

        drone_state = pd.read_csv(os.path.join(self.data_root, "index", "state.csv"))
        for col in self.state_input_names:
            frame_index[col] = drone_state[col]

        self.index = frame_index[frame_index["split"] == self.split].copy().reset_index(drop=True)

        self.index["label"] = 4
        for idx, (track_name, half) in enumerate([("flat", "left_half"), ("flat", "right_half"),
                                                  ("wave", "left_half"), ("wave", "right_half")]):
            self.index.loc[(self.index["track_name"] == track_name) & (self.index[half] == 1), "label"] = idx

        with open(config["split_config"] + "_info.json", "r") as f:
            split_index_info = json.load(f)
        input_statistics = {}
        for i in self.video_input_names:
            if config["no_normalisation"]:
                input_statistics[i] = {"mean": np.array([0, 0, 0]), "std": np.array([1, 1, 1])}
            else:
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

        self.random_crop = None
        if self.dreyeve_transforms:
            self.random_crop = ManualRandomCrop((256, 256), (112, 112))

            self.video_input_transforms = {
                "stack": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.video_input_names],
                "stack_crop": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    self.random_crop,
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"]),
                ]) for i in self.video_input_names],
                "last_frame": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.video_input_names],
            }

            # don't need any output transforms here
        else:
            # just a simple transform for every frame
            self.video_input_transforms = {
                "stack": [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(config["resize"]),
                    transforms.ToTensor(),
                    transforms.Normalize(input_statistics[i]["mean"], input_statistics[i]["std"])
                ]) for i in self.video_input_names]
            }

        self.video_readers = {}

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
            self.index.iloc[(start_index + self.stack_size - 1):end_index, df_col_index] = index

            # keep track of the current number of stacks (highest index)
            total_num_frame_stacks += num_frame_stacks

    def __len__(self):
        return self.index["stack_index"].max() + 1

    def __getitem__(self, item):
        # get the information about the current item
        # TODO: maybe just change the index of the dataframe? might make this particular step easier
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
                f"input_image_{idx}": get_indexed_reader(os.path.join(current_run_dir, f"{i}.mp4"))
                for idx, i in enumerate(self.video_input_names)
            }

        # prepare new "crop parameters" to get a random crop consistent across different tensors
        if self.random_crop is not None:
            self.random_crop.update()

        # read the frames
        video_inputs = [{k: [] for k in self.video_input_transforms} for _ in self.video_input_names]
        # inputs_original = [{} for _ in self.input_names]
        for idx, _ in enumerate(self.video_input_names):
            for in_stack_idx, (_, row) in enumerate(current_stack_df.iterrows()):
                frame = self.video_readers[current_run_dir][f"input_image_{idx}"][row["frame"]]

                # apply input transform here already
                for k in self.video_input_transforms:
                    if k != "last_frame":
                        video_inputs[idx][k].append(self.video_input_transforms[k][idx](frame))
                    elif in_stack_idx == self.stack_size - 1:
                        video_inputs[idx][k] = self.video_input_transforms[k][idx](frame)

            # stack the frames
            for k in self.video_input_transforms:
                if k != "last_frame":
                    video_inputs[idx][k] = torch.stack(video_inputs[idx][k], 1)

        # read the state input
        state_input = current_stack_df[self.state_input_names].iloc[-1].values

        # determine the "high level command" label for the sample
        high_level_label = current_stack_df["label"].iloc[-1]

        # extract the control GT from the dataframe
        output = current_stack_df[self.output_columns].iloc[-1].values

        # start constructing the output dictionary, left empty for now
        # out = {"original": {}} => strange interaction with get_batch_size if left empty
        out = {}

        # apply transform only for the output
        for idx, i in enumerate(video_inputs):
            out[f"input_image_{idx}"] = i
        out["input_state"] = torch.from_numpy(state_input).float()
        out["output_control"] = torch.from_numpy(output).float()
        out["label_high_level"] = high_level_label

        # return a dictionary
        return out


class StateToControlDataset(Dataset):

    def __init__(self, config, split):
        super().__init__()
        self.data_root = config["data_root"]
        self.input_names = config["drone_state_names"]
        self.output_name = config["ground_truth_name"]
        self.output_columns = []
        self.split = split

        frame_index = pd.read_csv(os.path.join(self.data_root, "index", "frame_index.csv"))
        split_index = pd.read_csv(config["split_config"] + ".csv")
        frame_index["split"] = split_index

        ground_truth = pd.read_csv(os.path.join(self.data_root, "index", f"{self.output_name}.csv"))
        for col in ground_truth:
            frame_index[col] = ground_truth[col]
            self.output_columns.append(col)

        drone_state = pd.read_csv(os.path.join(self.data_root, "index", "state.csv"))
        for col in self.input_names:
            frame_index[col] = drone_state[col]

        self.index = frame_index[frame_index["split"] == self.split].copy()

        self.index["label"] = 4
        for idx, (track_name, half) in enumerate([("flat", "left_half"), ("flat", "right_half"),
                                                  ("wave", "left_half"), ("wave", "right_half")]):
            self.index.loc[(self.index["track_name"] == track_name) & (self.index[half] == 1), "label"] = idx

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

        # read the state input
        state_input = self.index[self.input_names].iloc[item].values

        # determine the "high level command" label for the sample
        high_level_label = self.index["label"].iloc[item]

        # extract the control GT from the dataframe
        output = self.index[self.output_columns].iloc[item].values

        # compile everything
        out = {
            "input_state": torch.from_numpy(state_input).float(),
            "output_control": torch.from_numpy(output).float(),
            "label_high_level": high_level_label
        }

        # return a dictionary
        return out


if __name__ == "__main__":
    test_config = {
        "data_root": os.getenv("GAZESIM_ROOT"),
        "input_video_names": ["screen"],
        # "ground_truth_name": "moving_window_frame_mean_gt",
        "ground_truth_name": "drone_control_frame_mean_gt",
        "drone_state_names": ["DroneVelocityX", "DroneVelocityY", "DroneVelocityZ"],
        "resize": (122, 122),
        "split_config": resolve_split_index_path(11, data_root=os.getenv("GAZESIM_ROOT")),
        "stack_size": 4,
        "dreyeve_transforms": False,
        "no_normalisation": False
    }

    dataset = StackedImageAndStateToControlDataset(test_config, split="train")
    print(len(dataset))
    # print(dataset.index.columns)

    sample = dataset[0]
    print("sample:", sample.keys())

    exit(0)

    print("\ninput_image_0:", sample["input_image_0"].keys())
    for k, v in sample["input_image_0"].items():
        print(f"{k}: {v.shape}")

    # print("\ninput_image_1:", sample["input_image_1"].keys())
    # for k, v in sample["input_image_1"].items():
    #     print(f"{k}: {v.shape}")

    print("\noutput_attention:", sample["output_attention"].keys())
    for k, v in sample["output_attention"].items():
        print(f"{k}: {v.shape}")
