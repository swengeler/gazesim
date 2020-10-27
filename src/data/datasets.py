import os
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pims import PyAVReaderIndexed, PyAVReaderTimed
from src.data.utils import get_indexed_reader


STATISTICS = {
    "screen": {
        "turn_left": {"mean": [0.23293883, 0.21116218, 0.21415611], "std": [0.02027875, 0.01657429, 0.02141269]},
        "turn_right": {"mean": [0.23851419, 0.21621058, 0.21929578], "std": [0.02017207, 0.01647169, 0.02108472]}
    },
    "hard_mask_moving_window_gt": {
        "turn_left": {"mean": [0.00197208, 0.00194627, 0.00206955], "std": [0.00038268, 0.00034424, 0.000489]},
        "turn_right": {"mean": [0.00199134, 0.00198215, 0.00214822], "std": [0.00038682, 0.00035, 0.0005171]}
    },
    "mean_mask_moving_window_gt": {
        "turn_left": {"mean": [0.01235463, 0.01220624, 0.01134171], "std": [0.00174027, 0.00150813, 0.0019455]},
        "turn_right": {"mean": [0.01265004, 0.01247167, 0.01154284], "std": [0.0017945, 0.00154555, 0.00193905]}
    },
    "soft_mask_moving_window_gt": {
        "turn_left": {"mean": [0.04040431, 0.04105297, 0.03419323], "std": [0.00104593, 0.00093455, 0.00121713]},
        "turn_right": {"mean": [0.0415835, 0.04213016, 0.03518509], "std": [0.00104145, 0.00093271, 0.00123364]}
    }
}


class MakeValidDistribution(object):

    def __call__(self, sample):
        # expects torch tensor with
        pixel_sum = torch.sum(sample, [])
        if pixel_sum > 0.0:
            sample = sample / pixel_sum
        return sample


class TurnOnlyDataset(Dataset):

    def __init__(
            self,
            data_root,
            split="train",
            prefix="turn_left",
            gt_name="moving_window_gt",
            data_transform=None,
            label_transform=None,
            sub_index=None
    ):
        super().__init__()

        self.data_root = data_root
        self.gt_name = gt_name
        self.data_transform = data_transform
        self.label_transform = label_transform

        if prefix == "turn_both":
            df_left = pd.read_csv(os.path.join(self.data_root, f"turn_left_{split}.csv"))
            df_right = pd.read_csv(os.path.join(self.data_root, f"turn_right_{split}.csv"))
            self.df_index = pd.concat([df_left, df_right], ignore_index=True)
        else:
            self.df_index = pd.read_csv(os.path.join(self.data_root, f"{prefix}_{split}.csv"))

        if sub_index:
            # assume that sub_index is an iterable with two entries (start and end index)
            self.df_index = self.df_index.iloc[sub_index[0]:sub_index[1]]
            self.df_index = self.df_index.reset_index()

    def __len__(self):
        return len(self.df_index.index)

    def __getitem__(self, item):
        raise NotImplementedError


class TurnOnlyFrameDataset(TurnOnlyDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cap_dict = {}
        self.return_original = False

    def __getitem__(self, item):
        full_run_path = os.path.join(self.data_root, self.df_index["rel_run_path"].iloc[item])
        if full_run_path not in self.cap_dict:
            self.cap_dict[full_run_path] = {
                "data": cv2.VideoCapture(os.path.join(full_run_path, "screen.mp4")),
                "label": cv2.VideoCapture(os.path.join(full_run_path, f"{self.gt_name}.mp4"))
            }

        frame_index = self.df_index["frame"].iloc[item]
        self.cap_dict[full_run_path]["data"].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.cap_dict[full_run_path]["label"].set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        frame = cv2.cvtColor(self.cap_dict[full_run_path]["data"].read()[1], cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(self.cap_dict[full_run_path]["label"].read()[1], cv2.COLOR_BGR2RGB)

        frame_original = frame.copy()
        label_original = label.copy()
        if self.data_transform:
            frame = self.data_transform(frame)
        if self.label_transform:
            label = self.label_transform(label[:, :, 0])

        if self.return_original:
            return frame, label, frame_original, label_original
        return frame, label


class TurnOnlyFrameDatasetPIMS(TurnOnlyFrameDataset):

    def __getitem__(self, item):
        full_run_path = os.path.join(self.data_root, self.df_index["rel_run_path"].iloc[item])
        if full_run_path not in self.cap_dict:
            self.cap_dict[full_run_path] = {
                # "data": PyAVReaderIndexed(os.path.join(full_run_path, "screen.mp4")),# cache_size=1),
                # "label": PyAVReaderIndexed(os.path.join(full_run_path, f"{self.gt_name}.mp4"))#, cache_size=1)
                "data": get_indexed_reader(os.path.join(full_run_path, "screen.mp4")),
                "label": get_indexed_reader(os.path.join(full_run_path, f"{self.gt_name}.mp4"))
            }
            # TODO: could probably simplify a lot by just having a "setup" for the cap_dict
            #  and a method for getting a frame (and each class implements it differently)

        frame_index = self.df_index["frame"].iloc[item]
        frame = self.cap_dict[full_run_path]["data"][frame_index]
        label = self.cap_dict[full_run_path]["label"][frame_index]

        frame_original = frame.copy()
        label_original = label.copy()
        if self.data_transform:
            frame = self.data_transform(frame)
        if self.label_transform:
            label = self.label_transform(label[:, :, 0])

        if self.return_original:
            return frame, label, frame_original, label_original
        return frame, label


class DroneControlDataset(Dataset):

    def __init__(
            self,
            data_root,
            split="train",
            prefix="turn_left",
            video_name="screen",
            gt_name="moving_window_gt",
            data_transform=None,
            label_transform=None,
            sub_index=None
    ):
        super().__init__()

        # TODO: this is not the prettiest structure, maybe dividing datasets into "frame" datasets and
        #  "regression" (maybe also "classification") datasets is the way to go, because for one an index
        #  should be loaded, whereas for the other the data should already be in the dataframe
        #  ==> ACTUALLY, since we need to load frames for either of the datasets we need an index anyway,
        #  so my initial structure (everything subclassing one Dataset with the index) did make some sense;
        #  might want to change it back to that

        self.data_root = data_root
        self.video_name = video_name
        self.gt_name = gt_name
        self.data_transform = data_transform
        self.label_transform = label_transform

        if prefix == "turn_both":
            # index now contains both the information for finding the input frames and the actual "labels"
            # TODO: might make more sense to separate this, at least in the structure of the dataset
            df_left = pd.read_csv(os.path.join(self.data_root, f"turn_left_drone_control_gt_{split}.csv"))
            df_right = pd.read_csv(os.path.join(self.data_root, f"turn_right_drone_control_gt_{split}.csv"))
            self.df_index = pd.concat([df_left, df_right], ignore_index=True)
        else:
            self.df_index = pd.read_csv(os.path.join(self.data_root, f"{prefix}_{split}.csv"))

        if sub_index:
            # assume that sub_index is an iterable with two entries (start and end index)
            self.df_index = self.df_index.iloc[sub_index[0]:sub_index[1]]
            self.df_index = self.df_index.reset_index()

        self.cap_dict = {}
        self.return_original = False

    def __len__(self):
        return len(self.df_index.index)

    def __getitem__(self, item):
        full_run_path = os.path.join(self.data_root, self.df_index["rel_run_path"].iloc[item])
        if full_run_path not in self.cap_dict:
            self.cap_dict[full_run_path] = cv2.VideoCapture(os.path.join(full_run_path, f"{self.video_name}.mp4"))

        # loading the frame
        frame_index = self.df_index["frame"].iloc[item]
        self.cap_dict[full_run_path].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        frame = cv2.cvtColor(self.cap_dict[full_run_path].read()[1], cv2.COLOR_BGR2RGB)

        frame_original = frame.copy()
        if self.data_transform:
            frame = self.data_transform(frame)

        # selecting the label
        label = self.df_index[["Throttle", "Roll", "Pitch", "Yaw"]].iloc[item].values
        label = torch.from_numpy(label).float()

        if self.return_original:
            return frame, label, frame_original, label
        return frame, label


class DroneControlDatasetPIMS(DroneControlDataset):

    def __getitem__(self, item):
        full_run_path = os.path.join(self.data_root, self.df_index["rel_run_path"].iloc[item])
        if full_run_path not in self.cap_dict:
            # self.cap_dict[full_run_path] = PyAVReaderIndexed(os.path.join(full_run_path, "screen.mp4"))#, cache_size=1)
            self.cap_dict[full_run_path] = get_indexed_reader(os.path.join(full_run_path, f"{self.video_name}.mp4"))

        # loading the frame
        frame_index = self.df_index["frame"].iloc[item]
        frame = self.cap_dict[full_run_path][frame_index]

        frame_original = frame.copy()
        if self.data_transform:
            frame = self.data_transform(frame)

        # selecting the label
        label = self.df_index[["Throttle", "Roll", "Pitch", "Yaw"]].iloc[item].values
        label = torch.from_numpy(label).float()

        if self.return_original:
            return frame, label, frame_original, label
        return frame, label


class DualBranchDroneControlDatasetPIMS(Dataset):

    def __init__(
            self,
            data_root,
            split="train",
            prefix="turn_left",
            video_names=("screen", "hard_mask_moving_window_gt"),
            gt_name="moving_window_gt",
            data_transforms=None,
            label_transform=None,
            sub_index=None
    ):
        super().__init__()

        assert len(video_names) == 2, "Need exactly two input video names for dual-branch network/dataset."
        assert data_transforms is None or len(data_transforms) == len(video_names), \
            "Number of transforms must match number of input videos."

        self.data_root = data_root
        self.video_names = video_names
        self.gt_name = gt_name
        self.data_transforms = data_transforms
        self.label_transform = label_transform

        if prefix == "turn_both":
            df_left = pd.read_csv(os.path.join(self.data_root, f"turn_left_drone_control_gt_{split}.csv"))
            df_right = pd.read_csv(os.path.join(self.data_root, f"turn_right_drone_control_gt_{split}.csv"))
            self.df_index = pd.concat([df_left, df_right], ignore_index=True)
        else:
            self.df_index = pd.read_csv(os.path.join(self.data_root, f"{prefix}_{split}.csv"))

        if sub_index:
            # assume that sub_index is an iterable with two entries (start and end index)
            self.df_index = self.df_index.iloc[sub_index[0]:sub_index[1]]
            self.df_index = self.df_index.reset_index()

        self.cap_dict = {}
        self.return_original = False

    def __len__(self):
        return len(self.df_index.index)

    def __getitem__(self, item):
        full_run_path = os.path.join(self.data_root, self.df_index["rel_run_path"].iloc[item])
        if full_run_path not in self.cap_dict:
            self.cap_dict[full_run_path] = {}
            for vn in self.video_names:
                self.cap_dict[full_run_path][vn] = get_indexed_reader(os.path.join(full_run_path, f"{vn}.mp4"))

        # loading the frames
        frame_index = self.df_index["frame"].iloc[item]
        frames = []
        for vn in self.video_names:
            frame = self.cap_dict[full_run_path][vn][frame_index]
            frames.append(frame)

        frames_original = [f.copy() for f in frames]
        if self.data_transforms:
            frames = [self.data_transforms[f_idx](f) for f_idx, f in enumerate(frames)]

        # selecting the label
        label = self.df_index[["Throttle", "Roll", "Pitch", "Yaw"]].iloc[item].values
        label = torch.from_numpy(label).float()

        if self.return_original:
            return frames, label, frames_original, label
        return frames, label


class SingleVideoDataset(Dataset):

    def __init__(
            self,
            run_dir,
            gt_name="moving_window_gt",
            data_transform=None,
            label_transform=None,
            sub_index=None
    ):
        # need path to the run directory for the video
        # can then get access to the input video and GT video
        # not sure if anything else (e.g. screen_frame_info) will be needed/useful (maybe for sub-indexing?)
        self.run_dir = run_dir
        self.gt_name = gt_name
        self.data_transform = data_transform
        self.label_transform = label_transform

        # dataframe with all frames
        # TODO: think about whether this should automatically be sub-indexing to only include frames with GT?
        self.df_frame_info = pd.read_csv(os.path.join(self.run_dir, "screen_frame_info.csv"))
        self.cap_dict = {}
        self.return_original = False

        if sub_index:
            self.df_frame_info = self.df_frame_info.iloc[sub_index[0]:sub_index[1]]
            self.df_frame_info = self.df_frame_info.reset_index()

    def __len__(self):
        return len(self.df_frame_info.index)

    def __getitem__(self, item):
        if "data" not in self.cap_dict:
            self.cap_dict["data"] = cv2.VideoCapture(os.path.join(self.run_dir, "screen.mp4"))
            self.cap_dict["label"] = cv2.VideoCapture(os.path.join(self.run_dir, f"{self.gt_name}.mp4"))

        frame_index = self.df_frame_info["frame"].iloc[item]
        self.cap_dict["data"].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.cap_dict["label"].set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        frame = cv2.cvtColor(self.cap_dict["data"].read()[1], cv2.COLOR_BGR2RGB)

        label = None
        if self.df_frame_info["gt_available"].iloc[item] == 1:
            label = cv2.cvtColor(self.cap_dict["label"].read()[1], cv2.COLOR_BGR2RGB)

        frame_original = frame.copy()
        if self.data_transform is not None:
            frame = self.data_transform(frame)
        if label is not None and self.label_transform is not None:
            label = self.label_transform(label[:, :, 0])

        if self.return_original:
            return frame, label, frame_original
        return frame, label


class SingleVideoDatasetPIMS(SingleVideoDataset):

    def __getitem__(self, item):
        if "data" not in self.cap_dict:
            # PyAVReaderIndexed(os.path.join(full_run_path, "screen.mp4"), cache_size=1)
            # self.cap_dict["data"] = PyAVReaderIndexed(os.path.join(self.run_dir, "screen.mp4"))#, cache_size=1)
            # self.cap_dict["label"] = PyAVReaderIndexed(os.path.join(self.run_dir, f"{self.gt_name}.mp4"))#, cache_size=1)
            self.cap_dict["data"] = get_indexed_reader(os.path.join(self.run_dir, "screen.mp4"))
            self.cap_dict["label"] = get_indexed_reader(os.path.join(self.run_dir, f"{self.gt_name}.mp4"))

        frame_index = self.df_frame_info["frame"].iloc[item]
        frame = self.cap_dict["data"][frame_index]

        label = None
        if self.df_frame_info["gt_available"].iloc[item] == 1:
            label = self.cap_dict["label"][frame_index]

        frame_original = frame.copy()
        if self.data_transform is not None:
            frame = self.data_transform(frame)
        if label is not None and self.label_transform is not None:
            label = self.label_transform(label[:, :, 0])

        if self.return_original:
            return frame, label, frame_original
        return frame, label


class DroneControlStackedDatasetPIMS(DroneControlDataset):

    def __init__(self, *args, stack_size=16, **kwargs):
        super().__init__(*args, **kwargs)

        self.stack_size = stack_size

        # find contiguous sequences of frames
        sequences = []
        frames = self.df_index["frame"]
        jumps = (frames - frames.shift()) != 1
        frames = list(frames.index[jumps]) + [frames.index[-1] + 1]
        for i, start_index in enumerate(frames[:-1]):
            # note that the range [start_frame, end_frame) is exclusive
            sequences.append((start_index, frames[i + 1]))

        # need some way to not mess up the indexing, but also cannot remove data
        # maybe use second "index" that skips the (stack_size - 1) first frames of each contiguous sequence?
        self.df_index["stack_index"] = -1
        df_col_index = self.df_index.columns.get_loc("stack_index")
        total_num_frame_stacks = 0
        for start_index, end_index in sequences:
            num_frames_seq = end_index - start_index
            num_frame_stacks = num_frames_seq - self.stack_size + 1
            if end_index - start_index < self.stack_size:
                continue

            # assign index over entire dataframe to "valid" frames
            index = np.arange(num_frame_stacks) + total_num_frame_stacks
            self.df_index.iloc[(start_index + stack_size - 1):end_index, df_col_index] = index

            # keep track of the current number of stacks (highest index)
            total_num_frame_stacks += num_frame_stacks

    def __len__(self):
        return (self.df_index["stack_index"] != -1).sum()

    def __getitem__(self, item):
        stack_index = self.df_index.index[self.df_index["stack_index"] == item].values[0]
        df_stack = self.df_index.iloc[(stack_index - self.stack_size + 1):(stack_index + 1)]

        # this should probably always work since contiguous clips are from the same video
        full_run_path = os.path.join(self.data_root, self.df_index["rel_run_path"].iloc[stack_index])
        if full_run_path not in self.cap_dict:
            self.cap_dict[full_run_path] = get_indexed_reader(os.path.join(full_run_path, f"{self.video_name}.mp4"))

        stack = []
        stack_original = []
        for _, row in df_stack.iterrows():
            frame = self.cap_dict[full_run_path][row["frame"]]

            frame_original = frame.copy()
            if self.data_transform:
                frame = self.data_transform(frame)

            stack.append(frame)
            stack_original.append(frame_original)

        # stack the frames
        stack = torch.stack(stack, 1)
        stack_original = np.stack(stack_original, 0)

        # the label is still only from a single frame
        label = self.df_index[["Throttle", "Roll", "Pitch", "Yaw"]].iloc[stack_index].values
        label = torch.from_numpy(label).float()

        if self.return_original:
            return stack, label, stack_original, label
        return stack, label


def get_dataset(data_root, split="train", data_type="turn_left", video_name="screen", resize_height=150, use_pims=False, sub_index=None):
    dataset = None

    # TODO: this is super messy, really needs to be changed when datasets are refactored
    turn_type = None
    if data_type in ["turn_left", "turn_right"]:
        turn_type = data_type
    elif data_type == "turn_left_drone_control_gt":
        turn_type = "turn_left"
    elif data_type == "turn_right_drone_control_gt":
        turn_type = "turn_right"
    elif data_type == "single_video":
        turn_type = "turn_left"

    if isinstance(video_name, str):
        mean = STATISTICS[video_name][turn_type]["mean"]
        std = STATISTICS[video_name][turn_type]["std"]

        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_height),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        data_transform = [transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_height),
            transforms.ToTensor(),
            transforms.Normalize(STATISTICS[vn][turn_type]["mean"], STATISTICS[vn][turn_type]["std"])
        ]) for vn in video_name]

    if data_type in ["turn_left", "turn_right", "turn_both"]:
        label_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_height),
            transforms.ToTensor(),
            MakeValidDistribution()
        ])
        dataset_class = TurnOnlyFrameDatasetPIMS if use_pims else TurnOnlyFrameDataset
        dataset = dataset_class(
            data_root=data_root,
            split=split,
            prefix=data_type,
            data_transform=data_transform,
            label_transform=label_transform,
            sub_index=sub_index
        )
    elif data_type in ["turn_left_drone_control_gt", "turn_right_drone_control_gt", "turn_both_drone_control_gt"]:
        if isinstance(video_name, str):
            # dataset_class = DroneControlDatasetPIMS if use_pims else DroneControlDataset
            dataset = DroneControlStackedDatasetPIMS(
                data_root=data_root,
                split=split,
                prefix=data_type,
                video_name=video_name,
                gt_name="drone_control_gt",
                data_transform=data_transform,
                sub_index=sub_index
            )
        else:
            dataset = DualBranchDroneControlDatasetPIMS(
                data_root=data_root,
                split=split,
                prefix=data_type,
                video_names=video_name,
                gt_name="drone_control_gt",
                data_transforms=data_transform,
                sub_index=sub_index
            )
    elif data_type == "single_video":
        label_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_height),
            transforms.ToTensor(),
            MakeValidDistribution()
        ])
        dataset_class = SingleVideoDatasetPIMS if use_pims else SingleVideoDataset
        dataset = dataset_class(
            run_dir=data_root,
            gt_name="moving_window_gt",
            data_transform=data_transform,
            label_transform=label_transform,
            sub_index=sub_index
        )

    return dataset


if __name__ == "__main__":
    from time import time
    from tqdm import tqdm

    ds = DroneControlStackedDatasetPIMS(
        data_root=os.getenv("GAZESIM_ROOT"),
        stack_size=4,
        split="train",
        prefix="turn_left_drone_control_gt",
        gt_name="moving_window_gt",
        data_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(300),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])
    )

    print()
    print("len:", len(ds))
    test_stack, test_label = ds[5]
    print(test_stack.shape)

    exit()

    # test that everything with the dataset works as intended
    ds = get_dataset(os.getenv("GAZESIM_ROOT"), data_type="turn_left_drone_control_gt",
                     video_name=("screen", "hard_mask_moving_window_gt"), use_pims=True)
    print("Dataset length:", len(ds))
    test_data, test_label = ds[0]
    # print("test_data:", type(test_data), test_data.shape, test_data.min(), test_data.max())
    # print("test_label:", type(test_label), test_label.shape, test_label.max(), test_label.sum())

    exit(0)

    # for i in tqdm(range(len(ds))):
    #     hmmmm = ds[i]

    keys = list(ds.cap_dict.keys())

    lengths_list = []
    ts_list = []
    pims_list = []
    cv_list = []
    for k in tqdm(keys):
        tic_toc = ds.cap_dict[k].toc
        print(tic_toc)
        # print(tic_toc.keys())
        print(len(tic_toc["lengths"]))
        print(len(tic_toc["ts"]))

        lengths_list.append((np.array(tic_toc["lengths"]) != 1).sum())

        ts = np.array(tic_toc["ts"])
        test = ts[1:] - ts[:-1]
        ts_list.append((test != 256).sum())

        start = time()
        test_timed = PyAVReaderTimed(os.path.join(k, "screen.mp4"), cache_size=1)
        test_len = len(test_timed)
        end = time()
        pims_list.append(end - start)

        start = time()
        test_timed = cv2.VideoCapture(os.path.join(k, "screen.mp4"))
        test_len = int(test_timed.get(7))
        end = time()
        cv_list.append(end - start)

    print(np.sum(lengths_list))
    print(np.sum(ts_list))
    print(np.mean(pims_list))
    print(np.mean(cv_list))
