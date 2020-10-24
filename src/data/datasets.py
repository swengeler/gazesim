import os
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pims import PyAVReaderIndexed, PyAVReaderTimed

# TODO: this will definitely have to be changed to something more elegant
# TODO: would probably also be good to put this in some config file together with train, val, test indices etc.
CHANNEL_MEAN_LEFT_TURN = [0.23293883, 0.21116218, 0.21415611]
CHANNEL_MEAN_RIGHT_TURN = [0.23851419, 0.21621058, 0.21929578]
CHANNEL_STD_LEFT_TURN = [0.02027875, 0.01657429, 0.02141269]
CHANNEL_STD_RIGHT_TURN = [0.02017207, 0.01647169, 0.02108472]


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
                "data": PyAVReaderTimed(os.path.join(full_run_path, "screen.mp4"), cache_size=1),
                "label": PyAVReaderTimed(os.path.join(full_run_path, f"{self.gt_name}.mp4"), cache_size=1)
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
            self.cap_dict[full_run_path] = cv2.VideoCapture(os.path.join(full_run_path, "screen.mp4"))

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
            self.cap_dict[full_run_path] = PyAVReaderTimed(os.path.join(full_run_path, "screen.mp4"), cache_size=1)

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
            # PyAVReaderTimed(os.path.join(full_run_path, "screen.mp4"), cache_size=1)
            self.cap_dict["data"] = PyAVReaderTimed(os.path.join(self.run_dir, "screen.mp4"), cache_size=1)
            self.cap_dict["label"] = PyAVReaderTimed(os.path.join(self.run_dir, f"{self.gt_name}.mp4"), cache_size=1)

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


def get_dataset(data_root, split="train", data_type="turn_left", resize_height=150, use_pims=False, sub_index=None):
    dataset = None

    mean = CHANNEL_MEAN_LEFT_TURN if "turn_left" in data_type else CHANNEL_MEAN_RIGHT_TURN
    std = CHANNEL_STD_LEFT_TURN if "turn_left" in data_type else CHANNEL_STD_RIGHT_TURN

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_height),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

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
        dataset_class = DroneControlDatasetPIMS if use_pims else DroneControlDataset
        dataset = dataset_class(
            data_root=data_root,
            split=split,
            prefix=data_type,
            gt_name="drone_control_gt",
            data_transform=data_transform,
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

    # TODO: add dataset using all available "valid" frames

    return dataset


if __name__ == "__main__":
    # test that everything with the dataset works as intended
    ds = get_dataset(os.getenv("GAZESIM_ROOT"), data_type="turn_left_drone_control_gt")
    print("Dataset length:", len(ds))
    test_data, test_label = ds[0]
    print("test_data:", type(test_data), test_data.shape, test_data.min(), test_data.max())
    print("test_label:", type(test_label), test_label.shape, test_label.max(), test_label.sum())
