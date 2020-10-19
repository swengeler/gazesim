import os
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pims import PyAVReaderIndexed, PyAVReaderTimed


class MakeValidDistribution(object):

    def __call__(self, sample):
        # expects torch tensor with
        pixel_sum = torch.sum(sample, [])
        if pixel_sum > 0.0:
            sample = sample / pixel_sum
        return sample


class TurnOnlyFrameDataset(Dataset):

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

        self.cap_dict = {}
        self.return_original = False

    def __len__(self):
        return len(self.df_index.index)

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

        success, frame = self.cap_dict[full_run_path]["data"].read()
        success, label = self.cap_dict[full_run_path]["label"].read()

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

    def __init__(
            self,
            data_root,
            split="train",
            prefix="turn_left",
            gt_name="moving_window_gt",
            data_transform=None,
            label_transform=None
    ):
        super().__init__(data_root, split, prefix, gt_name, data_transform, label_transform)

    def __getitem__(self, item):
        full_run_path = os.path.join(self.data_root, self.df_index["rel_run_path"].iloc[item])
        if full_run_path not in self.cap_dict:
            self.cap_dict[full_run_path] = {
                "data": PyAVReaderTimed(os.path.join(full_run_path, "screen.mp4"), cache_size=1),
                "label": PyAVReaderTimed(os.path.join(full_run_path, f"{self.gt_name}.mp4"), cache_size=1)
            }

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


def get_dataset(data_root, split="train", name="turn_left", resize_height=150, use_pims=False, sub_index=None):
    dataset = None

    if name in ["turn_left", "turn_right", "turn_both"]:
        mean = [0.21415611, 0.21116218, 0.23293883] if name == "turn_left" else [0.21929578, 0.21621058, 0.23851419]
        std = [0.02141269, 0.01657429, 0.02027875] if name == "turn_left" else [0.02108472, 0.01647169, 0.02017207]
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_height),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
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
            data_transform=data_transform,
            label_transform=label_transform,
            sub_index=sub_index
        )

    # TODO: add dataset loading drone state/drone commands ==> average for each frame

    # TODO: add dataset using all available "valid" frames

    return dataset


if __name__ == "__main__":
    # test that everything with the dataset works as intended
    ds = get_dataset(os.environ["GAZESIM_ROOT"])
    print("Dataset length:", len(ds))
    test_data, test_label = ds[0]
    print("test_data:", type(test_data), test_data.shape, test_data.min(), test_data.max())
    print("test_label:", type(test_label), test_label.shape, test_label.max(), test_label.sum())
