import os
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
            label_transform=None
    ):
        super(TurnOnlyFrameDataset, self).__init__()

        self.data_root = data_root
        self.gt_name = gt_name
        self.data_transform = data_transform
        self.label_transform = label_transform

        self.df_index = pd.read_csv(os.path.join(self.data_root, f"{prefix}_{split}.csv"))
        self.cap_dict = {}

    def __len__(self):
        return len(self.df_index.index)

    def __getitem__(self, item):
        # TODO: need to update CSV files and change this to be relative path from the data root
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

        if self.data_transform:
            frame = self.data_transform(frame)
        if self.label_transform:
            label = self.label_transform(label[:, :, 0])

        return frame, label


def get_dataset(data_root, split="train", name="turn_left"):
    dataset = None

    if name in ["turn_left", "turn_right"]:
        mean = [0.21415611, 0.21116218, 0.23293883] if name == "turn_left" else [0.21929578, 0.21621058, 0.23851419]
        std = [0.02141269, 0.01657429, 0.02027875] if name == "turn_left" else [0.02108472, 0.01647169, 0.02017207]
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(150),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        label_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(150),
            transforms.ToTensor(),
            MakeValidDistribution()
        ])
        dataset = TurnOnlyFrameDataset(
            data_root=data_root,
            split=split,
            data_transform=data_transform,
            label_transform=label_transform
        )

    return dataset


if __name__ == "__main__":
    # test that everything with the dataset works as intended
    ds = get_dataset(os.environ["GAZESIM_ROOT"])
    print("Dataset length:", len(ds))
    test_data, test_label = ds[0]
    print("test_data:", type(test_data), test_data.shape, test_data.min(), test_data.max())
    print("test_label:", type(test_label), test_label.shape, test_label.max(), test_label.sum())
