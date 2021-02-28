# TODO: load attention model, load respective dataset (by default validation split), evaluate
#  "normal loss" (KL-divergence), MSE (?), correlation coefficient on this data
#  => simultaneously do the same thing for at least the random gaze thingy
#     => simply create ToAttentionDataset?
#  => also load mean mask and compare the same thing
# I think this should be separate from gaze predictions or at least there should be 2 different functions?

import os
import json
import numpy as np
import pandas as pd
import matplotlib.style as style
import cv2
import torch

from tqdm import tqdm
from gazesim.data.utils import find_contiguous_sequences, resolve_split_index_path, run_info_to_path
from gazesim.data.datasets import ImageDataset, ToAttentionDataset
from gazesim.training.config import parse_config as parse_train_config
from gazesim.training.helpers import resolve_model_class, resolve_dataset_class
from gazesim.training.utils import to_device, to_batch, load_model
from gazesim.training.helpers import resolve_output_processing_func
from gazesim.models.utils import image_softmax, image_log_softmax


def load(config):
    model, model_config = load_model(config["model_load_path"], config["gpu"], return_config=True)

    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(config["gpu"])
                          if use_cuda and config["gpu"] < torch.cuda.device_count() else "cpu")

    # move model to correct device
    model.to(device)
    model.eval()

    # modify model config for dataset loading
    model_config["data_root"] = config["data_root"]
    model_config["split_config"] = config["split_config"]

    return model, model_config, device


def prepare_data(config, model_config):
    # need dataset to load original videos (for input) and ground-truth
    dataset_gt = resolve_dataset_class(model_config["dataset_name"])(model_config, split=config["split"], training=False)

    # also need to load random gaze attention GT
    random_gaze_config = model_config.copy()
    random_gaze_config["attention_ground_truth"] = "shuffled_random_moving_window_frame_mean_gt"
    dataset_random_gaze = ToAttentionDataset(random_gaze_config, split=config["split"], training=False)

    # also load mean mask corresponding to split => kinda annoying tho
    mean_mask = cv2.imread(config["mean_mask_path"])
    mean_mask = cv2.cvtColor(mean_mask, cv2.COLOR_BGR2RGB)
    mean_mask = dataset_gt.attention_output_transform(mean_mask)
    mean_mask = {"output_attention": mean_mask}

    return dataset_gt, dataset_random_gaze, mean_mask


def evaluate_attention(config):
    # load model and config
    model, model_config, device = load(config)

    # prepare the data(sets)
    dataset_gt, dataset_random_gaze, mean_mask = prepare_data(config, model_config)

    # make sure they have the same number of frames
    assert len(dataset_gt) == len(dataset_random_gaze), "GT and random gaze dataset don't have the same length."

    # define loss functions
    loss_func_mse = torch.nn.MSELoss()
    loss_func_kl = torch.nn.KLDivLoss(reduction="batchmean")
    loss_func_cc = None  # TODO: implement this

    total_loss_kl_pred = 0
    total_loss_kl_random_gaze = 0
    total_loss_kl_mean_mask = 0
    total_loss_mse_pred = 0
    total_loss_mse_random_gaze = 0
    total_loss_mse_mean_mask = 0

    kl_not_inf_pred = 0
    kl_not_inf_random_gaze = 0
    kl_not_inf_mean_mask = 0

    # loop through the dataset (without batching I guess? might be better to do it?)
    for start_frame in tqdm(range(0, len(dataset_gt), config["batch_size"])):
        # construct a batch (maybe just repeat mean map if we even use it)
        batch_gt = []
        batch_random_gaze = []
        batch_mean_mask = []
        for in_batch_idx in range(start_frame, start_frame + config["batch_size"]):
            if in_batch_idx >= len(dataset_gt):
                break
            batch_gt.append(dataset_gt[in_batch_idx])
            batch_random_gaze.append(dataset_random_gaze[in_batch_idx])
            batch_mean_mask.append(mean_mask)
        batch_gt = to_device(to_batch(batch_gt), device)
        batch_random_gaze = to_device(to_batch(batch_random_gaze), device)
        batch_mean_mask = to_device(to_batch(batch_mean_mask), device)

        # get the attention ground-truth from the batch
        attention_gt = batch_gt["output_attention"]
        attention_random_gaze = batch_random_gaze["output_attention"]
        # TODO: need to transform this more I guess...
        # print(type(attention_gt), attention_gt.dtype)
        # print(type(attention_random_gaze), attention_random_gaze.dtype)
        attention_random_gaze_no_activation = torch.log(attention_random_gaze)
        attention_mean_mask = batch_mean_mask["output_attention"]
        attention_mean_mask_no_activation = torch.log(attention_mean_mask)

        # print(attention_gt.view(tuple(attention_gt.shape[:-2]) + (-1,)).sum(-1))
        # print(attention_random_gaze.view(tuple(attention_random_gaze.shape[:-2]) + (-1,)).sum(-1))
        #
        # print(attention_gt.max(), attention_gt.mean())

        # compute the attention prediction
        output = model(batch_gt)
        attention_pred_no_activation = image_log_softmax(output["output_attention"])
        attention_pred = image_softmax(attention_pred_no_activation)
        # print(attention_pred.shape)
        # print(attention_pred.view(tuple(attention_pred.shape[:-2]) + (-1,)).sum(-1))

        # compute the losses between the "actual" ground-truth and the other stuff
        loss_kl_pred = loss_func_kl(attention_pred_no_activation, attention_gt).item()
        loss_kl_random_gaze = loss_func_kl(attention_random_gaze_no_activation, attention_gt).item()
        loss_kl_mean_mask = loss_func_kl(attention_mean_mask_no_activation, attention_gt).item()

        loss_mse_pred = loss_func_mse(attention_pred, attention_gt).item()
        loss_mse_random_gaze = loss_func_mse(attention_random_gaze, attention_gt).item()
        loss_mse_mean_mask = loss_func_mse(attention_mean_mask, attention_gt).item()

        # print(loss_kl_pred, loss_kl_random_gaze)
        # print(loss_mse_pred, loss_mse_random_gaze)
        # exit()

        # TODO: mean over batch?
        # sum losses
        if not np.isinf(loss_kl_pred):
            total_loss_kl_pred += loss_kl_pred
            kl_not_inf_pred += 1
        if not np.isinf(loss_kl_random_gaze):
            total_loss_kl_random_gaze += loss_kl_random_gaze
            kl_not_inf_random_gaze += 1
        if not np.isinf(loss_kl_mean_mask):
            total_loss_kl_mean_mask += loss_kl_mean_mask
            kl_not_inf_mean_mask += 1

        total_loss_mse_pred += loss_mse_pred
        total_loss_mse_random_gaze += loss_mse_random_gaze
        total_loss_mse_mean_mask += loss_mse_mean_mask

    if kl_not_inf_pred > 0:
        print("Average KL-divergence loss for attention model: {}".format(total_loss_kl_pred / kl_not_inf_pred))
    else:
        print("Average KL-divergence loss for attention model is infinite.")

    if kl_not_inf_random_gaze > 0:
        print("Average KL-divergence loss for shuffled GT: {}".format(total_loss_kl_random_gaze / kl_not_inf_random_gaze))
    else:
        print("Average KL-divergence loss for shuffled GT is infinite.")

    if kl_not_inf_mean_mask > 0:
        print("Average KL-divergence loss for mean mask: {}".format(total_loss_kl_mean_mask / kl_not_inf_mean_mask))
    else:
        print("Average KL-divergence loss for mean mask is infinite.")

    print("\nAverage MSE loss for attention model: {}".format(total_loss_mse_pred / len(dataset_gt)))
    print("Average MSE loss for shuffled GT: {}".format(total_loss_mse_random_gaze / len(dataset_gt)))
    print("Average MSE loss for mean mask: {}".format(total_loss_mse_mean_mask / len(dataset_gt)))


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
    parser.add_argument("-mm", "--mean_mask_path", type=str,
                        default=os.path.join(os.getenv("GAZESIM_ROOT"), "preprocessing_info", "split011_mean_mask.png"),
                        help="The path to the mean mask to compare stuff against.")
    parser.add_argument("-vn", "--video_name", type=str, default="screen",
                        help="The name of the input video.")
    parser.add_argument("-s", "--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Splits for which to create videos.")
    parser.add_argument("-sc", "--split_config", type=str, default=11,
                        help="TODO.")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="The GPU to use.")
    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="Batch size.")

    # parse the arguments
    arguments = parser.parse_args()

    # main
    evaluate_attention(parse_config(arguments))
