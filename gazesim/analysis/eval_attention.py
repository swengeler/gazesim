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
from gazesim.data.utils import resolve_split_index_path
from gazesim.data.datasets import ToAttentionDataset, ToGazeDataset
from gazesim.training.helpers import resolve_dataset_class
from gazesim.training.utils import to_device, to_batch, load_model
from gazesim.models.utils import image_softmax, image_log_softmax


def cc_numeric(y_pred, y_true):
    """
    Function to evaluate Pearson's correlation coefficient (sec 4.2.2 of [1]) on two samples.
    The two distributions are numpy arrays having arbitrary but coherent shapes.
    Taken from https://github.com/ndrplz/dreyeve/blob/5ba32174dff8fdbb5644b1cc8ecd2752308c06ce/experiments/metrics/metrics.py#L27
    :param y_true: groundtruth.
    :param y_pred: predictions.
    :return: numeric cc.
    """
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)

    eps = np.finfo(np.float32).eps

    cv2.normalize(y_pred, dst=y_pred, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(y_true, dst=y_true, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    y_pred = y_pred.ravel()
    y_true = y_true.ravel()

    y_pred = (y_pred - np.mean(y_pred)) / (eps + np.std(y_pred))
    y_true = (y_true - np.mean(y_true)) / (eps + np.std(y_true))

    cc = np.corrcoef(y_pred, y_true)

    return cc[0][1]


def pearson_corr_coeff(prediction, target):
    """
    prediction = prediction.reshape(prediction.shape[0], -1).cpu().detach().numpy()
    target = target.reshape(target.shape[0], -1).cpu().detach().numpy()
    coefficients = []
    for test in range(prediction.shape[0]):
        coefficients.append(pearsonr(prediction[test], target[test])[0])
    return np.mean(coefficients)
    """
    prediction = prediction.squeeze().cpu().detach().numpy()
    target = target.squeeze().cpu().detach().numpy()
    coefficients = []
    for test in range(prediction.shape[0]):
        coefficients.append(cc_numeric(prediction[test], target[test]))
    return np.mean(coefficients)


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
    random_gt_config = model_config.copy()
    if "gaze" in model_config["model_name"]:
        random_gt_config["gaze_ground_truth"] = "shuffled_random_frame_mean_gaze_gt"
        dataset_random_gt = ToGazeDataset(random_gt_config, split=config["split"], training=False)
    else:
        random_gt_config["attention_ground_truth"] = "shuffled_random_moving_window_frame_mean_gt"
        dataset_random_gt = ToAttentionDataset(random_gt_config, split=config["split"], training=False)

    # also load mean mask corresponding to the split
    if "gaze" in model_config["model_name"]:
        split_df = pd.read_csv(config["split_config"] + ".csv")
        gaze_df = pd.read_csv(os.path.join(config["data_root"], "index", "frame_mean_gaze_gt.csv"))
        gaze_df = gaze_df[~split_df["split"].isin(["none", config["split"]])]
        mean_baseline = gaze_df.mean().values
        if dataset_gt.output_scaling:
            mean_baseline *= np.array([800.0, 600.0])
        mean_baseline = {"output_gaze": mean_baseline}
    else:
        mean_baseline = cv2.imread(config["mean_mask_path"])
        mean_baseline = cv2.cvtColor(mean_baseline, cv2.COLOR_BGR2RGB)
        mean_baseline = dataset_gt.attention_output_transform(mean_baseline)
        mean_baseline = {"output_attention": mean_baseline}

    return dataset_gt, dataset_random_gt, mean_baseline


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
    total_loss_cc_pred = 0
    total_loss_cc_random_gaze = 0
    total_loss_cc_mean_mask = 0

    kl_not_inf_pred = 0
    kl_not_inf_random_gaze = 0
    kl_not_inf_mean_mask = 0

    # loop through the dataset (without batching I guess? might be better to do it?)
    num_batches = 0
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

        loss_cc_pred = pearson_corr_coeff(attention_pred, attention_gt)
        loss_cc_random_gaze = pearson_corr_coeff(attention_random_gaze, attention_gt)
        loss_cc_mean_mask = pearson_corr_coeff(attention_mean_mask, attention_gt)

        # print(loss_kl_pred, loss_kl_random_gaze)
        # print(loss_mse_pred, loss_mse_random_gaze)
        # exit()

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

        total_loss_cc_pred += loss_cc_pred
        total_loss_cc_random_gaze += loss_cc_random_gaze
        total_loss_cc_mean_mask += loss_cc_mean_mask

        num_batches += 1

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

    print("\nAverage MSE loss for attention model: {}".format(total_loss_mse_pred / num_batches))
    print("Average MSE loss for shuffled GT: {}".format(total_loss_mse_random_gaze / num_batches))
    print("Average MSE loss for mean mask: {}".format(total_loss_mse_mean_mask / num_batches))

    print("\nAverage CC loss for attention model: {}".format(total_loss_cc_pred / num_batches))
    print("Average CC loss for shuffled GT: {}".format(total_loss_cc_random_gaze / num_batches))
    print("Average CC loss for mean mask: {}".format(total_loss_cc_mean_mask / num_batches))


def evaluate_gaze(config):
    # load model and config
    model, model_config, device = load(config)

    # prepare the data
    dataset_gt, dataset_random_gaze, mean_gaze = prepare_data(config, model_config)

    # loss functions
    loss_func_mse = torch.nn.MSELoss()
    loss_func_l1 = torch.nn.L1Loss()

    total_loss_pred_mse = 0
    total_loss_pred_mse_partial_x = 0
    total_loss_pred_mse_partial_y = 0
    total_loss_pred_l1 = 0
    total_loss_pred_l1_partial_x = 0
    total_loss_pred_l1_partial_y = 0

    total_loss_random_gaze_mse = 0
    total_loss_random_gaze_mse_partial_x = 0
    total_loss_random_gaze_mse_partial_y = 0
    total_loss_random_gaze_l1 = 0
    total_loss_random_gaze_l1_partial_x = 0
    total_loss_random_gaze_l1_partial_y = 0

    total_loss_mean_gaze_mse = 0
    total_loss_mean_gaze_mse_partial_x = 0
    total_loss_mean_gaze_mse_partial_y = 0
    total_loss_mean_gaze_l1 = 0
    total_loss_mean_gaze_l1_partial_x = 0
    total_loss_mean_gaze_l1_partial_y = 0

    # loop through the dataset (without batching I guess? might be better to do it?)
    num_batches = 0
    for start_frame in tqdm(range(0, len(dataset_gt), config["batch_size"])):
        # construct a batch (maybe just repeat mean map if we even use it)
        batch_gt = []
        batch_random_gaze = []
        batch_mean_gaze = []
        for in_batch_idx in range(start_frame, start_frame + config["batch_size"]):
            if in_batch_idx >= len(dataset_gt):
                break
            batch_gt.append(dataset_gt[in_batch_idx])
            batch_random_gaze.append(dataset_random_gaze[in_batch_idx])
            batch_mean_gaze.append(mean_gaze)
        batch_gt = to_device(to_batch(batch_gt), device)
        batch_random_gaze = to_device(to_batch(batch_random_gaze), device)
        batch_mean_gaze = to_device(to_batch(batch_mean_gaze), device)

        # get the attention ground-truth from the batch
        gaze_gt = batch_gt["output_gaze"]
        gaze_random_gaze = batch_random_gaze["output_gaze"]
        gaze_mean_gaze = batch_mean_gaze["output_gaze"]
        if dataset_gt.output_scaling:
            gaze_gt /= torch.tensor([800.0, 600.0], dtype=gaze_gt.dtype, device=device)
            gaze_random_gaze /= torch.tensor([800.0, 600.0], dtype=gaze_random_gaze.dtype, device=device)
            gaze_mean_gaze /= torch.tensor([800.0, 600.0], dtype=gaze_mean_gaze.dtype, device=device)

        # compute the attention prediction
        output = model(batch_gt)
        gaze_pred = output["output_gaze"]
        if dataset_gt.output_scaling:
            gaze_pred /= torch.tensor([800.0, 600.0], dtype=gaze_pred.dtype, device=device)

        # compute the losses between the "actual" ground-truth and the other stuff
        loss_pred_mse = loss_func_mse(gaze_pred, gaze_gt).item()
        loss_mse_pred_partial_x = loss_func_mse(gaze_pred[:, 0], gaze_gt[:, 0]).item()
        loss_mse_pred_partial_y = loss_func_mse(gaze_pred[:, 1], gaze_gt[:, 1]).item()
        loss_pred_l1 = loss_func_l1(gaze_pred, gaze_gt).item()
        loss_l1_pred_partial_x = loss_func_l1(gaze_pred[:, 0], gaze_gt[:, 0]).item()
        loss_l1_pred_partial_y = loss_func_l1(gaze_pred[:, 1], gaze_gt[:, 1]).item()

        loss_random_gaze_mse = loss_func_mse(gaze_random_gaze, gaze_gt).item()
        loss_mse_random_gaze_partial_x = loss_func_mse(gaze_random_gaze[:, 0], gaze_gt[:, 0]).item()
        loss_mse_random_gaze_partial_y = loss_func_mse(gaze_random_gaze[:, 1], gaze_gt[:, 1]).item()
        loss_random_gaze_l1 = loss_func_l1(gaze_random_gaze, gaze_gt).item()
        loss_l1_random_gaze_partial_x = loss_func_l1(gaze_random_gaze[:, 0], gaze_gt[:, 0]).item()
        loss_l1_random_gaze_partial_y = loss_func_l1(gaze_random_gaze[:, 1], gaze_gt[:, 1]).item()

        loss_mean_gaze_mse = loss_func_mse(gaze_mean_gaze, gaze_gt).item()
        loss_mse_mean_gaze_partial_x = loss_func_mse(gaze_mean_gaze[:, 0], gaze_gt[:, 0]).item()
        loss_mse_mean_gaze_partial_y = loss_func_mse(gaze_mean_gaze[:, 1], gaze_gt[:, 1]).item()
        loss_mean_gaze_l1 = loss_func_l1(gaze_mean_gaze, gaze_gt).item()
        loss_l1_mean_gaze_partial_x = loss_func_l1(gaze_mean_gaze[:, 0], gaze_gt[:, 0]).item()
        loss_l1_mean_gaze_partial_y = loss_func_l1(gaze_mean_gaze[:, 1], gaze_gt[:, 1]).item()

        # sum losses
        total_loss_pred_mse += loss_pred_mse
        total_loss_pred_mse_partial_x += loss_mse_pred_partial_x
        total_loss_pred_mse_partial_y += loss_mse_pred_partial_y
        total_loss_pred_l1 += loss_pred_l1
        total_loss_pred_l1_partial_x += loss_l1_pred_partial_x
        total_loss_pred_l1_partial_y += loss_l1_pred_partial_y

        total_loss_random_gaze_mse += loss_random_gaze_mse
        total_loss_random_gaze_mse_partial_x += loss_mse_random_gaze_partial_x
        total_loss_random_gaze_mse_partial_y += loss_mse_random_gaze_partial_y
        total_loss_random_gaze_l1 += loss_random_gaze_l1
        total_loss_random_gaze_l1_partial_x += loss_l1_random_gaze_partial_x
        total_loss_random_gaze_l1_partial_y += loss_l1_random_gaze_partial_y

        total_loss_mean_gaze_mse += loss_mean_gaze_mse
        total_loss_mean_gaze_mse_partial_x += loss_mse_mean_gaze_partial_x
        total_loss_mean_gaze_mse_partial_y += loss_mse_mean_gaze_partial_y
        total_loss_mean_gaze_l1 += loss_mean_gaze_l1
        total_loss_mean_gaze_l1_partial_x += loss_l1_mean_gaze_partial_x
        total_loss_mean_gaze_l1_partial_y += loss_l1_mean_gaze_partial_y

        num_batches += 1

    print("Average MSE loss for gaze model: {}".format(total_loss_pred_mse / num_batches))
    print("Average partial MSE loss (x-axis) for gaze model: {}".format(total_loss_pred_mse_partial_x / num_batches))
    print("Average partial MSE loss (y-axis) for gaze model: {}".format(total_loss_pred_mse_partial_y / num_batches))

    print("\nAverage L1 loss for gaze model: {}".format(total_loss_pred_l1 / num_batches))
    print("Average partial L1 loss (x-axis) for gaze model: {}".format(total_loss_pred_l1_partial_x / num_batches))
    print("Average partial L1 loss (y-axis) for gaze model: {}".format(total_loss_pred_l1_partial_y / num_batches))

    print("\n-------------------------------------------------------------------")

    print("\nAverage MSE loss for random gaze: {}".format(total_loss_random_gaze_mse / num_batches))
    print("Average partial MSE loss (x-axis) for random gaze: {}".format(total_loss_random_gaze_mse_partial_x / num_batches))
    print("Average partial MSE loss (y-axis) for random gaze: {}".format(total_loss_random_gaze_mse_partial_y / num_batches))

    print("\nAverage L1 loss for random gaze: {}".format(total_loss_random_gaze_l1 / num_batches))
    print("Average partial L1 loss (x-axis) for random gaze: {}".format(total_loss_random_gaze_l1_partial_x / num_batches))
    print("Average partial L1 loss (y-axis) for random gaze: {}".format(total_loss_random_gaze_l1_partial_y / num_batches))

    print("\n-------------------------------------------------------------------")

    print("\nAverage MSE loss for mean gaze: {}".format(total_loss_mean_gaze_mse / num_batches))
    print("Average partial MSE loss (x-axis) for mean gaze: {}".format(total_loss_mean_gaze_mse_partial_x / num_batches))
    print("Average partial MSE loss (y-axis) for mean gaze: {}".format(total_loss_mean_gaze_mse_partial_y / num_batches))

    print("\nAverage L1 loss for mean gaze: {}".format(total_loss_mean_gaze_l1 / num_batches))
    print("Average partial L1 loss (x-axis) for mean gaze: {}".format(total_loss_mean_gaze_l1_partial_x / num_batches))
    print("Average partial L1 loss (y-axis) for mean gaze: {}".format(total_loss_mean_gaze_l1_partial_y / num_batches))


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
    parser.add_argument("-md", "--mode", type=str, default="attention", choices=["attention", "gaze"],
                        help="The path to the model checkpoint to use for computing the predictions.")
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
    arguments = parse_config(parser.parse_args())

    # main
    if arguments["mode"] == "attention":
        evaluate_attention(arguments)
    elif arguments["mode"] == "gaze":
        evaluate_gaze(arguments)
