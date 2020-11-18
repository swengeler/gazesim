import numpy as np
import torch

from src.training.loggers import ControlLogger, TestLogger, AttentionLogger
from src.data.datasets import ImageToControlDataset, ImageAndStateToControlDataset, StateToControlDataset, StackedImageAndStateToControlDataset
from src.data.datasets import StackedImageToAttentionDataset
from src.models.c3d import C3DRegressor, C3DStateRegressor
from src.models.codevilla import Codevilla, Codevilla300, CodevillaSkip, CodevillaMultiHead, CodevillaDualBranch, CodevillaMultiHeadNoState
from src.models.resnet import ResNetStateRegressor, ResNetRegressor, ResNetStateLargerRegressor, StateOnlyRegressor, ResNetLargerRegressor
from src.models.dreyeve import SaliencyBranch, DrEYEveNet
from src.models.utils import image_log_softmax


def get_batch_size(batch):
    if isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        return batch.shape[0]
    elif isinstance(batch, dict):
        return get_batch_size(batch[list(batch.keys())[0]])
    elif isinstance(batch, list):
        return get_batch_size(batch[0])
    return np.nan


# TODO: really not sure that this should be here...
def to_device(batch, device, make_batch=False):
    # TODO: maybe add something for a list too...
    for k in batch:
        if torch.is_tensor(batch[k]):
            if make_batch:
                batch[k] = batch[k].unsqueeze(0)
            batch[k] = batch[k].to(device)
        elif isinstance(batch[k], dict):
            batch[k] = to_device(batch[k], device, make_batch)
    return batch


def resolve_model_class(model_name):
    return {
        "c3d": C3DRegressor,
        "c3d_state": C3DStateRegressor,
        "codevilla": Codevilla,
        "codevilla300": Codevilla300,
        "codevilla_skip": CodevillaSkip,
        "codevilla_multi_head": CodevillaMultiHead,
        "codevilla_dual_branch": CodevillaDualBranch,
        "codevilla_no_state": CodevillaMultiHeadNoState,
        "resnet_state": ResNetStateRegressor,
        "resnet": ResNetRegressor,
        "resnet_larger": ResNetLargerRegressor,
        "resnet_state_larger": ResNetStateLargerRegressor,
        "state_only": StateOnlyRegressor,
        "dreyeve_branch": SaliencyBranch
    }[model_name]


def resolve_optimiser_class(optimiser_name):
    return {
        "adam": torch.optim.Adam
    }[optimiser_name]


def get_outputs(dataset_name):
    return {
        "StackedImageToControlDataset": ["output_control"],
        "StackedImageAndStateToControlDataset": ["output_control"],
        "ImageToControlDataset": ["output_control"],
        "ImageAndStateToControlDataset": ["output_control"],
        "StateToControlDataset": ["output_control"],
        "StackedImageToAttentionDataset": ["output_attention", "output_attention_crop"]
        # TODO: this might actually depend on more than just this  (e.g. if some dreyeve architecture is used)
    }[dataset_name]


def get_valid_losses(dataset_name):
    # first level: for which output is the loss for? (e.g. attention, control etc.)
    # second level: what are the valid losses one can choose? (e.g. KL-div, MSE for attention)
    return {
        "StackedImageToControlDataset": {"output_control": ["mse"]},
        "StackedImageAndStateToControlDataset": {"output_control": ["mse"]},
        "ImageToControlDataset": {"output_control": ["mse"]},
        "ImageAndStateToControlDataset": {"output_control": ["mse"]},
        "StateToControlDataset": {"output_control": ["mse"]},
        "StackedImageToAttentionDataset": {"output_attention": ["kl", "mse"], "output_attention_crop": ["kl", "mse"]}
    }[dataset_name]


def resolve_loss(loss_name):
    # TODO: maybe return the class instead, should there be losses with parameters
    return {
        "mse": torch.nn.MSELoss(),
        "kl": torch.nn.KLDivLoss(reduction="batchmean")
    }[loss_name]


def resolve_losses(losses):
    return {output: resolve_loss(loss) for output, loss in losses.items()}


def resolve_output_processing_func(output_name):
    # TODO: I think this should probably be removed and any of that sort of processing moved to the models
    #  themselves or to the loggers (if it needs to be logged in a different format)
    return {
        "output_attention": image_log_softmax,
        "output_attention_crop": image_log_softmax,
        # TODO: might not be the best way to do this, would be nicer if output could be
        #  structured as a nested dictionary as well and this would be compatible with that...
        "output_control": lambda x: x
    }[output_name]


def resolve_dataset_name(model_name):
    return {
        "c3d": "StackedImageToControlDataset",
        "c3d_state": "StackedImageAndStateToControlDataset",
        "codevilla": "ImageAndStateToControlDataset",
        "codevilla300": "ImageAndStateToControlDataset",
        "codevilla_skip": "ImageAndStateToControlDataset",
        "codevilla_multi_head": "ImageAndStateToControlDataset",
        "codevilla_dual_branch": "ImageAndStateToControlDataset",
        "codevilla_no_state": "ImageToControlDataset",
        "resnet_state": "ImageAndStateToControlDataset",
        "resnet": "ImageToControlDataset",
        "resnet_larger": "ImageToControlDataset",
        "resnet_state_larger": "ImageAndStateToControlDataset",
        "state_only": "StateToControlDataset",
        "dreyeve_branch": "StackedImageToAttentionDataset"
    }[model_name]


def resolve_dataset_class(dataset_name):
    return {
        "StackedImageToControlDataset": ImageToControlDataset,  # TODO: change this
        "StackedImageAndStateToControlDataset": StackedImageAndStateToControlDataset,
        "ImageToControlDataset": ImageToControlDataset,
        "ImageAndStateToControlDataset": ImageAndStateToControlDataset,
        "StateToControlDataset": StateToControlDataset,
        "StackedImageToAttentionDataset": StackedImageToAttentionDataset
    }[dataset_name]


def resolve_logger_class(dataset_name):
    return {
        "StackedImageToControlDataset": ControlLogger,
        "StackedImageAndStateToControlDataset": ControlLogger,
        "ImageToControlDataset": ControlLogger,
        "ImageAndStateToControlDataset": ControlLogger,
        "StateToControlDataset": ControlLogger,
        "StackedImageToAttentionDataset": AttentionLogger
    }[dataset_name]


def resolve_resize_parameters(model_name):
    # TODO: might be better to have something more flexible to experiment with different sizes?
    return {
        "c3d": (122, 122),
        "c3d_state": (122, 122),
        "codevilla": 150,
        "codevilla300": 300,
        "codevilla_skip": 150,
        "codevilla_multi_head": 150,
        "codevilla_dual_branch": 150,
        "codevilla_no_state": 150,
        "resnet_state": 300,
        "resnet": 300,
        "resnet_larger": 150,
        "resnet_state_larger": 150,
        "state_only": None,
        "dreyeve_branch": None
    }[model_name]


def resolve_gt_name(dataset_name):
    return {
        "StackedImageToControlDataset": "drone_control_frame_mean_gt",
        "StackedImageAndStateToControlDataset": "drone_control_frame_mean_gt",
        "ImageToControlDataset": "drone_control_frame_mean_gt",
        "ImageAndStateToControlDataset": "drone_control_frame_mean_gt",
        "StateToControlDataset": "drone_control_frame_mean_gt",
        "StackedImageToAttentionDataset": "moving_window_frame_mean_gt"
    }[dataset_name]
