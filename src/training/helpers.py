import torch

from src.training.loggers import ControlLogger
from src.data.datasets import ImageToControlDataset, ImageAndStateToControlDataset
from src.models.c3d import C3DRegressor
from src.models.codevilla import Codevilla, Codevilla300, CodevillaSkip
from src.models.resnet import ResNetStateRegressor, ResNetRegressor, ResNetStateLargerRegressor
from src.models.utils import image_log_softmax


# TODO: really not sure that this should be here...
def to_device(batch, device, make_batch=False):
    for k in batch:
        if torch.is_tensor(batch[k]):
            if make_batch:
                batch[k] = batch[k].unsqueeze(0)
            batch[k] = batch[k].to(device)
    return batch


def resolve_model_class(model_name):
    return {
        "c3d": C3DRegressor,
        "codevilla": Codevilla,
        "codevilla300": Codevilla300,
        "codevilla_skip": CodevillaSkip,
        "resnet_state": ResNetStateRegressor,
        "resnet": ResNetRegressor,
        "resnet_larger": ResNetStateLargerRegressor
    }[model_name]


def resolve_optimiser_class(optimiser_name):
    return {
        "adam": torch.optim.Adam
    }[optimiser_name]


def get_outputs(dataset_name):
    return {
        "StackedImageToControlDataset": ["output_control"],
        "ImageAndStateToControlDataset": ["output_control"]
    }[dataset_name]


def get_valid_losses(dataset_name):
    # returns lists of lists
    # first level: for which output is the loss for? (e.g. attention, control etc.)
    # second level: what are the valid losses one can choose? (e.g. KL-div, MSE for attention)
    # TODO: might be better to change the first level to dictionaries, so that it can match the output
    #  of the networks that will return this type of output, e.g. something like this:
    """
    return {
        "StackedImageToControlDataset": [["mse"]],
        "ImageAndStateToControlDataset": [["mse"]]
    }[dataset_name]
    """
    return {
        "StackedImageToControlDataset": {"output_control": ["mse"]},
        "ImageAndStateToControlDataset": {"output_control": ["mse"]}
    }[dataset_name]


def resolve_loss(loss_name):
    # TODO: maybe return the class instead, should there be losses with parameters
    return {
        "mse": torch.nn.MSELoss(),
        "kl": torch.nn.KLDivLoss()
    }[loss_name]


def resolve_losses(losses):
    return {output: resolve_loss(loss) for output, loss in losses.items()}


def resolve_output_processing_func(output_name):
    # TODO: I think this should probably be removed and any of that sort of processing moved to the models
    #  themselves or to the loggers (if it needs to be logged in a different format)
    return {
        "output_attention": image_log_softmax,
        "output_control": lambda x: x
    }[output_name]


def resolve_dataset_name(model_name):
    return {
        "c3d": "StackedImageToControlDataset",
        "codevilla": "ImageAndStateToControlDataset",
        "codevilla300": "ImageAndStateToControlDataset",
        "codevilla_skip": "ImageAndStateToControlDataset",
        "resnet_state": "ImageAndStateToControlDataset",
        "resnet": "ImageToControlDataset",
        "resnet_larger": "ImageAndStateToControlDataset"
    }[model_name]


def resolve_dataset_class(dataset_name):
    return {
        "StackedImageToControlDataset": ImageToControlDataset,  # TODO: change this
        "ImageAndStateToControlDataset": ImageAndStateToControlDataset
    }[dataset_name]


def resolve_logger_class(dataset_name):
    return {
        "StackedImageToControlDataset": ControlLogger,
        "ImageAndStateToControlDataset": ControlLogger
    }[dataset_name]


def resolve_resize_parameters(model_name):
    # TODO: might be better to have something more flexible to experiment with different sizes?
    return {
        "c3d": (122, 122),
        "codevilla": 150,
        "codevilla300": 300,
        "codevilla_skip": 150,
        "resnet_state": 300,
        "resnet": 300,
        "resnet_larger": 150
    }[model_name]


def resolve_gt_name(dataset_name):
    return {
        "StackedImageToControlDataset": "drone_control_frame_mean_gt",
        "ImageAndStateToControlDataset": "drone_control_frame_mean_gt"
    }[dataset_name]
