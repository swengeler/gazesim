import torch
import torch.nn.functional as F


def _image_apply_function(tensor, function):
    # assumes that the last two dimensions are the image dimensions H x W
    original_shape = tuple(tensor.shape)
    tensor = torch.reshape(tensor, original_shape[:-2] + (-1,))
    tensor = function(tensor, dim=-1)
    tensor = torch.reshape(tensor, original_shape)
    return tensor


def image_softmax(tensor):
    return _image_apply_function(tensor, F.softmax)


def image_log_softmax(tensor):
    return _image_apply_function(tensor, F.log_softmax)
