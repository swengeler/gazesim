import torch
import torch.nn.functional as F


def _image_apply_function(tensor, function, **kwargs):
    # assumes that the last two dimensions are the image dimensions H x W
    original_shape = tuple(tensor.shape)
    tensor = torch.reshape(tensor, original_shape[:-2] + (-1,))
    tensor = function(tensor, **kwargs)
    tensor = torch.reshape(tensor, original_shape)
    return tensor


def _image_apply_reduction(tensor, reduction, **kwargs):
    # assumes that the last two dimensions are the image dimensions H x W
    original_shape = tuple(tensor.shape)
    tensor = torch.reshape(tensor, original_shape[:-2] + (-1,))
    tensor = reduction(tensor, **kwargs)
    return tensor


def image_max(tensor):
    tensor, _ = _image_apply_reduction(tensor, torch.max, dim=-1)
    return tensor


def image_softmax(tensor):
    return _image_apply_function(tensor, F.softmax, dim=-1)


def image_log_softmax(tensor):
    return _image_apply_function(tensor, F.log_softmax, dim=-1)


def convert_attention_to_image(attention):
    # divide by the maximum
    maximum = image_max(attention).unsqueeze(-1).unsqueeze(-1)
    return attention / maximum
