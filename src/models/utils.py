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


def convert_attention_to_image(attention, out_shape=None):
    assert len(attention.shape) == 4, \
        "Tensor must have exactly 4 dimensions (batch, channel, height, width in some order)."

    # make sure that we are not doing integer types (because of the division)
    attention = attention.double()

    # make sure all the dimensions are correct
    if not attention.shape[1] == 3 and attention.shape[3] == 3:
        attention = attention.permute(0, 3, 1, 2)

    # resize the image if necessary
    if out_shape is not None and (attention.shape[2] != out_shape[0] or attention.shape[3] != out_shape[1]):
        attention = torch.nn.functional.interpolate(attention, out_shape)

    # TODO: maybe also add color channels

    # divide by the maximum
    maximum = image_max(attention).unsqueeze(-1).unsqueeze(-1)
    return attention / maximum
