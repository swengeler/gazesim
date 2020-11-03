import re
import json
import torch

from datetime import datetime
from src.data.utils import resolve_split_index_path
from src.training.helpers import resolve_dataset_name, resolve_resize_parameters, get_outputs, get_valid_losses, resolve_gt_name


def parse_config(args):
    config = vars(args)
    if config["config_file"] is not None:
        # load the config file
        # TODO: should there be something similar to the split index files going on, i.e. different configs
        with open(args.config_file, "r") as f:
            loaded_config = json.load(f)

        # get the default values that may not be specified in the loaded config
        for k in config:
            if k not in loaded_config:
                loaded_config[k] = config[k]
        config = loaded_config

    # config entries related to the data to load
    config["split_config"] = resolve_split_index_path(config["split_config"], data_root=config["data_root"])

    # config entries related to the model
    config["model_info"] = None
    if config["model_load_path"] is not None:
        # for now assume that all relevant information is given
        model_info = torch.load(config["model_load_path"], map_location="cuda:{}".format(config["gpu"]))
        config["model_info"] = model_info
        config["model_name"] = model_info["model_name"]
    config["dataset_name"] = resolve_dataset_name(config["model_name"])
    config["resize"] = resolve_resize_parameters(config["model_name"])

    # check that supplied loss(es) are valid
    if not isinstance(config["losses"], dict):
        outputs = get_outputs(config["dataset_name"])
        valid_losses = get_valid_losses(config["dataset_name"])
        updated_losses = {}
        for o_idx, o in enumerate(outputs):
            if o_idx < len(config["losses"]) and config["losses"][o_idx] in valid_losses[o]:
                # if supplied and a valid choice for the loss, take the specified loss
                updated_losses[o] = config["losses"][o_idx]
            else:
                # just take the default
                # TODO: probably add info here (once logging is implemented)
                updated_losses[o] = valid_losses[o][0]
        config["losses"] = updated_losses

    # determine the experiment name to save logs and checkpoints under
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config["experiment_name"] is not None:
        if re.search(r"\d\d-\d\d-\d\d_\d\d-\d\d-\d\d", config["experiment_name"]):
            if len(config["experiment_name"]) >= 18:
                config["experiment_name"] = config["experiment_name"][18:]
            else:
                config["experiment_name"] = ""
        config["experiment_name"] = timestamp + ("_" if len(config["experiment_name"]) > 0 else "") + config["experiment_name"]
    else:
        config["experiment_name"] = timestamp

    # TODO: all of these are pretty hacky right now and need to be cleaned up so that more flexible selection is possible
    config["drone_state_names"] = ["DroneVelocityX", "DroneVelocityY", "DroneVelocityZ",
                                   "DroneAccelerationX", "DroneAccelerationY", "DroneAccelerationZ",
                                   "DroneAngularX", "DroneAngularY", "DroneAngularZ"]
    config["ground_truth_name"] = resolve_gt_name(config["dataset_name"])

    # which parser arguments to keep:
    # data_root
    # data_type => this would pretty much be taken care of by the selection of the split index file...
    # video_name => input_video_names
    # resize_height? maybe should be set automatically based on architecture => should also be called "resize"
    # use_pims? => probably just use as default if there is no reason to use opencv
    # would it make any sense to add a mode in which data is selected dynamically? for now I don't think so

    # TODO: maybe allow using certain abbreviations for model names?
    # model_name => can probably be kept the same
    # model_load_path => add this to resume from => need to check whether there is clash with other
    #                    parameters, but especially with the input data chosen
    # other model parameters: would probably add those when more models are added (could add them as list of pairs,
    # but probably better to have some short prefix for specific parameters)

    # gpu
    # num_workers
    # batch_size
    # epochs
    # optimiser => add this
    # learning_rate => optimiser_lr
    # weight_decay => we can probably leave it out for now, doesn't really seem to be used a lot
    # other (e.g. optimiser, loss) parameters might be added later but this is probably all that's needed for now
    # are there any choices when it comes to loss? I guess one could e.g. do either KL-divergence or MSE
    # for attention... should maybe be an option => then we'd also need the respective transforms somewhere

    # log_root => can stay the same
    # image_frequency => not sure if this will even be kept, this will probably become a new set of parameters for the
    #                    "metrics" to log for different models (if there will even be any that aren't hard-coded)
    # validation_frequency => actually pretty happy with how this one has worked out
    # check_point_frequency => I'd like to keep this using whole epochs instead of doing the same thing as
    #                          validation_frequency but maybe checking different things that should be updated
    #                          periodically would be more consistent? although I think there are few scenarios
    #                          (especially with more diverse data) in which we would want to save between epochs
    # need to add experiment name or something of the sort

    # convert argparse namespace to dictionary and maybe change some of the entries
    # TODO: if config file is provided, should just load it and only complain if anything is missing
    return config
