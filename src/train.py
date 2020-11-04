import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from src.training.config import parse_config
from src.training.helpers import to_device, resolve_model_class, resolve_dataset_class, resolve_optimiser_class
from src.training.helpers import resolve_losses, resolve_output_processing_func, resolve_logger_class


def train(config):
    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(config["gpu"])
                          if use_cuda and config["gpu"] < torch.cuda.device_count() else "cpu")

    # generators
    training_set = resolve_dataset_class(config["dataset_name"])(config, split="train")
    training_generator = DataLoader(training_set, batch_size=config["batch_size"],
                                    shuffle=True, num_workers=config["num_workers"])

    validation_set = resolve_dataset_class(config["dataset_name"])(config, split="val")
    validation_generator = DataLoader(validation_set, batch_size=config["batch_size"],
                                      shuffle=True, num_workers=config["num_workers"])

    # define the model
    model_class = resolve_model_class(config["model_name"])
    model = model_class(config)
    model = model.to(device)

    # define the optimiser
    optimiser = resolve_optimiser_class(config["optimiser"])(model.parameters(), lr=config["learning_rate"])

    # define the loss function(s)
    loss_functions = resolve_losses(config["losses"])

    # define the logger
    logger = resolve_logger_class(config["dataset_name"])(config, model, training_set)

    # prepare for doing pass over validation data args.validation_frequency times each epoch
    validation_check = np.linspace(0, len(training_set), config["validation_frequency"] + 1)
    validation_check = np.round(validation_check).astype(int)
    validation_check = validation_check[1:]

    # loop over epochs
    global_step = 0
    for epoch in range(config["num_epochs"]):
        print("Starting epoch {:03d}!".format(epoch))
        model.train()
        validation_current = 0

        for batch_index, batch in tqdm(enumerate(training_generator), total=len(training_generator)):
            # transfer to GPU
            batch = to_device(batch, device)

            # forward pass, loss computation and backward pass
            optimiser.zero_grad()
            predictions = model(batch)
            loss = None
            for output in predictions:
                current_prediction = resolve_output_processing_func(output)(predictions[output])
                current_loss = loss_functions[output](current_prediction, batch[output])
                if loss is None:
                    loss = current_loss
                else:
                    loss += current_loss
            loss.backward()
            optimiser.step()

            with torch.no_grad():
                global_step += batch[sorted(batch.keys())[0]].shape[0]

                # log at the end of each training step (each batch)
                # scalar_loss = loss.item()
                logger.training_step_end(global_step, loss, batch, predictions)

                # do validation if it should be done
                if (global_step - epoch * len(training_set)) >= validation_check[validation_current]:
                    disable = True
                    if config["validation_frequency"] == 1:
                        print("Validation for epoch {:03d}!".format(epoch))
                        disable = False

                    model.eval()
                    for val_batch_index, val_batch in tqdm(enumerate(validation_generator), disable=disable):
                        # transfer to GPU
                        val_batch = to_device(val_batch, device)

                        # forward pass and loss computation
                        val_predictions = model(val_batch)
                        val_loss = None
                        for output in val_predictions:
                            current_prediction = resolve_output_processing_func(output)(val_predictions[output])
                            current_loss = loss_functions[output](current_prediction, val_batch[output])
                            if val_loss is None:
                                val_loss = current_loss
                            else:
                                val_loss += current_loss

                        # tracking the loss in the logger
                        # val_scalar_loss = val_loss.item()
                        logger.validation_step_end(global_step, val_loss, val_batch, val_predictions)

                    # log after the complete pass over the validation set
                    logger.validation_epoch_end(global_step, epoch, model, optimiser)

                    # update index for checking whether we should run validation loop
                    validation_current += 1
                    model.train()

        # log at the end of the epoch
        logger.training_epoch_end(global_step, epoch, model, optimiser)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # arguments related to the dataset
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-sc", "--split_config", default=0,
                        help="The split configuration/index to get information about the division into training "
                             "and validation (and test) data from. Can either be the path to a file or an index "
                             "(will search in $DATA_ROOT/splits/).")
    parser.add_argument("-ivn", "--input_video_names", type=str, nargs="+", default=["screen"],
                        choices=["screen", "hard_mask_moving_window_mean_frame_gt",
                                 "soft_mask_moving_window_mean_frame_gt"],
                        help="The (file) name(s) for the video(s) to use as input.")
    parser.add_argument("-gtn", "--ground_truth_name", type=str, default="moving_window_mean_frame_gt",
                        choices=["screen", "hard_mask_moving_window_mean_frame_gt",
                                 "soft_mask_moving_window_mean_frame_gt"],
                        help="The (file) name(s) for the video(s) to use as input.")
    parser.add_argument("-c", "--config_file", type=str,
                        help="Config file to load parameters from.")

    # arguments related to the model
    parser.add_argument("-m", "--model_name", type=str, default="codevilla",
                        choices=["codevilla", "c3d", "codevilla300", "resnet_state", "resnet"],
                        help="The name of the model to use.")
    parser.add_argument("-mlp", "--model_load_path", type=str,
                        help="Path to load a model checkpoint from (including information about the "
                             "architecture, the current weights and the state of the optimiser).")

    # arguments related to training
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="GPU to use for training if any are available.")
    parser.add_argument("-w", "--num_workers", type=int, default=4,
                        help="Number of workers to use for loading the data.")
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Batch size to use for training.")
    parser.add_argument("-e", "--num_epochs", type=int, default=5,
                        help="Maximum number of epochs to train for.")
    parser.add_argument("-o", "--optimiser", type=str, default="adam", choices=["adam"],
                        help="The optimiser to use.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                        help="The learning rate to start with.")
    parser.add_argument("-l", "--losses", type=str, nargs="+", default=["mse"],
                        help="The loss to use. Depends on the model architecture and what kinds of outputs "
                             "(and how many) it has. For now only one loss can be specified (no architecture "
                             "with multiple outputs/losses). If the wrong loss is supplied, it will be changed "
                             "automatically to the default loss for a given architecture/output type.")

    # arguments related to logging information
    parser.add_argument("-lg", "--log_root", type=str, default=os.getenv("GAZESIM_LOG"),
                        help="Root directory where log folders for each run should be created.")
    parser.add_argument("-exp", "--experiment_name", type=str,
                        help="The name under which to save the logs and checkpoints (in addition to a timestamp).")
    parser.add_argument("-vf", "--validation_frequency", type=int, default=1,
                        help="How often to compute the validation loss during each epoch. When set to 1 "
                             "(the default value) this is only done at the end of the epoch, as is standard.")
    parser.add_argument("-cf", "--checkpoint_frequency", type=int, default=1,
                        help="Frequency at which to save model checkpoints (in epochs).")

    # parse the arguments
    arguments = parser.parse_args()

    # train
    train(parse_config(arguments))
