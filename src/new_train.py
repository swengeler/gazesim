import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from tqdm import tqdm
from src.models.utils import image_softmax, image_log_softmax, image_max
from src.config import parse_config, save_config
from src.config import resolve_model_class, resolve_dataset_class, resolve_optimiser_class, resolve_losses


def convert_attention_to_image(attention):
    # divide by the maximum
    maximum = image_max(attention).unsqueeze(-1).unsqueeze(-1)
    return attention / maximum


def prepare_logging(config):
    # creating log and checkpoint directory and saving config file
    log_dir = os.path.join(config["log_root"], config["experiment_name"])
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(tensorboard_dir)
        os.makedirs(checkpoint_dir)
    save_config(config, os.path.join(log_dir, "config.json"))

    # creating tensorboard writer
    tb_writer = SummaryWriter(tensorboard_dir)
    return tb_writer


def train(config):
    # create logging directory, save current config and create the tensorboard writer
    tb_writer = prepare_logging(config)

    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(config["gpu"])
                          if use_cuda and config["gpu"] < torch.cuda.device_count() else "cpu")

    # parameters
    params = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": config.num_workers
    }
    epochs = config.epochs

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

    # define the loss function(s)
    loss_functions = resolve_losses(config["losses"])

    # define the optimiser
    optimiser = resolve_optimiser_class(config["optimiser"])(model.parameters(), lr=config["learning_rate"],
                                                             weight_decay=config["weight_decay"])

    # prepare for doing pass over validation data args.validation_frequency times each epoch
    validation_check = np.linspace(0, len(training_set), config["validation_frequency"] + 1)
    validation_check = np.round(validation_check).astype(int)
    validation_check = validation_check[1:]

    # loop over epochs
    global_step = 0
    for epoch in range(epochs):
        print("Starting epoch {:03d}!".format(epoch))
        running_loss = 0
        model.train()
        validation_current = 0

        for batch_index, (batch, labels) in tqdm(enumerate(training_generator), total=len(training_generator)):
            # transfer to GPU
            if isinstance(batch, list):
                batch = [b.to(device) for b in batch]
                labels = labels.to(device)
            else:
                batch, labels = batch.to(device), labels.to(device)

            # forward pass, loss computation and backward pass
            optimiser.zero_grad()
            predicted_labels = model(batch)
            if "drone_control_gt" not in config.data_type:
                predicted_labels = image_log_softmax(predicted_labels)
            loss = loss_function(predicted_labels, labels)
            # loss = l2_loss_function(predicted_labels, labels)
            loss.backward()
            optimiser.step()

            # tracking total loss over the epoch
            # print(loss.item())
            current_loss = loss.item()
            running_loss += current_loss

            # log to tensorboard
            if isinstance(batch, list):
                global_step += batch[0].shape[0]
            else:
                global_step += batch.shape[0]
            tb_writer.add_scalar("loss/train", current_loss, global_step)

            # logging ground-truth and predicted images every X batches
            # also changed validation to occur here to be able to validate at higher frequency
            with torch.no_grad():
                if "drone_control_gt" not in config.data_type and (batch_index + 1) % config.image_frequency == 0:
                    images_l = convert_attention_to_image(labels)
                    images_p = convert_attention_to_image(image_softmax(predicted_labels))

                    tb_writer.add_images("attention/train/ground_truth", images_l, global_step, dataformats="NCHW")
                    tb_writer.add_images("attention/train/prediction", images_p, global_step, dataformats="NCHW")

                if (global_step - epoch * len(training_set)) >= validation_check[validation_current]:
                    if "drone_control_gt" not in config.data_type:
                        # run validation loop
                        kl_running_loss = 0
                        l2_running_loss = 0
                        log_batch = None
                        model.eval()
                        for val_batch_index, (val_batch, val_labels) in tqdm(enumerate(validation_generator), disable=True):
                            # transfer to GPU
                            if isinstance(batch, list):
                                val_batch = [b.to(device) for b in val_batch]
                                val_labels = val_labels.to(device)
                            else:
                                val_batch, val_labels = val_batch.to(device), val_labels.to(device)

                            # forward pass and recording the losses
                            predicted_labels = model(val_batch)
                            kl_loss = loss_function(image_log_softmax(predicted_labels), val_labels)
                            l2_loss = l2_loss_function(image_softmax(predicted_labels), val_labels)
                            kl_running_loss += kl_loss.item()
                            l2_running_loss += l2_loss.item()

                            if log_batch is None:
                                log_batch = (val_labels, predicted_labels)

                        # printing out the validation loss
                        kl_epoch_loss = kl_running_loss / len(validation_generator)
                        l2_epoch_loss = l2_running_loss / len(validation_generator)

                        # logging the validation loss
                        tb_writer.add_scalar("loss/val/kl", kl_epoch_loss, global_step)
                        tb_writer.add_scalar("loss/val/l2", l2_epoch_loss, global_step)

                        # logging the last batch of ground-truth data and predictions
                        images_l = convert_attention_to_image(log_batch[0])
                        images_p = convert_attention_to_image(image_softmax(log_batch[1]))

                        tb_writer.add_images("attention/val/ground_truth", images_l, global_step,
                                             dataformats="NCHW")
                        tb_writer.add_images("attention/val/prediction", images_p, global_step,
                                             dataformats="NCHW")
                    else:
                        # run validation loop
                        val_running_loss = 0
                        individual_controls = torch.zeros((4,)).to(device)
                        model.eval()
                        for val_batch_index, (val_batch, val_labels) in tqdm(enumerate(validation_generator),
                                                                             disable=True):
                            # transfer to GPU
                            if isinstance(batch, list):
                                val_batch = [b.to(device) for b in val_batch]
                                val_labels = val_labels.to(device)
                            else:
                                val_batch, val_labels = val_batch.to(device), val_labels.to(device)

                            # forward pass and recording the loss
                            predicted_labels = model(val_batch)
                            val_loss = loss_function(predicted_labels, val_labels)
                            val_running_loss += val_loss.item()

                            # compute L2-error for each individual command
                            individual_loss = nn.functional.mse_loss(predicted_labels, val_labels, reduction="none")
                            individual_loss = torch.mean(individual_loss, dim=0)
                            individual_controls += individual_loss

                        # printing out the validation loss
                        val_epoch_loss = val_running_loss / len(validation_generator)
                        individual_controls = individual_controls / len(validation_generator)

                        # logging the validation loss and individual errors
                        tb_writer.add_scalar("loss/val", val_epoch_loss, global_step)
                        tb_writer.add_scalar("loss/val/thrust", individual_controls[0], global_step)
                        tb_writer.add_scalar("loss/val/roll", individual_controls[1], global_step)
                        tb_writer.add_scalar("loss/val/pitch", individual_controls[2], global_step)
                        tb_writer.add_scalar("loss/val/yaw", individual_controls[3], global_step)

                    # update index for checking whether we should run validation loop
                    validation_current += 1
                    model.train()

        # printing out the loss for this epoch
        epoch_loss = running_loss / len(training_generator)
        print("Epoch {:03d} average training loss: {:.8f}".format(epoch, epoch_loss))

        if (epoch + 1) % config.checkpoint_frequency == 0:
            print("Epoch {:03d}: Saving checkpoint to '{}'".format(epoch, checkpoint_dir))
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "epoch_loss": epoch_loss
            }, os.path.join(checkpoint_dir, "epoch{:03d}.pt".format(epoch)))


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
    parser.add_argument("-ivn", "--input_video_names", type=str, default="screen", nargs="+",
                        choices=["screen", "hard_mask_moving_window_mean_frame_gt",
                                 "soft_mask_moving_window_mean_frame_gt"],
                        help="The (file) name(s) for the video(s) to use as input.")

    # arguments related to the model
    parser.add_argument("-m", "--model_name", type=str, default="codevilla", choices=["codevilla", "c3d"],
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
    parser.add_argument("-l", "--losses", type=str, nargs="+", default="mse",
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
