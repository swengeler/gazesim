import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from tqdm import tqdm
from src.data.old_datasets import get_dataset
from src.models.vgg import VGG16BaseModel
from src.models.resnet import ResNet18BaseModel, ResNet18BaseModelSimple
from src.models.resnet import ResNet18Regressor, ResNet18SimpleRegressor, ResNet18DualBranchRegressor
from src.models.c3d import C3DRegressor
from src.models.utils import image_softmax, image_log_softmax, image_max


def resolve_model_class(name):
    if name == "vgg16":
        return VGG16BaseModel
    elif name == "resnet18":
        return ResNet18BaseModel
    elif name == "resnet18_simple":
        return ResNet18BaseModelSimple
    elif name == "resnet18_regressor":
        return ResNet18Regressor
    elif name == "resnet18_simple_regressor":
        return ResNet18SimpleRegressor
    elif name == "resnet18_dual":
        return ResNet18DualBranchRegressor
    elif name == "c3d":
        return C3DRegressor
    return VGG16BaseModel


def convert_attention_to_image(attention):
    # divide by the maximum
    maximum = image_max(attention).unsqueeze(-1).unsqueeze(-1)
    return attention / maximum


def parse_config(args):
    # convert argparse namespace to dictionary and maybe change some of the entries
    pass


def train(args):
    # creating log and checkpoint directory and saving "config" file
    run_dir = os.path.join(args.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        os.makedirs(tensorboard_dir)
        os.makedirs(checkpoint_dir)

        args_dict = vars(args)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(args_dict, f)

    # creating tensorboard writer
    tb_writer = SummaryWriter(tensorboard_dir)

    # use GPU if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda and args.gpu < torch.cuda.device_count() else "cpu")

    # parameters
    params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers
    }
    epochs = args.epochs

    # generators
    training_set = get_dataset(
        data_root=args.data_root,
        split="train",
        data_type=args.data_type,
        video_name=args.video_name,
        resize_height=args.resize_height,
        use_pims=args.use_pims
    )
    training_generator = DataLoader(training_set, **params)

    validation_set = get_dataset(
        data_root=args.data_root,
        split="val",
        data_type=args.data_type,
        video_name=args.video_name,
        resize_height=args.resize_height,
        use_pims=args.use_pims
    )
    validation_generator = DataLoader(validation_set, **params)

    # define the model
    model_class = resolve_model_class(args.model_name)
    model = model_class(transfer_weights=(not args.not_pretrained))
    model = model.to(device)

    # define the loss function(s)
    if "drone_control_gt" in args.data_type:
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.KLDivLoss()
    l2_loss_function = nn.MSELoss()

    # define the optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # prepare for doing pass over validation data args.validation_frequency times each epoch
    validation_check = np.linspace(0, len(training_set), args.validation_frequency + 1)
    validation_check = np.round(validation_check).astype(int)
    validation_check = validation_check[1:]
    # print("VALIDATION_CHECK:", validation_check)

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
            if "drone_control_gt" not in args.data_type:
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
                if "drone_control_gt" not in args.data_type and (batch_index + 1) % args.image_frequency == 0:
                    images_l = convert_attention_to_image(labels)
                    images_p = convert_attention_to_image(image_softmax(predicted_labels))

                    tb_writer.add_images("attention/train/ground_truth", images_l, global_step, dataformats="NCHW")
                    tb_writer.add_images("attention/train/prediction", images_p, global_step, dataformats="NCHW")

                if (global_step - epoch * len(training_set)) >= validation_check[validation_current]:
                    if "drone_control_gt" not in args.data_type:
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

        if (epoch + 1) % args.checkpoint_frequency == 0:
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
    parser.add_argument("-t", "--data_type", type=str, default="turn_left",
                        choices=["turn_left", "turn_right", "turn_both", "turn_left_drone_control_gt",
                                 "turn_right_drone_control_gt", "turn_both_drone_control_gt",
                                 "soft_mask", "hard_mask", "mean_mask"],
                        help="The type of turn to train on (left or right).")
    parser.add_argument("-vn", "--video_name", type=str, default="screen", nargs="+",
                        choices=["screen", "hard_mask_moving_window_gt",
                                 "mean_mask_moving_window_gt", "soft_mask_moving_window_gt"],
                        help="The type of turn to train on (left or right).")
    parser.add_argument("-rh", "--resize_height", type=int, default=150, nargs="+",
                        help="Height that input images and the ground-truth are rescaled to (with width being "
                             "adjusted accordingly). For VGG16 this should be 150, for ResNet18 200.")
    parser.add_argument("--use_pims", action="store_true",
                        help="Whether to use PIMS (PyAV) instead of OpenCV for reading frames.")

    # arguments related to the model
    parser.add_argument("-m", "--model_name", type=str, default="vgg16",
                        choices=["vgg16", "resnet18", "resnet18_simple", "resnet18_regressor",
                                 "resnet18_simple_regressor", "resnet18_dual", "c3d"],
                        help="The name of the model to use (only VGG16 and ResNet18 available currently).")
    parser.add_argument("-np", "--not_pretrained", action="store_true",
                        help="Disable using pretrained weights for the encoder where available.")
    parser.add_argument("-ar", "--activate_regressor", action="store_true",
                        help="Whether to use activation functions for the regressor output.")

    # arguments related to training
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="GPU to use for training if any are available.")
    parser.add_argument("-w", "--num_workers", type=int, default=2,
                        help="Number of workers to use for loading the data.")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size to use for training.")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Maximum number of epochs to train for.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                        help="Learning rate (only for Adam optimiser for now).")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0,
                        help="Weight decay (only for Adam optimiser for now).")

    # arguments related to logging information
    parser.add_argument("-lg", "--log_root", type=str, default=os.getenv("GAZESIM_LOG"),
                        help="Root directory where log folders for each run should be created.")
    parser.add_argument("-if", "--image_frequency", type=int, default=50,
                        help="Frequency at which to log ground-truth and predicted attention as images (in batches).")
    parser.add_argument("-vf", "--validation_frequency", type=int, default=1,
                        help="How often to compute the validation loss during each epoch. When set to 1 "
                             "(the default value) this is only done at the end of the epoch, as is standard.")
    parser.add_argument("-cf", "--checkpoint_frequency", type=int, default=1,
                        help="Frequency at which to save model checkpoints (in epochs).")

    # parse the arguments
    arguments = parser.parse_args()
    if len(arguments.video_name) == 1:
        arguments.video_name = arguments.video_name[0]
    if len(arguments.resize_height) == 1:
        arguments.resize_height = arguments.resize_height[0]
    else:
        arguments.resize_height = tuple(arguments.resize_height)

    # train
    train(arguments)