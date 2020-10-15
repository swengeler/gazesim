import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from tqdm import tqdm
from src.data.dataset import get_dataset
from src.models.vgg import VGG16BaseModel
from src.models.resnet import ResNet18BaseModel, ResNet18BaseModelSimple
from src.models.utils import image_softmax, image_log_softmax, image_max


def resolve_model_class(name):
    if name == "vgg16":
        return VGG16BaseModel
    elif name == "resnet18":
        return ResNet18BaseModel
    elif name == "resnet18_simple":
        return ResNet18BaseModelSimple
    return VGG16BaseModel


def convert_attention_to_image(attention):
    # divide by the maximum
    maximum = image_max(attention).unsqueeze(-1).unsqueeze(-1)
    return attention / maximum


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
        name=args.turn_type,
        resize_height=args.resize_height,
        use_pims=args.use_pims
    )
    training_generator = DataLoader(training_set, **params)

    validation_set = get_dataset(
        data_root=args.data_root,
        split="val",
        name=args.turn_type,
        resize_height=args.resize_height,
        use_pims=args.use_pims
    )
    validation_generator = DataLoader(validation_set, **params)

    # define the model
    model_class = resolve_model_class(args.model_name)
    model = model_class(transfer_weights=(not args.not_pretrained))  # TODO: should be able to input arguments here as well
    model = model.to(device)

    # define the loss function(s)
    loss_function = nn.KLDivLoss()
    l2_loss_function = nn.MSELoss()

    # define the optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # prepare for doing pass over validation data args.validation_frequency times each epoch
    validation_check = np.linspace(0, len(training_generator), args.validation_frequency + 1)
    validation_check = np.round(validation_check).astype(int)
    validation_check = validation_check[1:]

    # loop over epochs
    global_step = 0
    for epoch in range(epochs):
        print("Starting epoch {:03d}!".format(epoch))
        running_loss = 0
        model.train()
        validation_current = 0

        for batch_index, (batch, labels) in tqdm(enumerate(training_generator)):
            # transfer to GPU
            batch, labels = batch.to(device), labels.to(device)

            # forward pass, loss computation and backward pass
            optimiser.zero_grad()
            predicted_labels = model(batch)
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
            global_step += batch.shape[0]
            tb_writer.add_scalar("loss/train", current_loss, global_step)

            # logging ground-truth and predicted images every X batches
            # also changed validation to occur here to be able to validate at higher frequency
            with torch.no_grad():
                if (batch_index + 1) % args.image_frequency == 0:
                    images_l = convert_attention_to_image(labels)
                    images_p = convert_attention_to_image(image_softmax(predicted_labels))

                    tb_writer.add_images("attention/train/ground_truth", images_l, global_step, dataformats="NCHW")
                    tb_writer.add_images("attention/train/prediction", images_p, global_step, dataformats="NCHW")

                if (global_step - epoch * len(training_generator)) >= validation_check[validation_current]:
                    # run validation loop
                    kl_running_loss = 0
                    l2_running_loss = 0
                    model.eval()
                    for val_batch_index, (val_batch, val_labels) in tqdm(enumerate(validation_generator)):
                        # transfer to GPU
                        val_batch, val_labels = val_batch.to(device), val_labels.to(device)

                        # forward pass and recording the losses
                        predicted_labels = model(val_batch)
                        kl_loss = loss_function(image_log_softmax(predicted_labels), val_labels)
                        l2_loss = l2_loss_function(image_softmax(predicted_labels), val_labels)
                        kl_running_loss += kl_loss.item()
                        l2_running_loss += l2_loss.item()

                    # printing out the validation loss
                    kl_epoch_loss = kl_running_loss / len(validation_generator)
                    l2_epoch_loss = l2_running_loss / len(validation_generator)
                    # print("Epoch {:03d} average KL-divergence validation loss: {:.8f}".format(epoch, kl_epoch_loss))
                    # print("Epoch {:03d} average MSE validation loss: {:.8f}".format(epoch, l2_epoch_loss))

                    # logging the validation loss
                    tb_writer.add_scalar("loss/val/kl", kl_epoch_loss, global_step)
                    tb_writer.add_scalar("loss/val/l2", l2_epoch_loss, global_step)

                    # logging the last batch of ground-truth data and predictions
                    images_l = convert_attention_to_image(val_labels)
                    images_p = convert_attention_to_image(image_softmax(predicted_labels))

                    tb_writer.add_images("attention/val/ground_truth", images_l, global_step,
                                         dataformats="NCHW")
                    tb_writer.add_images("attention/val/prediction", images_p, global_step,
                                         dataformats="NCHW")

                    # update index for checking whether we should run validation loop
                    validation_current += 1

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
    parser.add_argument("-t", "--turn_type", type=str, default="turn_left", choices=["turn_left", "turn_right"],
                        help="The type of turn to train on (left or right).")
    parser.add_argument("-rh", "--resize_height", type=int, default=150,
                        help="Height that input images and the ground-truth are rescaled to (with width being "
                             "adjusted accordingly). For VGG16 this should be 150, for ResNet18 200.")
    parser.add_argument("--use_pims", action="store_true",
                        help="Whether to use PIMS (PyAV) instead of OpenCV for reading frames.")

    # arguments related to training
    parser.add_argument("-m", "--model_name", type=str, default="vgg16", choices=["vgg16", "resnet18", "resnet18_simple"],
                        help="The name of the model to use (only VGG16 and ResNet18 available currently).")
    parser.add_argument("-np", "--not_pretrained", action="store_true",
                        help="Disable using pretrained weights for the encoder where available.")
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
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.00005,
                        help="Weight decay (only for Adam optimiser for now).")

    # arguments related to logging information
    parser.add_argument("-lg", "--log_root", type=str, default=os.environ["GAZESIM_LOG"],
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

    # train
    train(arguments)
