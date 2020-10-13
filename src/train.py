import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from tqdm import tqdm
from src.data.dataset import get_dataset
from src.models.vgg16 import VGG16BaseModel
from src.models.utils import image_softmax, image_log_softmax


def train(args):
    # creating log and checkpoint directory and saving "config" file
    run_dir = os.path.join(arguments.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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
    training_set = get_dataset(data_root=args.data_root, split="train", name=args.turn_type)
    training_generator = DataLoader(training_set, **params)

    validation_set = get_dataset(data_root=args.data_root, split="val", name=args.turn_type)
    validation_generator = DataLoader(validation_set, **params)

    # define the model
    model = VGG16BaseModel()
    model = model.to(device)

    # define the loss function(s)
    loss_function = nn.KLDivLoss()
    l2_loss_function = nn.MSELoss()

    # define the optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # loop over epochs
    global_step = 0
    for epoch in range(epochs):
        # training
        print("Starting epoch {:03d}!".format(epoch))
        running_loss = 0
        for batch_index, (local_batch, local_labels) in tqdm(enumerate(training_generator)):
            # transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # forward pass, loss computation and backward pass
            optimiser.zero_grad()
            predicted_labels = model(local_batch)
            predicted_labels = image_log_softmax(predicted_labels)
            loss = loss_function(predicted_labels, local_labels)
            # loss = l2_loss_function(predicted_labels, local_labels)
            loss.backward()
            optimiser.step()

            # tracking total loss over the epoch
            # print(loss.item())
            current_loss = loss.item()
            running_loss += current_loss

            # log to tensorboard
            global_step += local_batch.shape[0]
            tb_writer.add_scalar("loss/train", current_loss, global_step)

            # printing out loss every X batches
            """
            if (batch_index + 1) % arguments.print_frequency == 0:
                current_loss = running_loss / ((batch_index + 1) * arguments.batch_size)
                print("    Batch {:04d} (epoch {:03d}) training loss: {:.8f}".format(batch_index + 1, epoch, current_loss))
            """

        # printing out the loss for this epoch
        epoch_loss = running_loss / len(training_generator)
        print("Epoch {:03d} average training loss: {:.4f}".format(epoch, epoch_loss))

        if (epoch + 1) % arguments.checkpoint_frequency == 0:
            print("Epoch {:03d}: Saving checkpoint to '{}'".format(epoch, arguments.log_root))
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "epoch_loss": epoch_loss
            }, os.path.join(checkpoint_dir, "epoch{:03d}.pt".format(epoch)))

        # validation
        print("Computing loss on validation data for epoch {:03d}!".format(epoch))
        with torch.no_grad():
            kl_running_loss = 0
            l2_running_loss = 0

            for local_batch, local_labels in tqdm(validation_generator):
                # transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # forward pass and recording the losses
                predicted_labels = model(local_batch)
                kl_loss = loss_function(predicted_labels, local_labels)
                l2_loss = l2_loss_function(image_softmax(predicted_labels), local_labels)
                kl_running_loss += kl_loss.item()
                l2_running_loss += l2_loss.item()

            # printing out the validation loss
            kl_epoch_loss = kl_running_loss / len(validation_generator)
            l2_epoch_loss = l2_running_loss / len(validation_generator)
            print("Epoch {:03d} average KL-divergence validation loss: {:.4f}".format(epoch, kl_epoch_loss))
            print("Epoch {:03d} average MSE validation loss: {:.4f}".format(epoch, l2_epoch_loss))

            # logging the validation loss
            tb_writer.add_scalar("loss/val/kl", kl_epoch_loss, global_step)
            tb_writer.add_scalar("loss/val/l2", l2_epoch_loss, global_step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # arguments related to the dataset
    parser.add_argument("-r", "--data_root", type=str, default=os.getenv("GAZESIM_ROOT"),
                        help="The root directory of the dataset (should contain only subfolders for each subject).")
    parser.add_argument("-t", "--turn_type", type=str, default="turn_left", choices=["turn_left", "turn_right"],
                        help="The type of turn to train on (left or right).")

    # arguments related to training
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="GPU to use for training if any are available.")
    parser.add_argument("-w", "--num_workers", type=int, default=2,
                        help="Number of workers to use for loading the data.")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size to use for training.")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Maximum number of epochs to train for.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="Learning rate (only for Adam optimiser for now).")

    # arguments related to logging information
    parser.add_argument("-lg", "--log_root", type=str, default=os.environ["GAZESIM_LOG"],
                        help="Root directory where log folders for each run should be created.")
    parser.add_argument("-pf", "--print_frequency", type=int, default=200,
                        help="Frequency at which to print the current average loss (in batches).")
    parser.add_argument("-cf", "--checkpoint_frequency", type=int, default=1,
                        help="Frequency at which to save model checkpoints (in epochs).")

    # parse the arguments
    arguments = parser.parse_args()

    # train
    train(arguments)
