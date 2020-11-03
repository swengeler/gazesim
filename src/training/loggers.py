import os
import json
import torch

from torch.utils.tensorboard import SummaryWriter


def save_config(config, config_save_path):
    # remove those entries that cannot be saved with json
    excluded = ["model_info"]
    config_to_save = {}
    for k in config:
        if k not in excluded:
            config_to_save[k] = config[k]

    # save the config
    with open(config_save_path, "w") as f:
        json.dump(config_to_save, f)


class Logger:

    def __init__(self, config, model, dataset):
        # create log and checkpoint directory
        self.log_dir = os.path.join(config["log_root"], config["experiment_name"])
        self.tensorboard_dir = os.path.join(self.log_dir, "tensorboard")
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            os.makedirs(self.tensorboard_dir)
            os.makedirs(self.checkpoint_dir)

        # create tensorboard writer
        self.tb_writer = SummaryWriter(self.tensorboard_dir)

        # store config for information and save config file
        self.config = config
        save_config(self.config, os.path.join(self.log_dir, "config.json"))

    def training_step_end(self, global_step, loss, batch, predictions):
        # to be called when a single training step (1 batch) is performed
        raise NotImplementedError()

    def training_epoch_end(self, global_step, epoch, model, optimiser):
        # to be called after each full pass over the training set
        raise NotImplementedError()

    def validation_step_end(self, global_step, loss, batch, predictions):
        # to be called when a single validation step (1 batch) is performed
        raise NotImplementedError()

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        # to be called after each full pass over the validation set
        raise NotImplementedError()


class ControlLogger(Logger):

    def __init__(self, config, model, dataset):
        super().__init__(config, model, dataset)

        self.control_names = dataset.output_columns
        self.total_loss_val = None
        self.individual_losses_val = None

    def training_step_end(self, global_step, loss, batch, predictions):
        # log total loss
        self.tb_writer.add_scalar("loss/train/total", loss.item(), global_step)

        # determine individual losses
        individual_losses = torch.nn.functional.mse_loss(predictions["output_control"],
                                                         batch["output_control"],
                                                         reduction="none")
        individual_losses = torch.mean(individual_losses, dim=0)

        # log individual losses
        for n, l in zip(self.control_names, individual_losses):
            self.tb_writer.add_scalar(f"loss/train/{n}", l, global_step)

    def training_epoch_end(self, global_step, epoch, model, optimiser):
        # save model checkpoint
        if (epoch + 1) % self.config["checkpoint_frequency"] == 0:
            torch.save({
                "global_step": global_step,
                "epoch": epoch,
                "model_name": self.config["model_name"],
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict()
            }, os.path.join(self.checkpoint_dir, "epoch{:03d}.pt".format(epoch)))
            print("Epoch {:03d}: Saving checkpoint to '{}'".format(epoch, self.checkpoint_dir))

    def validation_step_end(self, global_step, loss, batch, predictions):
        # determine individual losses
        individual_losses = torch.nn.functional.mse_loss(predictions["output_control"],
                                                         batch["output_control"],
                                                         reduction="none")
        individual_losses = torch.mean(individual_losses, dim=0)

        # accumulate total loss
        if self.total_loss_val is None:
            self.total_loss_val = torch.zeros_like(loss)
        self.total_loss_val += loss

        # accumulate individual losses
        if self.individual_losses_val is None:
            self.individual_losses_val = torch.zeros_like(individual_losses)
        self.individual_losses_val += individual_losses

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        # log total loss
        self.tb_writer.add_scalar("loss/val/total", self.total_loss_val.item(), global_step)

        # log individual losses
        for n, l in zip(self.control_names, self.individual_losses_val):
            self.tb_writer.add_scalar(f"loss/val/{n}", l, global_step)

        # reset the loss accumulators
        self.total_loss_val = torch.zeros_like(self.total_loss_val)
        self.individual_losses_val = torch.zeros_like(self.individual_losses_val)


class AttentionLogger(Logger):

    def __init__(self, config, model, dataset):
        super().__init__(config, model, dataset)

    def training_step_end(self, global_step, loss, batch, predictions):
        pass

    def training_epoch_end(self, global_step, epoch, model, optimiser):
        pass

    def validation_step_end(self, global_step, loss, batch, predictions):
        pass

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        pass
