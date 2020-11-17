import os
import json
import torch

from torch.utils.tensorboard import SummaryWriter
from src.models.utils import image_softmax, convert_attention_to_image


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

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # to be called when a single training step (1 batch) is performed
        raise NotImplementedError()

    def training_epoch_end(self, global_step, epoch, model, optimiser):
        # to be called after each full pass over the training set
        raise NotImplementedError()

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # to be called when a single validation step (1 batch) is performed
        raise NotImplementedError()

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        # to be called after each full pass over the validation set
        raise NotImplementedError()


class TestLogger(Logger):

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        pass

    def training_epoch_end(self, global_step, epoch, model, optimiser):
        pass

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        pass

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        pass


class ControlLogger(Logger):

    def __init__(self, config, model, dataset):
        super().__init__(config, model, dataset)

        self.control_names = dataset.output_columns
        self.total_loss_val_mse = None
        self.total_loss_val_l1 = None
        self.individual_losses_val_mse = None
        self.individual_losses_val_l1 = None
        self.counter_val = 0

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # log total loss
        self.tb_writer.add_scalar("loss/train/total/mse", total_loss.item(), global_step)

        # determine individual losses
        individual_losses_mse = torch.nn.functional.mse_loss(predictions["output_control"],
                                                             batch["output_control"],
                                                             reduction="none")
        individual_losses_mse = torch.mean(individual_losses_mse, dim=0)

        # log individual losses
        for n, l_mse in zip(self.control_names, individual_losses_mse):
            self.tb_writer.add_scalar(f"loss/train/{n}/mse", l_mse, global_step)

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

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # determine individual losses
        individual_losses_mse = torch.nn.functional.mse_loss(predictions["output_control"],
                                                             batch["output_control"],
                                                             reduction="none")
        individual_losses_mse = torch.mean(individual_losses_mse, dim=0)
        individual_losses_l1 = torch.nn.functional.l1_loss(predictions["output_control"],
                                                           batch["output_control"],
                                                           reduction="none")
        individual_losses_l1 = torch.mean(individual_losses_l1, dim=0)

        # accumulate total loss
        if self.total_loss_val_mse is None:
            self.total_loss_val_mse = torch.zeros_like(total_loss)
            self.total_loss_val_l1 = torch.zeros_like(total_loss)
        self.total_loss_val_mse += total_loss
        self.total_loss_val_l1 += torch.mean(individual_losses_l1)

        # accumulate individual losses
        if self.individual_losses_val_mse is None:
            self.individual_losses_val_mse = torch.zeros_like(individual_losses_mse)
            self.individual_losses_val_l1 = torch.zeros_like(individual_losses_l1)
        self.individual_losses_val_mse += individual_losses_mse
        self.individual_losses_val_l1 += individual_losses_l1

        self.counter_val += 1

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        # log total loss
        self.tb_writer.add_scalar("loss/val/total/mse", self.total_loss_val_mse.item() / self.counter_val, global_step)
        self.tb_writer.add_scalar("loss/val/total/l1", self.total_loss_val_l1.item() / self.counter_val, global_step)

        # log individual losses
        for n, l_mse, l_l1 in zip(self.control_names, self.individual_losses_val_mse, self.individual_losses_val_l1):
            self.tb_writer.add_scalar(f"loss/val/{n}/mse", l_mse / self.counter_val, global_step)
            self.tb_writer.add_scalar(f"loss/val/{n}/l1", l_l1 / self.counter_val, global_step)

        # reset the loss accumulators
        self.total_loss_val_mse = torch.zeros_like(self.total_loss_val_mse)
        self.total_loss_val_l1 = torch.zeros_like(self.total_loss_val_l1)
        self.individual_losses_val_mse = torch.zeros_like(self.individual_losses_val_mse)
        self.individual_losses_val_l1 = torch.zeros_like(self.individual_losses_val_l1)
        self.counter_val = 0


class AttentionLogger(Logger):

    def __init__(self, config, model, dataset):
        super().__init__(config, model, dataset)

        self.total_loss_val_kl = None
        self.partial_losses_val_kl = {}
        self.counter_val = 0

        self.log_attention_val = True

    def training_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # log total loss
        # TODO: maybe the loss "type" shouldn't be hard-coded
        self.tb_writer.add_scalar("loss/train/total/kl", total_loss.item(), global_step)

        # if len(partial_losses) > 1:
        # log the partial losses
        for ln, l in partial_losses.items():
            self.tb_writer.add_scalar(f"loss/train/{ln}/kl", l.item(), global_step)

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

    def validation_step_end(self, global_step, total_loss, partial_losses, batch, predictions):
        # accumulate total loss
        if self.total_loss_val_kl is None:
            self.total_loss_val_kl = torch.zeros_like(total_loss)
        self.total_loss_val_kl += total_loss

        # accumulate partial losses
        for ln, l in partial_losses.items():
            if ln not in self.partial_losses_val_kl:
                self.partial_losses_val_kl[ln] = torch.zeros_like(l)
            self.partial_losses_val_kl[ln] += l

        self.counter_val += 1

        if self.log_attention_val:
            # get the original from the batch and the predictions and plot them
            # probably only include the original of the uncropped attention map...
            images_original = convert_attention_to_image(batch["original"]["output_attention"])
            images_prediction = convert_attention_to_image(image_softmax(predictions["output_attention"]),
                                                           out_shape=images_original.shape[2:])

            self.tb_writer.add_images("attention/val/ground_truth", images_original, global_step, dataformats="NCHW")
            self.tb_writer.add_images("attention/val/prediction", images_prediction, global_step, dataformats="NCHW")

            self.log_attention_val = False

    def validation_epoch_end(self, global_step, epoch, model, optimiser):
        # TODO: also record a batch or so of original and predicted attention maps and save them

        # log total loss
        self.tb_writer.add_scalar("loss/val/total/kl", self.total_loss_val_kl.item() / self.counter_val, global_step)

        # log individual losses
        for ln, l in self.partial_losses_val_kl.items():
            self.tb_writer.add_scalar(f"loss/val/{ln}/kl", l.item() / self.counter_val, global_step)

        # reset the loss accumulators
        self.total_loss_val_kl = torch.zeros_like(self.total_loss_val_kl)
        for ln, l in self.partial_losses_val_kl.items():
            self.partial_losses_val_kl[ln] = torch.zeros_like(l)
        self.counter_val = 0

        self.log_attention_val = True
