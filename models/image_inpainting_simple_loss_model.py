import numpy as np
import os
import torch
import pytorch_lightning as pl

from models.pconv_unet import PConvUNet
from models.base_model import BaseModel

from loss.l2_loss import L2Loss as LossCompute

from argparse import ArgumentParser

# from utils import unnormalize

MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]


def unnormalize(x):
    x = x.transpose(0, 2, 3, 1)
    x = x * STDDEV + MEAN
    return x


class ImageInpaintingSimpleLossModel(pl.LightningModule):

    def __init__(self, loss_factor_l2_valid: float, loss_factor_l2_mask: float,
                 learning_rate: float):
        super(ImageInpaintingSimpleLossModel, self).__init__()

        self.loss_factors = {
            "l2_valid": loss_factor_l2_valid,
            "l2_mask": loss_factor_l2_mask,
        }

        self.learning_rate = learning_rate
        self.pConvUNet = PConvUNet(channels=1)

        self.lossCompute = LossCompute(self.loss_factors)  # , device="cuda"

    def forward(self, masked_img_tensor, mask_tensor):
        return self.pConvUNet(masked_img_tensor, mask_tensor)

    def training_step(self, batch, batch_nb):
        images = batch[0].unsqueeze(1)
        masks = batch[1].unsqueeze(1)
        masked_images = masks * images

        output = self.forward(masked_images, masks)


        ls_fn = self.lossCompute.loss_total(masks)
        loss, dict_losses = ls_fn(images, output)

        dict_losses_train = {}
        for key, value in dict_losses.items():
            dict_losses_train[key] = value.item()

        return {'loss': loss, 'progress_bar': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):

        images = batch[0].unsqueeze(1)  # .expand(-1,3,-1,-1)  # Retrieve the images from the batch tensor
        masks = batch[1].unsqueeze(1)  # .expand(-1,3,-1,-1)  # Retrieve the masks from the batch tensor
        masked_images = masks * images

        output = self.forward(masked_images, masks)
        images = images.expand(-1, 3, -1, -1)
        output = output.expand(-1, 3, -1, -1)
        masks = masks.expand(-1, 3, -1, -1)

        ls_fn = self.lossCompute.loss_total(masks)

        loss, dict_losses = ls_fn(images, output)

        dict_valid = {'val_loss': loss.mean(), **dict_losses}

        return dict_valid

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_loss_hole = torch.stack([x['loss_mask'] for x in outputs]).mean()
        avg_loss_valid = torch.stack([x['loss_valid'] for x in outputs]).mean()

        valid_dict = {
            "loss_hole": avg_loss_hole,
            "loss_valid": avg_loss_valid,
        }

        tqdm_dict = {'val_loss': avg_loss}

        return {'val_loss': avg_loss, 'progress_bar': tqdm_dict}

    def validation_epoch_end(self, outputs):
        # 'outputs' is a list of dictionaries from each validation step

        # Initialize dictionaries to store the sum and count for each metric
        sum_metrics = {}
        count_metrics = {}

        # Aggregate the metrics across all validation steps
        for output in outputs:
            for key, value in output.items():
                if key not in sum_metrics:
                    sum_metrics[key] = value
                    count_metrics[key] = 1
                else:
                    sum_metrics[key] += value
                    count_metrics[key] += 1

        # Calculate the mean for each metric
        mean_metrics = {key: sum_metrics[key] / count_metrics[key] for key in sum_metrics}

        # Log the mean metrics using self.log
        for key, value in mean_metrics.items():
            self.log(key, value)

        # Return a dictionary with any desired information or metrics
        return mean_metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
