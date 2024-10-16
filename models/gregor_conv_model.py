import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl


class GregorConvModel(pl.LightningModule):
    def __init__(self, channels: int, conv_sizes: int, padding: int,  lr:float, optimizer_kwargs = {}):
        super(GregorConvModel, self).__init__()
        self.learning_rate = lr
        self.optimizer_kwargs = optimizer_kwargs
        block = [
            nn.Conv2d(1, channels[0], kernel_size=(conv_sizes[0], conv_sizes[0]), stride=(1, 1), padding=(padding[0], padding[0])),
            nn.ReLU(inplace=True)]

        for i in range(1, len(channels) - 1):
            block.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=(conv_sizes[i], conv_sizes[i]), stride=(1, 1),
                                   padding=(padding[i], padding[i])))
            block.append(torch.nn.ReLU(inplace=True))

        block.append(nn.Conv2d(channels[-2], 1, kernel_size=(conv_sizes[-1], conv_sizes[-1]), stride=(1, 1),
                               padding=(padding[-1], padding[-1])))

        self.model = torch.nn.Sequential(*block)

    def forward(self, masked_img_tensor, mask_tensor= None): # I do not need the mask tensor here, but in this way I can reuse the plot callbacks easier
        if masked_img_tensor.dim() == 3: #add color channel if not present
            masked_img_tensor = torch.unsqueeze(masked_img_tensor,1)
        x = masked_img_tensor

        return self.model(x)

    def training_step(self, batch, batch_nb):
        images,masks = batch[0], batch[1]
        x = images*masks
        x_hat = self.forward(x)
        x_hat = torch.squeeze(x_hat)

        loss_recon = (((images - x_hat) ** 2).sum())
        loss = loss_recon

        # logging metrics we calculated by hand
        self.log('train/loss', loss_recon)

        return loss

    def validation_step(self, batch, batch_nb):
        images,masks = batch[0], batch[1]
        x = images*masks
        x_hat = self.forward(x)
        x_hat = torch.squeeze(x_hat)

        loss_recon = (((x - x_hat) ** 2).sum())
        loss = loss_recon

        # logging metrics we calculated by hand
        dict_valid = {
            'val/loss': loss,
        }

        return dict_valid

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
        return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)
