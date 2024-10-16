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
#from here: https://github.com/hiepph/unet-lightning/blob/master/Unet.py

class UnetMaskedModel(pl.LightningModule):
    def __init__(self, lr, optimizer_kwargs = {}, n_channels:int=1):
        super(UnetMaskedModel, self).__init__()
        self.optimizer_kwargs = optimizer_kwargs
        self.n_channels = n_channels
        self.bilinear = True
        self.learning_rate = lr

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranpose2d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1) ## why 1?
                return self.conv(x)

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, masked_img_tensor, mask_tensor= None): # I do not need the mask tensor here, but in this way I can reuse the plot callbacks easier
        if masked_img_tensor.dim() == 3: #add color channel if not present
            masked_img_tensor = torch.unsqueeze(masked_img_tensor,1)
        x = masked_img_tensor
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)

    def training_step(self, batch, batch_nb):
        images,masks = batch[0], batch[1]
        x = images*masks
        x_hat = self.forward(x)
        x_hat = torch.squeeze(x_hat)

        loss_recon = (((images - x_hat) ** 2).mean())
        loss = loss_recon

        # logging metrics we calculated by hand
        self.log('train/step_loss', loss_recon)

        return loss

    def validation_step(self, batch, batch_nb):
        images,masks = batch[0], batch[1]
        x = images*masks
        x_hat = self.forward(x)
        x_hat = torch.squeeze(x_hat)

        loss_recon = (((images - x_hat) ** 2).mean())
        loss = loss_recon
        self.log('val/step_loss', loss_recon)

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
