import numpy as np
import os
import torch
import pytorch_lightning as pl

from models.pconv_unet import PConvUNet
from models.vgg16_extractor import VGG16Extractor
from models.base_model import BaseModel

from loss.loss_compute import LossCompute

from argparse import ArgumentParser
#from utils import unnormalize

MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]

def unnormalize(x):
	x = x.transpose(0, 2,3, 1)
	x = x * STDDEV + MEAN
	return x

class ImageInpaintingModel(pl.LightningModule):

    def __init__(self, loss_factor_hole: float, loss_factor_valid: float,
                 loss_factor_perceptual: float, loss_factor_out: float,
                 loss_factor_comp: float, loss_factor_tv: float,
                 learning_rate: float):
        super(ImageInpaintingModel, self).__init__()

        self.loss_factors = {
            "loss_hole": loss_factor_hole,
            "loss_valid": loss_factor_valid,
            "loss_perceptual": loss_factor_perceptual,
            "loss_style_out": loss_factor_out,
            "loss_style_comp": loss_factor_comp,
            "loss_tv": loss_factor_tv,
        }

        self.learning_rate = learning_rate
        self.pConvUNet = PConvUNet(channels=1)

        self.vgg16extractor = VGG16Extractor()#.to("cuda")
        for param in self.vgg16extractor.parameters():
            param.requires_grad = False
        self.lossCompute = LossCompute(self.vgg16extractor, self.loss_factors) #, device="cuda"

    def forward(self, masked_img_tensor, mask_tensor):
        return self.pConvUNet(masked_img_tensor, mask_tensor)

    def training_step(self, batch, batch_nb):
        images = batch[0].unsqueeze(1)#.expand(-1,3,-1,-1)  # Retrieve the images from the batch tensor
        masks = batch[1].unsqueeze(1)  # Retrieve the masks from the batch tensor
        masked_images = masks * images


        output = self.forward(masked_images, masks)

        images = images.expand(-1,3,-1,-1)
        output = output.expand(-1,3,-1,-1)
        masks = masks.expand(-1,3,-1,-1)


        ls_fn = self.lossCompute.loss_total(masks)
        loss, dict_losses = ls_fn(images, output)

        dict_losses_train = {}
        for key, value in dict_losses.items():
            dict_losses_train[key] = value.item()

        return {'loss': loss, 'progress_bar': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):

        images = batch[0].unsqueeze(1)#.expand(-1,3,-1,-1)  # Retrieve the images from the batch tensor
        masks = batch[1].unsqueeze(1)#.expand(-1,3,-1,-1)  # Retrieve the masks from the batch tensor
        masked_images = masks * images

        output = self.forward(masked_images, masks)
        images = images.expand(-1,3,-1,-1)
        output = output.expand(-1,3,-1,-1)
        masks = masks.expand(-1,3,-1,-1)

        ls_fn = self.lossCompute.loss_total(masks)

        loss, dict_losses = ls_fn(images, output)

        psnr = self.lossCompute.PSNR(images, output)
        if batch_nb == 0:
            res = np.clip(unnormalize(output.detach().cpu().numpy()), 0, 1)
            original_img = np.clip(unnormalize(masked_images.detach().cpu().numpy()), 0, 1)
            target_img = np.clip(unnormalize(images.detach().cpu().numpy()), 0, 1)
            combined_imgs = []
            for i in range(images.shape[0]):
                combined_img = np.concatenate((original_img[i], res[i], target_img[i]), axis=1)
                combined_imgs.append(combined_img)
            combined_imgs = np.concatenate(combined_imgs)
        dict_valid = {'val_loss': loss.mean(), 'psnr': psnr.mean(), **dict_losses}

        return dict_valid

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in outputs]).mean()

        avg_loss_hole = torch.stack([x['loss_hole'] for x in outputs]).mean()
        avg_loss_valid = torch.stack([x['loss_valid'] for x in outputs]).mean()
        avg_loss_perceptual = torch.stack([x['loss_perceptual'] for x in outputs]).mean()
        avg_loss_style_out = torch.stack([x['loss_style_out'] for x in outputs]).mean()
        avg_loss_style_comp = torch.stack([x['loss_style_comp'] for x in outputs]).mean()
        avg_loss_tv = torch.stack([x['loss_tv'] for x in outputs]).mean()
        valid_dict = {
            "loss_hole": avg_loss_hole,
            "loss_valid": avg_loss_valid,
            "loss_perceptual": avg_loss_perceptual,
            "loss_style_out": avg_loss_style_out,
            "loss_style_comp": avg_loss_style_comp,
            "loss_tv": avg_loss_tv
        }

        tqdm_dict = {'valid_psnr': avg_psnr, 'val_loss': avg_loss}

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
