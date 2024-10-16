import torch
import torch.nn as nn
import pytorch_lightning as pl

from loss.maskMSE_loss import MaskMSE as LossCompute

class ConvReduceModel(pl.LightningModule):
    def __init__(self, lr, optimizer_kwargs = {},loss_kwargs={} ,n_channels:int=1):
        super(ConvReduceModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 15 * 60, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 50 * 240),
            nn.Sigmoid()  # Use sigmoid activation for the final output
        )

        self.lossCompute = LossCompute(**loss_kwargs)
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs

    def forward(self, masked_img_tensor, mask_tensor= None):
        if masked_img_tensor.dim() == 3: #add color channel if not present
            masked_img_tensor = torch.unsqueeze(masked_img_tensor,1)
        x = masked_img_tensor
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        x = x.view(x.size(0), 1, 240, 50)  # Reshape the tensor to match the desired output size
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
        return optimizer

    def training_step(self, batch, batch_idx):

        images,masks = batch[0], batch[1]
        x = images*masks
        y_hat = self.forward(x)

        ls_fn = self.lossCompute.loss_total(masks)
        loss, dict_losses = ls_fn(images, y_hat)

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        images,masks = batch[0], batch[1]
        x = images*masks
        y_hat = self.forward(x)
        ls_fn = self.lossCompute.loss_total(masks)
        loss, dict_valid = ls_fn(images, y_hat)

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
        self.log('val/loss', mean_metrics['loss_MSE'])
        # Return a dictionary with any desired information or metrics
        return mean_metrics
