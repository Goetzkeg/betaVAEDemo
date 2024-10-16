from abc import ABC, abstractmethod
import pytorch_lightning as pl

"""This module implements an abstract base class (ABC) 'BaseModel' for all models.
"""


class BaseModel(pl.LightningModule, ABC):

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def training_step(self, train_batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, val_batch, batch_idx):
        pass
