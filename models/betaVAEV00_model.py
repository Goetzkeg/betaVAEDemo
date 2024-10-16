import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200

import pickle
import sys

sys.path.append("../")
from utils import get_size_after_pooling2D, get_size_after_con2D
from models.base_model import BaseModel
import pytorch_lightning as pl

class Conv2d_encoder(nn.Module):

    def __init__(self,
                 input_dim=(640, 2360),
                 train_bool=True,
                 activations='mish',
                 conv_sizes_encoder=[5, 5, 5],
                 conv_nr_encoder=[1, 8, 16, 32],
                 conv_strides_encoder=1,
                 pooling_kernel_size=[3, 3, 3],
                 pooling_stride=[3, 3, 3],
                 ful_cn_nodes_encoder=[-1, 500, 24],
                 dropout_encoder=-1,
                 batch_norm=False,
                 padding=0):

        super().__init__()
        self.train_bool = train_bool
        self.latent_dims = int(ful_cn_nodes_encoder[-1] / 2.)

        ## all this is to determine the dimension after all convolutional and pooling layers
        dim = input_dim
        self.pooling_dims = []
        self.conv_dims = []  # they are needed for transpose convs

        assert activations in ['mish', 'relu']

        if type(conv_strides_encoder) == int:
            conv_strides_encoder = [conv_strides_encoder for _ in conv_sizes_encoder]

        # if all([x == 1 for x in conv_strides_encoder]):
        #     for pks, ps in zip(pooling_kernel_size,pooling_stride):

        #         self.pooling_dims.append([dim[0], dim[1]])
        #         dim = get_size_after_pooling2D(
        #                 dim_in = dim,
        #                 padding =padding,
        #                 kernel_size = pks,
        #                 stride = ps,
        #                 )
        #     self.conv_dims = self.pooling_dims

        #   else:
        for i in range(len(conv_sizes_encoder)):
            self.conv_dims.append([dim[0], dim[1]])
            dim = get_size_after_con2D(
                dim_in=dim,
                padding=padding,
                kernel_size=conv_sizes_encoder[i],
                stride=conv_strides_encoder[i],
            )
            if len(pooling_kernel_size) > 0:
                self.pooling_dims.append([dim[0], dim[1]])
                dim = get_size_after_pooling2D(
                    dim_in=dim,
                    padding=0,
                    kernel_size=pooling_kernel_size[i],
                    stride=pooling_stride[i],
                )

        self.last_dim = dim
        ##now we have our dim.

        if ful_cn_nodes_encoder[0] == -1:
            ful_cn_nodes_encoder[0] = self.last_dim[0] * self.last_dim[1] * conv_nr_encoder[-1]

        conv_nr = conv_nr_encoder
        conv_sizes = conv_sizes_encoder
        ful_cn_nodes = ful_cn_nodes_encoder

        layer_con = [torch.nn.Conv2d(
            in_channels=conv_nr[i],
            out_channels=conv_nr[i + 1],
            kernel_size=conv_sizes[i],
            stride=conv_strides_encoder[i],
            padding=padding) for i in range(len(conv_sizes))]
        self.layer_con = nn.ModuleList(layer_con)

        if len(pooling_kernel_size) > 0:
            self.pooling = True
            layer_pool = [torch.nn.MaxPool2d(
                kernel_size=pks,
                stride=ps) for pks, ps in zip(pooling_kernel_size, pooling_stride)]
            self.layer_pool = nn.ModuleList(layer_pool)
        else:
            self.pooling = False
            self.layer_pool = [[] for _ in self.layer_con]

        if dropout_encoder > 0:
            self.use_dropout = True
            layer_dropout = [nn.Dropout(p=dropout_encoder) for _ in zip(conv_sizes)]
            self.layer_dropout = nn.ModuleList(layer_dropout)

        else:
            self.use_dropout = False
            self.layer_dropout = [[] for _ in self.layer_con]

        if batch_norm:
            layer_norm = [nn.BatchNorm2d(conv_nr[i + 1]) for i in range(len(conv_sizes))]
            self.layer_norm = nn.ModuleList(layer_norm)
            self.use_batchnorm = True
        else:
            self.use_batchnorm = False
            self.layer_norm = [[] for _ in self.layer_con]

        layer_fc = [nn.Linear(ful_cn_nodes[i], ful_cn_nodes[i + 1]) for i in range(len(ful_cn_nodes) - 1)]
        self.layer_fc = nn.ModuleList(layer_fc)

        if activations == 'mish':
            self.activations_con = [F.mish for _ in conv_sizes]
            self.activations_fc = [F.mish for _ in ful_cn_nodes[:-2]]
            activation_strings = ['mish' for _ in range(len(conv_sizes) + len(ful_cn_nodes) - 2)]

        if activations == 'relu':
            self.activations_con = [F.relu for _ in conv_sizes]
            self.activations_fc = [F.relu for _ in ful_cn_nodes[:-2]]
            activation_strings = ['relu' for _ in range(len(conv_sizes) + len(ful_cn_nodes) - 2)]

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        self.encoder_args = {
            'convolution layer number encoder': conv_nr,
            'filter sizes encoder': conv_sizes,
            'conv strides encoder': conv_strides_encoder,
            'fully_connected encoder': ful_cn_nodes,
            'pooling stride encoder': pooling_stride,
            'pooling size encoder': pooling_kernel_size,
            'encoder_activations encoder': activation_strings,
        }

    def forward(self, x):
        batch, input_dim0, input_dim1 = x.shape
        x = x.view(batch, 1, input_dim0, input_dim1)

        for activations, layer, pool, dropout, batchnorm in zip(self.activations_con, self.layer_con, self.layer_pool,
                                                                self.layer_dropout, self.layer_norm):
            x = activations(layer(x))

            if self.use_dropout:
                x = dropout(x)

            if self.pooling:
                x = pool(x)

            if self.use_batchnorm:
                x = batchnorm(x)

        x = x.view(batch, -1)

        for activations, layer in zip(self.activations_fc, self.layer_fc):
            x = activations(layer(x))

        x = self.layer_fc[-1](x)

        x = x.view(batch, 2, -1)

        if self.train_bool:
            mu = x[:, 0, :]
            logvar = x[:, 1, :]
            sigma = torch.exp(0.5 * logvar)
            try:
                z = mu + sigma * self.N.sample(mu.shape)
            except RuntimeError:
                self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
                self.N.scale = self.N.scale.cuda()
                z = mu + sigma * self.N.sample(mu.shape)

            self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            z = torch.empty(x.shape[0], 2, self.latent_dims)
            z[:, 0, :] = x[:, 0, :]
            z[:, 1, :] = torch.exp(0.5 * x[:, 1, :])

        return z


class Conv2d_decoder(nn.Module):

    def __init__(self,
                 last_dim,  # 2d
                 activations='mish',
                 conv_sizes_decoder=[5, 5, 5],
                 conv_nr_decoder=[1, 8, 16, 32],
                 conv_strides_decoder=1,
                 pooling_dims=[3, 3, 3],
                 conv_dims=[3, 3, 3],
                 pooling_mode='nearest',
                 ful_cn_nodes_decoder=[-1, 500, 24],
                 dropout_decoder=-1,
                 conv_type='Transpose',
                 padding=0):

        super().__init__()
        self.last_dim = last_dim
        self.conv_dims = conv_dims

        ful_cn_nodes = ful_cn_nodes_decoder
        layer_fc = [nn.Linear(ful_cn_nodes[i], ful_cn_nodes[i + 1]) for i in range(len(ful_cn_nodes) - 1)]
        self.layer_fc = nn.ModuleList(layer_fc)

        conv_nr = conv_nr_decoder
        self.conv_nr = conv_nr
        conv_sizes = conv_sizes_decoder

        if type(conv_strides_decoder) == int:
            conv_strides_decoder = [conv_strides_decoder for _ in conv_sizes]

        if conv_type == 'Transpose':
            convFunction = torch.nn.ConvTranspose2d
            self.transpose = True
        else:
            convFunction = torch.nn.Conv2d
            self.transpose = False

        layer_con = [convFunction(
            in_channels=conv_nr[i],
            out_channels=conv_nr[i + 1],
            kernel_size=conv_sizes[i],
            stride=conv_strides_decoder[i],
            padding=padding) for i in range(len(conv_sizes))]

        self.layer_con = nn.ModuleList(layer_con)

        self.upsample = False
        if len(pooling_dims) > 0:
            self.upsample = True
            layer_up = [torch.nn.Upsample(size=tuple(pd), mode=pooling_mode) for pd in pooling_dims]
            self.layer_up = nn.ModuleList(layer_up)

        else:
            self.layer_up = [[] for _ in self.layer_con]

        if dropout_decoder > 0:
            self.use_dropout = True
            layer_dropout = [nn.Dropout(p=dropout_decoder) for _ in zip(conv_sizes)]
            self.layer_dropout = nn.ModuleList(layer_dropout)

        else:
            self.use_dropout = False
            self.layer_dropout = [[] for _ in self.layer_con]

        if activations == 'mish':
            self.activations_con = [F.mish for _ in conv_sizes]
            self.activations_fc = [F.mish for _ in ful_cn_nodes[:-1]]
            activation_strings = ['mish' for _ in range(len(conv_sizes) + len(ful_cn_nodes) - 1)]

        if activations == 'relu':
            self.activations_con = [F.relu for _ in conv_sizes]
            self.activations_fc = [F.relu for _ in ful_cn_nodes[:-1]]
            activation_strings = ['relu' for _ in range(len(conv_sizes) + len(ful_cn_nodes) - 1)]

        self.decoder_args = {
            'convolution layer decoder': conv_nr,
            'filter sizes decoder': conv_sizes,
            'fully_connected decoder': ful_cn_nodes,
            'encoder_activations decoder': activation_strings,
        }

    def forward(self, x):
        batch, _ = x.shape

        for activations, layer in zip(self.activations_fc, self.layer_fc):
            x = activations(layer(x))

        x = x.view(batch, self.conv_nr[0], self.last_dim[0], self.last_dim[1])  # -1 is imdim

        for activations, layer, upsam, cs, dropout in zip(self.activations_con, self.layer_con, self.layer_up,
                                                          self.conv_dims, self.layer_dropout):
            if self.upsample:
                x = upsam(x)

            if self.use_dropout:
                x = dropout(x)

            if self.transpose:
                x = activations(layer(x, output_size=cs))
            else:
                x = activations(layer(x))

                # x = x.view(batch, -1)
        # assert x.shape[0] == 3, 'output dim has to be batch, imdim0,imdim1'
        torch.squeeze(x, 1)
        return x


class BetaVAEv00Model(pl.LightningModule):  # Conv2d_autoencoder

    def __init__(self,
                 input_dim=(640, 2360),
                 train_bool=True,
                 beta=0.1,
                 activations='mish',
                 conv_sizes_encoder=[5, 5, 5],
                 conv_nr_encoder=[1, 8, 16, 32],
                 conv_strides_encoder=1,
                 pooling_kernel_size=[],
                 pooling_stride=[],
                 interpolation_mode='nearest',
                 ful_cn_nodes_encoder=[-1, 500, 24],
                 dropout_encoder=-1,
                 dropout_decoder=-1,
                 batch_norm=False,
                 conv_type_decoder='Transpose',
                 optimizer_kwargs: dict = {}):

        super().__init__()
        self.kwargs = locals()
        self.kwargs.pop('__class__', None)
        self.kwargs.pop('self', None)
        self.save_hyperparameters()
        self.latent_dims = int(ful_cn_nodes_encoder[-1] / 2)
        self.beta = beta
        self.optimizer_kwargs = optimizer_kwargs

        # assert len(pooling_kernel_size)<=0: 'pooling is not implemented yet, as upsampling is not trivial'
        # gib die dem encoder
        if conv_type_decoder == 'Transpose':
            padding = 0
        else:
            padding = 'same'

        self.encoder = Conv2d_encoder(
            input_dim=input_dim,
            train_bool=train_bool,
            activations=activations,
            conv_sizes_encoder=conv_sizes_encoder,
            conv_nr_encoder=conv_nr_encoder,
            conv_strides_encoder=conv_strides_encoder,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            ful_cn_nodes_encoder=ful_cn_nodes_encoder,
            dropout_encoder=dropout_encoder,
            batch_norm=batch_norm,
            padding=padding)

        # bilde die Argumente für den Decoder (inklusive last dim)
        last_dim_encoder = self.encoder.last_dim
        if type(conv_sizes_encoder) == list:
            conv_sizes_decoder = conv_sizes_encoder[::-1]
        else:
            conv_sizes_decoder = conv_sizes_encoder
        if type(conv_nr_encoder) == list:
            conv_nr_decoder = conv_nr_encoder[::-1]
        else:
            conv_nr_decoder = conv_nr_encoder
        if type(conv_strides_encoder) == list:
            conv_strides_decoder = conv_strides_encoder[::-1]
        else:
            conv_strides_decoder = conv_strides_encoder

        ful_cn_nodes_decoder = ful_cn_nodes_encoder[::-1]
        ful_cn_nodes_decoder[0] = int(ful_cn_nodes_decoder[0] / 2)
        ful_cn_nodes_decoder[-1] = int(self.encoder.last_dim[0] * self.encoder.last_dim[1] * conv_nr_encoder[-1])

        pooling_dims = self.encoder.pooling_dims[::-1]  # may cause problems due to rounding problems
        conv_dims = self.encoder.conv_dims[::-1]

        self.decoder = Conv2d_decoder(
            last_dim=last_dim_encoder,  # 2d
            activations=activations,
            conv_sizes_decoder=conv_sizes_decoder,
            conv_nr_decoder=conv_nr_decoder,
            conv_strides_decoder=conv_strides_decoder,
            pooling_dims=pooling_dims,
            conv_dims=conv_dims,
            pooling_mode=interpolation_mode,
            ful_cn_nodes_decoder=ful_cn_nodes_decoder,
            dropout_decoder=dropout_decoder,
            conv_type=conv_type_decoder,
            padding=padding)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        x_hat = torch.squeeze(x_hat)

        loss_recon = (((x - x_hat) ** 2).sum())
        loss_disent = self.beta * self.encoder.kl
        loss = loss_recon + loss_disent

        # logging metrics we calculated by hand
        self.log('train/loss', loss)
        self.log('train/recon_loss', loss_recon)
        self.log('train/disent_loss', loss_disent)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.forward(x)
        x_hat = torch.squeeze(x_hat)

        loss_recon = (((x - x_hat) ** 2).sum())
        loss_disent = self.beta * self.encoder.kl
        loss = loss_recon + loss_disent

        # logging metrics we calculated by hand
        self.log('val/loss', loss)
        self.log('val/recon_loss', loss_recon)
        self.log('val/disent_loss', loss_disent)

        return loss

    def save(self, path='testpath'):
        torch.save(self.encoder.state_dict(), path + '_encoder')
        torch.save(self.decoder.state_dict(), path + '_decoder')

    def load(self, device = 'cuda', path='testpath'):
        self.encoder.load_state_dict(torch.load(path + '_encoder', map_location=torch.device(device)))
        self.decoder.load_state_dict(torch.load(path + '_decoder', map_location=torch.device(device)))

    def save_model_param(self, path='testpath'):
        save_dict = {
            'modeltype': 'conv2d',
            'kwargs': self.kwargs
        }

        with open(path + 'paramdict.pkl', 'wb') as f:
            pickle.dump(save_dict, f)
