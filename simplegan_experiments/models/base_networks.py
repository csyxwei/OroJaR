import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as SN
import numpy as np
from collections import OrderedDict
def add_normalization_2d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(('batchnorm', nn.BatchNorm2d(n_out, affine=True, eps=1e-03, momentum=0.001)))
    elif fn == 'instancenorm':
        layers.append(('instancenorm', nn.InstanceNorm2d(n_out, affine=True, eps=1e-06)))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(('batchnorm', nn.BatchNorm1d(n_out)))
    elif fn == 'instancenorm':
        layers.append(('instancenorm', nn.InstanceNorm1d(n_out, affine=True)))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(('active', nn.ReLU(inplace=False)))
    elif fn == 'lrelu':
        layers.append(('active', nn.LeakyReLU(negative_slope=0.2, inplace=False)))
    elif fn == 'sigmoid':
        layers.append(('active', nn.Sigmoid()))
    elif fn == 'tanh':
        layers.append(('active', nn.Tanh()))
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu', use_bias=False, use_spn=False):
        super(LinearBlock, self).__init__()
        # initialize fully connected layer
        if use_spn:
            layers = [('conv', SN(nn.Linear(input_dim, output_dim, bias=use_bias)))]
        else:
            layers = [('conv', nn.Linear(input_dim, output_dim, bias=use_bias))]
        layers = add_normalization_1d(layers, norm, output_dim)
        layers = add_activation(layers, activation)
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x, reg_layer='none'):
        out, fea = x, None
        for name, m in self.layers.named_children():
            out = m(out)
            if reg_layer in name:
                fea = out

        if fea is not None:
            return fea, out
        else:
            return out


class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm='instancenorm', activation='lrelu', use_spn=False, bias=None):
        super(Conv2dBlock, self).__init__()
        if bias is None:
            bias = (norm == 'none')

        if use_spn:
            layers = [('conv', SN(nn.Conv2d(n_in, n_out, kernel_size, stride=stride,
                                              padding=padding, bias=bias)))]
        else:
            layers = [('conv', nn.Conv2d(n_in, n_out, kernel_size, stride=stride,
                                padding=padding, bias=bias))]
        layers = add_normalization_2d(layers, norm, n_out)
        layers = add_activation(layers, activation)
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x, reg_layer='none'):
        out, fea = x, None
        for name, m in self.layers.named_children():
            out = m(out)
            if reg_layer in name:
                fea = out

        if fea is not None:
            return fea, out
        else:
            return out


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm='instancenorm', activation='relu', use_spn=False, bias=None):
        if bias is None:
            bias = (norm == 'none')

        super(ConvTranspose2dBlock, self).__init__()
        if use_spn:
            layers = [
                ('conv', SN(nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=bias)))]
        else:
            layers = [
                ('conv', nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=bias))]
        layers = add_normalization_2d(layers, norm, n_out)
        layers = add_activation(layers, activation)
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x, reg_layer='none'):
        out, fea = x, None
        for name, m in self.layers.named_children():
            out = m(out)
            if reg_layer in name:
                fea = out

        if fea is not None:
            return fea, out
        else:
            return out

"""
Generator and Discrimator for 64x64 resolution (Dsprites etc).
"""

class Generator(nn.Module):
    def __init__(self,  nz=12, nc_out=3, first_norm=False, norm='instancenorm', activation='lrelu'):
        super(Generator, self).__init__()

        layers = []

        if first_norm:
            layers += [ConvTranspose2dBlock(nz, 512, 4, 1, 0, norm=norm, activation=activation)]
        else:
            layers += [ConvTranspose2dBlock(nz, 512, 4, 1, 0, norm='none', activation='none')]

        layers += [
            ConvTranspose2dBlock(512, 256, (4, 4), stride=2, padding=1, norm=norm, activation=activation)]  # 8 * 8
        layers += [
            ConvTranspose2dBlock(256, 128, (4, 4), stride=2, padding=1, norm=norm, activation=activation)]  # 16 * 16
        layers += [
            ConvTranspose2dBlock(128, 64, (4, 4), stride=2, padding=1, norm=norm, activation=activation)]  # 32 * 32
        layers += [
            ConvTranspose2dBlock(64, 64, (4, 4), stride=2, padding=1, norm=norm, activation=activation)]  # 64 * 64
        layers += [
            Conv2dBlock(64, nc_out, (3, 3), stride=1, padding=1, norm='none', activation='tanh')]  # output layer

        self.layers = nn.ModuleList(layers)

    def forward(self, z, reg_layer='none', select_layers=[]):
        out_list = []

        out = z.view(z.size(0), -1, 1, 1)

        for i, layer in enumerate(self.layers):
            if reg_layer != 'none' and i > np.max(select_layers):
                continue

            if reg_layer != 'none' and i in select_layers:
                outs = layer(out, reg_layer)
                out_list.append(outs[0])
                out = outs[1]
            else:
                out = layer(out)

        return out_list if reg_layer != 'none' else out

class Discriminator(nn.Module):
    def __init__(self, nc_in=3, out_dim=1, norm='instancenorm', activation='relu'):
        super(Discriminator, self).__init__()

        self.encoder = nn.Sequential(
            Conv2dBlock(nc_in, 64, 4, 2, 1, norm='none', activation=activation),  # B,  64, 32, 32
            Conv2dBlock(64, 128, 4, 2, 1, norm=norm, activation=activation),  # B,  128, 16, 16
            Conv2dBlock(128, 256, 4, 2, 1, norm=norm, activation=activation),  # B,  256, 8, 8
            Conv2dBlock(256, 512, 4, 2, 1, norm=norm, activation=activation),  # B,  512, 4, 4
        )

        self.fc = nn.Sequential(
            LinearBlock(512 * 4 * 4, 512, activation=activation, use_bias=True),
            LinearBlock(512, out_dim, activation='none', use_bias=True),
        )


    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

"""
Generator and Discrimator for 128x128 resolution (CelebA, CLEVR etc).
"""

class Generator128(nn.Module):
    def __init__(self, nz=30, nc_out=3, first_norm=False, norm='instancenorm', activation='lrelu'):
        super(Generator128, self).__init__()

        layers = []

        if first_norm:
            layers += [ConvTranspose2dBlock(nz, 1024, 4, 1, 0, norm=norm, activation=activation)]
        else:
            layers += [ConvTranspose2dBlock(nz, 1024, 4, 1, 0, norm='none', activation='none')]

        layers += [
            ConvTranspose2dBlock(1024, 1024, (4, 4), stride=2, padding=1, norm=norm, activation=activation)]  # 8 * 8
        layers += [
            ConvTranspose2dBlock(1024, 512, (4, 4), stride=2, padding=1, norm=norm, activation=activation)]  # 16 * 16
        layers += [
            ConvTranspose2dBlock(512, 256, (4, 4), stride=2, padding=1, norm=norm, activation=activation)]  # 32 * 32
        layers += [
            ConvTranspose2dBlock(256, 128, (4, 4), stride=2, padding=1, norm=norm, activation=activation)]  # 64 * 64
        layers += [
            ConvTranspose2dBlock(128, 64, (4, 4), stride=2, padding=1, norm=norm, activation=activation)]  # 128 * 128
        layers += [
            Conv2dBlock(64, nc_out, (3, 3), stride=1, padding=1, norm='none', activation='tanh')]  # 128 * 128

        self.layers = nn.ModuleList(layers)

    def forward(self, z, reg_layer='none', select_layers=[]):
        out_list = []

        out = z.view(z.size(0), -1, 1, 1)

        for i, layer in enumerate(self.layers):
            if reg_layer != 'none' and i > np.max(select_layers):
                continue

            if reg_layer != 'none' and i in select_layers:
                outs = layer(out, reg_layer)
                out_list.append(outs[0])
                out = outs[1]
            else:
                out = layer(out)

        return out_list if reg_layer != 'none' else out

class Discriminator128(nn.Module):
    def __init__(self, nc_in=3, out_dim=1, norm='instancenorm', activation='relu'):
        super(Discriminator128, self).__init__()

        self.encoder = nn.Sequential(
            Conv2dBlock(nc_in, 64, 4, 2, 1, norm='none', activation=activation),  # B,  64, 64, 64
            Conv2dBlock(64, 128, 4, 2, 1, norm=norm, activation=activation),  # B,  128, 32, 32
            Conv2dBlock(128, 256, 4, 2, 1, norm=norm, activation=activation),  # B,  256, 16, 16
            Conv2dBlock(256, 512, 4, 2, 1, norm=norm, activation=activation),  # B,  512, 8, 8
            Conv2dBlock(512, 1024, 4, 2, 1, norm=norm, activation=activation),  # B,  1024, 4, 4
        )

        self.fc = nn.Sequential(
            LinearBlock(1024 * 4 * 4, 1024, activation='none', use_bias=True),
            LinearBlock(1024, out_dim, activation='none', use_bias=True),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


"""
Classifier for VP model.
"""

class Classifier(nn.Module):
    def __init__(self, nc_in=3, out_dim=1, norm='instancenorm', activation='lrelu'):
        super(Classifier, self).__init__()

        self.encoder = nn.Sequential(
            Conv2dBlock(nc_in, 4, 4, 2, 1, norm='none', activation=activation),  # B,  32, 32, 32
            Conv2dBlock(4, 4, 4, 2, 1, norm=norm, activation=activation),  # B,  32, 16, 16
            Conv2dBlock(4, 4, 4, 2, 1, norm=norm, activation=activation),  # B,  32, 4, 4
            Conv2dBlock(4, 4, 4, 2, 1, norm=norm, activation=activation),  # B,  32, 4, 4
        )

        self.fc = nn.Sequential(
            LinearBlock(4 * 4 * 4, 16, activation=activation, use_bias=True),
            LinearBlock(16, out_dim, activation='none', use_bias=True),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)




class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor list) - - tpyically the prediction output from a discriminator; supports multi Ds.
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
