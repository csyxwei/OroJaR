"""
Code for generating the paired images for calculate VP metric.

For example, the following command works:

python gen_pairs.py
--model_path path_to_OroJaR_netG_model
--model_name OroJaR
--model_type gan
"""

import argparse
import torch
from util.util import loop_print, torch_load
import os
import numpy as np
import cv2
from models import base_networks as net


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the Disentanglement of ProgressiveGAN')

    # Model/Dataset Parameters:
    parser.add_argument('--model_path', required=True,
                        help='Number of model paths. You can specify multiple experiments '
                             'to generate visuals for all of them with one call to this script.')
    parser.add_argument('--nz', default=6, type=int,
                        help='Number of components in G\'s latent space.')
    parser.add_argument('--nc_out', default=1, type=int,
                        help='Channal number of the output image')
    parser.add_argument('--model_name', default='OroJaR',
                        help='Give names to the models you are evaluating')
    parser.add_argument('--model_type', default='gan',
                        help='Give model types to the models you are evaluating')
    parser.add_argument('--sefa', default=False, type=str2bool,
                        help='Use SeFa on the first conv/fc layer to achieve disentanglement.')
    parser.add_argument('--save_dir', type=str, default='./pairs', help='figures are saved here')

    opt = parser.parse_args()

    if opt.model_type == 'gan':
        netG = net.Generator(nz=opt.nz, nc_out=opt.nc_out)
    elif opt.model_type == 'gan128':
        netG = net.Generator128(nz=opt.nz, nc_out=opt.nc_out)
    else:
        raise TypeError()

    state_dict = torch_load(opt.model_path, map_location='cpu')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    if opt.sefa:
        weight_name = 'layers.0.layers.conv.weight'
        weight = state_dict[weight_name]
        size = weight.size()
        weight = weight.reshape((weight.size(0), -1)).T
        U, S, V = torch.svd(weight)
        new_weight = U * S.unsqueeze(0).repeat((U.size(0), 1))
        state_dict[weight_name] = new_weight.T.reshape(size)

    netG.load_state_dict(state_dict)
    netG = netG.cuda().eval()

    out_path = os.path.join(opt.save_dir, opt.model_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    nz = opt.nz
    n_samples = 10000
    batch_size = 10
    n_batches = n_samples // batch_size

    for i in range(n_batches):
        print('Generating image pairs %d/%d ...' % (i, n_batches))
        grid_labels = np.zeros([batch_size, 0], dtype=np.float32)

        z_1 = np.random.uniform(low=-2,
                                high=2,
                                size=[batch_size, nz])

        z_2 = np.random.uniform(low=-2,
                                high=2,
                                size=[batch_size, nz])

        idx = np.array(list(range(100)))  # full

        delta_dim = np.random.randint(0, opt.nz, size=[batch_size])
        delta_dim = idx[delta_dim]

        delta_onehot = np.zeros((batch_size, nz))
        delta_onehot[np.arange(delta_dim.size), delta_dim] = 1

        z_2 = np.where(delta_onehot > 0, z_2, z_1)

        delta_z = z_1 - z_2


        if i == 0:
            labels = delta_z
        else:
            labels = np.concatenate([labels, delta_z], axis=0)
        z_1_th = torch.from_numpy(z_1).float().cuda()
        z_2_th = torch.from_numpy(z_2).float().cuda()
        fakes_1 = netG(z_1_th)
        fakes_2 = netG(z_2_th)

        for j in range(fakes_1.shape[0]):
            img_1 = fakes_1[j].cpu().detach().numpy().transpose((1, 2, 0))
            img_2 = fakes_2[j].cpu().detach().numpy().transpose((1, 2, 0))
            pair_np = np.concatenate([img_1, img_2], axis=1)
            img = (pair_np + 1) * 127.5

            cv2.imwrite(
                os.path.join(out_path,
                             'pair_%06d.jpg' % (i * batch_size + j)), img)

    np.save(os.path.join(out_path, 'labels.npy'), labels)