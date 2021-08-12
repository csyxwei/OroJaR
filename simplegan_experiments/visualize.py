"""
Code for creating a disentanglement visualization for network checkpoints. Each column in a saved video corresponds to
interpolating a single z component from -extent to +extent (usually extent=2 but you can control it as an command line
argument). The i-th col controls the i-th z component.

For example, the following command works:

python visualize.py
--model_path path_to_OroJaR_netG_model
--model_name OroJaR
--model_type gan
"""

import argparse
import torch
from util.util import  torch_load, scaling, th_save_image
import os
from util.vis_tools import make_mp4_video
from models import base_networks as net
import imageio
from os.path import join

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
    parser.add_argument('--samples', default=1, type=int,
                        help='Number of z samples to use (=interp_batch_size). This controls the "width" of the '
                             'generated videos.')
    parser.add_argument('--extent', default=2.0, type=float,
                        help='How "far" to move the z components (from -extent to extent)')
    parser.add_argument('--steps', default=40, type=int,
                        help='Number of frames in video (=granularity of interpolation).')
    parser.add_argument('--n_frames_to_save', type=int, default=9,
                        help='Number of "flattened" frames from video to save to png (0=disable).')
    parser.add_argument('--model_name', default='OroJaR',
                        help='Give names to the models you are evaluating')
    parser.add_argument('--model_type', default='gan',
                        help='Give model types to the models you are evaluating')
    parser.add_argument('--sefa', default=False, type=str2bool,
                        help='Use SeFa on the first conv/fc layer to achieve disentanglement.')
    parser.add_argument('--save_dir', type=str, default='./', help='figures are saved here')


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

    out_path = os.path.join(opt.save_dir, 'visuals', opt.model_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    frame_path = os.path.join(out_path, 'frames')
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    sample_z = torch.randn(opt.samples, opt.nz).cuda()

    with torch.no_grad():
        video, frames = make_mp4_video(netG, sample_z, extent=opt.extent, interp_steps=opt.steps, n_frames_to_save=opt.n_frames_to_save, return_frames=True)
        imageio.mimwrite(join(out_path, f'z000_to_z{opt.nz-1:03}.mp4'), video)
        for i, v in enumerate(frames):
            x = scaling(torch.cat(v, dim=3), multi=1).clamp_(0, 1)
            th_save_image(x.data.cpu(), join(frame_path, f'frame{i:03}.png'), nrow=1, padding=0)

    print('Visulized..................')
