"""
Code for generating activeness figure from our paper.

you need to add --model_names and --dataset_names. For example, the
following command works:

python activeness.py
--model_paths path_to_GAN_net_G_model path_to_OroJaR_netG_model
--model_names GAN OroJaR
--model_types gan gan
--dataset_names Dsprites
"""

import argparse
import torch
from util.util import loop_print
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from models import base_networks as net
from util.util import torch_load

def sample(nz, n_samples):
    z = np.random.randn(n_samples, nz)
    z = torch.from_numpy(z.astype('float32')).cuda()
    return z

def activeness(Gs, nz, extent=2):

    scores = [[] for _ in range(len(Gs))]
    for i, G in enumerate(Gs):
        z = sample(nz, 300)
        for z_i in tqdm(range(nz)):
            z1, z2, z3 = deepcopy(z), deepcopy(z), deepcopy(z)
            z1[:, z_i] = -extent
            z2[:, z_i] = 0
            z3[:, z_i] = extent
            Gz1 = G(z1).cpu().detach().numpy()
            Gz2 = G(z2).cpu().detach().numpy()
            Gz3 = G(z3).cpu().detach().numpy()
            Gzs = np.stack([Gz1, Gz2, Gz3])
            score = np.var(Gzs, axis=0, ddof=1).mean()
            scores[i].append(score)
    return scores


def activeness_histogram(Gs, nz, out_path, model_names=None, dataset_names=None, extent=2, sort_type='per_model'):
    n_models = len(model_names)
    n_datasets = len(dataset_names)
    assert sort_type in ['per_model', 'shared']

    # Each subplot corresponds to a single dataset:
    fig = make_subplots(rows=int(np.ceil(n_datasets / 2)), cols=2, subplot_titles=dataset_names)
    scores = activeness(Gs, nz, extent)
    colors = px.colors.qualitative.Pastel[:n_models]

    for dset_ix in range(n_datasets):
        # Sort the first method's z components by activeness and use that ranking for others:
        if sort_type == 'shared':
            global_ranks = np.argsort(scores[0])[::-1]
            global_z_components = [f'z{z_ix:02}' for z_ix in global_ranks]
            global_ranks = [global_ranks] * n_models
            global_z_components = [global_z_components] * n_models
        else:  # Otherwise, the order of z components will be determined per-method
            global_ranks = None
            global_z_components = None

        for model_ix in range(n_models):
            G_scores = np.asarray(scores.pop(0))
            if sort_type == 'shared':
                G_scores = G_scores[global_ranks]
                z_components = global_z_components
            else:
                ranks = np.argsort(G_scores)[::-1]
                print(ranks)
                z_components = None
                G_scores = G_scores[ranks]
                print(G_scores)
            graph = go.Bar(x=z_components, y=G_scores, name=model_names[model_ix],
                           marker_color=colors[model_ix], showlegend=not dset_ix)
            fig.add_trace(graph, row=1 + dset_ix // 2, col=1 + dset_ix % 2)  # row and col are 1-indexed

    fig.update_xaxes(title_text='Z Component', titlefont=dict(size=18))
    fig.update_yaxes(title_text='Activeness', titlefont=dict(size=18))
    fig.update_layout(font_size=18)
    for i in fig['layout']['annotations']:  # https://github.com/plotly/plotly.py/issues/985
        i['font']['size'] = 26

    plotly.offline.plot(fig, filename=os.path.join(out_path, 'activeness.html'))


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
    parser.add_argument('--model_paths', required=True, nargs='+',
                        help='Number of model paths. You can specify multiple experiments '
                             'to generate visuals for all of them with one call to this script.')
    parser.add_argument('--nz', default=6, type=int,
                        help='Number of components in G\'s latent space.')
    parser.add_argument('--nc_out', default=1, type=int,
                        help='Channal number of the output image')
    parser.add_argument('--model_names', default=None, nargs='+',
                        help='Give names to the models you are evaluating')
    parser.add_argument('--model_types', default=None, nargs='+',
                        help='Give model types to the models you are evaluating')
    parser.add_argument('--dataset_names', default=None, nargs='+',
                        help='Give names to the datasets you are evaluating.')
    parser.add_argument('--save_dir', type=str, default='./activeness', help='figures are saved here')

    opt = parser.parse_args()

    model_paths = opt.model_paths
    model_types = opt.model_types
    model_names = opt.model_names
    dataset_names = opt.dataset_names

    Gs = []
    for model_path, model_type, model_name in zip(model_paths, model_types, model_names):
        if model_type == 'gan':
            netG = net.Generator(nz=opt.nz, nc_out=opt.nc_out)
        elif model_type == 'gan128':
            netG = net.Generator128(nz=opt.nz, nc_out=opt.nc_out)
        else:
            raise TypeError()

        state_dict = torch_load(model_path, map_location='cpu')
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        if 'sefa' in model_name.lower():
            weight_name = 'layers.0.layers.conv.weight'
            weight = state_dict[weight_name]
            size = weight.size()
            weight = weight.reshape((weight.size(0), -1)).T
            U, S, V = torch.svd(weight)
            new_weight = U * S.unsqueeze(0).repeat((U.size(0), 1))
            state_dict[weight_name] = new_weight.T.reshape(size)

        netG.load_state_dict(state_dict)
        netG = netG.cuda().eval()
        Gs.append(netG)

    out_path = os.path.join(opt.save_dir, 'visuals', 'activeness')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    activeness_histogram(Gs, nz=opt.nz, out_path=out_path, model_names=model_names, dataset_names=dataset_names)
