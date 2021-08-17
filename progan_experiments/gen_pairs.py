"""
Code for generating the paired images for calculate VP metric.

For example, the following command works:

python gen_pairs.py --models path_to_orojar_model
"""

from dnnlib import EasyDict
import dnnlib.tflib as tflib
import argparse
import os
import config
from training.misc import load_pkl, find_pkl
import numpy as np
from glob import glob
from download import find_model

# export TF_FORCE_GPU_ALLOW_GROWTH='true'

def main():
    parser = argparse.ArgumentParser(description='Generate Paired Images for VP Metric')

    # Model/Dataset Parameters:
    parser.add_argument('--models', required=True, nargs='+',
                        help='Either the number of experiment in results directory, or a direct path to a .pkl '
                             'network checkpoint. You can specify multiple experiments/pkls '
                             'to generate visuals for all of them with one call to this script.')
    parser.add_argument('--snapshot_kimgs', default=['latest'], nargs='+',
                        help='network-snapshot-<snapshot_kimg>.pkl to evaluate. This should either be "latest" or '
                             'a list of length equal to exp_numbers (each model needs a snapshot_kimg). '
                             'Not used if you are passing-in direct paths to .pkl checkpoints using --models.')
    parser.add_argument('--seeds', type=int, default=[0], nargs='+',
                        help='Seed for sampling the latent noise')
    parser.add_argument('--nz_per_vid', default=12, type=int,
                        help='Number of z components to visualize per-video. This controls the "height" of the '
                             'generated videos.')
    parser.add_argument('--samples', default=10000, type=int,
                        help='Number of z samples to use (=interp_batch_size). This controls the "width" of the '
                             'generated videos.')
    parser.add_argument('--steps', default=90, type=int,
                        help='Number of frames in video (=granularity of interpolation).')
    parser.add_argument('--extent', default=2.0, type=float,
                        help='How "far" to move the z components (from -extent to extent)')
    parser.add_argument('--minibatch_size', default=10, type=int,
                        help='Batch size to use when generating frames. If you get memory errors, try decreasing this.')
    parser.add_argument('--interpolate_pre_norm', action='store_true', default=False,
                        help='If specified, interpolations are performed before the first pixel norm layer in G.'
                             'You should use this when nz is small (e.g., CLEVR-U).')
    parser.add_argument('--no_loop', action='store_true', default=False,
                        help='If specified, saved video will not "loop".')
    parser.add_argument('--stills_only', action='store_true', default=False,
                        help='If specified, only save frames instead of an mp4 video.')
    parser.add_argument('--n_frames_to_save', type=int, default=0,
                        help='Number of "flattened" frames from video to save to png (0=disable).')
    parser.add_argument('--transpose', action='store_true', default=False,
                        help='If specified, flips columns with rows in the video.')
    parser.add_argument('--pad_x', default=0, type=int,
                        help='Padding between samples in video. WARNING: This can '
                             'cause weird problems with the video when '
                             'nz_per_vid > 1, so be careful using this.')
    parser.add_argument('--pad_y', default=0, type=int,
                        help='Padding between rows in video. WARNING: This can '
                             'cause weird problems with the video when '
                             'nz_per_vid > 1, so be careful using this.')

    opt = parser.parse_args()
    opt = EasyDict(vars(opt))
    if opt.pad_x > 0 or opt.pad_y > 0:
        print('Warning: Using non-zero pad_x or pad_y can '
              'cause moviepy to take a long time to make the video. '
              'Also, there might be problems viewing the video with some applications '
              'such as QuickTime.')

    if os.path.isdir(opt.models[0]):
        opt.models = sorted(glob(f'{opt.models[0]}/*/*.pkl'))
    model_paths = [find_model(model) for model in opt.models]
    opt.models = [m1 if m1 else m2 for m1, m2 in zip(model_paths, opt.models)]

    if len(opt.snapshot_kimgs) == 1 and len(opt.models) > 1:
        assert opt.snapshot_kimgs[0] == 'latest'
        opt.snapshot_kimgs = ['latest'] * len(opt.models)
    else:
        assert len(opt.snapshot_kimgs) == len(opt.models)

    for model, snapshot_kimg in zip(opt.models, opt.snapshot_kimgs):
        for seed in opt.seeds:
            opt.exp_number = model
            opt.snapshot_kimg = snapshot_kimg
            opt.seed = seed
            run(opt)


from PIL import Image

def convert_to_pil_image(image):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC
    image = (image + 1) * 127.5
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return Image.fromarray(image, fmt)


def run(opt):
    # Find and load the network checkpoint:
    if not opt.exp_number.endswith('.pkl'):
        results_dir = os.path.join(config.result_dir)
        resume_pkl = find_pkl(results_dir, int(opt.exp_number), opt.snapshot_kimg)
    else:
        resume_pkl = opt.exp_number
    tflib.init_tf()
    _, _, Gs = load_pkl(resume_pkl)
    print(f'Visualizing pkl: {resume_pkl} with seed={opt.seed}')

    # SeFa code
    # weight_name = '4x4/Dense/weight'
    # weight = Gs.get_var(weight_name)
    # U, S, V = np.linalg.svd(weight, full_matrices=False)
    # new_weight = S[:, None] * V
    # Gs.set_var(weight_name, new_weight)

    # Sample latent noise:
    nz = Gs.input_shapes[0][1]
    np.random.seed(opt.seed)


    if nz < 12 and not opt.interpolate_pre_norm:
        print(f'This model uses a small z vector (nz={nz}); you might want to add '
              f'--interpolate_pre_norm to your command.')

    # Create directory for saving visualizations (hessian_penalty/visuals/{experiment_name}/seed_{N}):
    checkpoint_kimg = resume_pkl.split('network-snapshot-')[-1].split('.pkl')[0]
    checkpoint_kimg = checkpoint_kimg.split('/')[-1]
    exp_name = resume_pkl.split('/')[-2]
    out_path = os.path.join(config.test_dir, exp_name, checkpoint_kimg, f'seed_{opt.seed}')
    os.makedirs(out_path, exist_ok=True)


    n_samples = opt.samples
    batch_size = opt.minibatch_size
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


        idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # original
        delta_dim = np.random.randint(0, 12, size=[batch_size])



        ## Used for test simple dataset, the paired images are generated from top six activeness score dimensions.
        ## You should also uncomment the corresponding idx line.
        # idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) # progan or sefa
        # idx = np.array([1, 5, 9, 4, 3, 11, 2, 8, 10, 7, 0, 6]) # simple orojar
        # idx = np.array([6, 11, 2, 1, 10, 7, 8, 3, 5, 4, 9, 0]) # simple hessian ft
        # delta_dim = np.random.randint(0, 6, size=[batch_size])


        delta_dim = idx[delta_dim]

        delta_onehot = np.zeros((batch_size, nz))
        delta_onehot[np.arange(delta_dim.size), delta_dim] = 1
        z_2 = np.where(delta_onehot > 0, z_2, z_1)

        delta_z = z_1 - z_2

        if i == 0:
            labels = delta_z
        else:
            labels = np.concatenate([labels, delta_z], axis=0)

        fakes_1 = Gs.run(z_1, grid_labels, is_validation=True,
                                  minibatch_size=batch_size, normalize_latents=False)
        fakes_2 = Gs.run(z_2, grid_labels, is_validation=True,
                         minibatch_size=batch_size, normalize_latents=False)

        for j in range(fakes_1.shape[0]):
            pair_np = np.concatenate([fakes_1[j], fakes_2[j]], axis=2)
            img = convert_to_pil_image(pair_np)
            img.save(
                os.path.join(out_path,
                             'pair_%06d.jpg' % (i * batch_size + j)))

    np.save(os.path.join(out_path, 'labels.npy'), labels)


if __name__ == "__main__":
    main()
