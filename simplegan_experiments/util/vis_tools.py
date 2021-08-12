import torch
from . import util
from torchvision.utils import make_grid

def make_mp4_video(G, z, extent, interp_steps, n_frames_to_save=9, return_frames=False):
    """
    Generates interpolation videos using G and interp_z, then saves them in "vis_path".
    """
    assert len(z.size()) == 2

    # video
    video_list = []
    step = (extent * 2) / interp_steps
    for s in range(1 + interp_steps):
        row_list = []
        img_fake = G(z)
        row_list.append(img_fake.cpu())
        for i in range(z.size(1)):
            z_i = z.clone()
            z_i[:, i] = (-extent + s * step)
            img_fake = G(z_i)
            row_list.append(img_fake.cpu())
        # transform
        x_concat = torch.cat(row_list, dim=3)
        grid = make_grid(x_concat.data.cpu(), nrow=1, padding=0, pad_value=0,
                         normalize=False, range=None, scale_each=None)
        ndarr = util.scaling(grid).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        ndarr = ndarr.transpose((1, 2, 0))
        video_list.append(ndarr)

    # frames
    frame_list = [[] for _ in range(z.size(1))]
    step = (extent * 2) / (n_frames_to_save - 1)
    for s in range(n_frames_to_save):
        row_list = []
        img_fake = G(z)
        row_list.append(img_fake.cpu())
        for i in range(z.size(1)):
            z_i = z.clone()
            z_i[:, i] = (-extent + s * step)
            img_fake = G(z_i)
            frame_list[i].append(img_fake.cpu())

    if return_frames:
        return video_list, frame_list
    else:
        return video_list
