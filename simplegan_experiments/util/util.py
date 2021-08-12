"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from torchvision.utils import save_image as th_save_image
import imageio

def scaling(img, istanh=True, multi=255.0):
    if istanh:
        img = (img + 1.0) / 2.0
    return img * multi

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = scaling(np.transpose(image_numpy, (1, 2, 0)))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


#### for hpc IO

import time
from functools import wraps
import torch
import numpy as np

# 修饰函数，重新尝试600次，每次间隔1秒钟
# 能对func本身处理，缺点在于无法查看func本身的提示
def loop_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(600):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError as e:
                print('os error', e)
                time.sleep(1)
        return ret
    return wrapper

@loop_until_success
def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

@loop_until_success
def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


@loop_until_success
def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


@loop_until_success
def loop_print(log_file, message, also_print=True):
    with open(log_file, 'a+') as f:
        f.write('%s\n' % message)  # save the message
    if also_print:
        print(message)

@loop_until_success
def torch_save(*args, **kwargs):
    torch.save(*args, **kwargs)

@loop_until_success
def torch_save_image(*args, **kwargs):
    th_save_image(*args, **kwargs)

@loop_until_success
def torch_load(*args, **kwargs):
    file = torch.load(*args, **kwargs)
    return file

@loop_until_success
def np_save(*args, **kwargs):
    np.save(*args, **kwargs)

@loop_until_success
def np_load(*args, **kwargs):
    file = np.load(*args, **kwargs)
    return file

@loop_until_success
def tf_write(writer, k, v, epoch):
    writer.add_scalar('iter_loss/%s' % k, v, epoch)
    writer.flush()

@loop_until_success
def loop_mimsave(*args, **kwargs):
    imageio.mimsave(*args, **kwargs)