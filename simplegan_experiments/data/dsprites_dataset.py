from data.base_dataset import BaseDataset
import os
import torch
from util.util import np_load

class DSpritesDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        data_dir = opt.dataroot
        self.data = np_load(os.path.join(data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), encoding='bytes')
        self.images = self.data['imgs']
        self.latents_values = self.data['latents_values']


    def __getitem__(self, index):
        img = self.images[index:index+1]
        # to tensor
        img = torch.from_numpy(img.astype('float32'))
        # normalize
        img = img.mul(2).sub(1)

        return  {'img':img, 'path':'',}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)