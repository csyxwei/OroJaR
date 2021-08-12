from data.base_dataset import BaseDataset
import cv2
import os
import torch

class CelebADataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        data_dir = opt.dataroot
        img_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]
        self.img_paths = img_paths
        self._img_num = len(img_paths)
        self.load_size = 128

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[24:-24, 4:-4, :]
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        # to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1).astype('float32'))
        # normalize
        img = img.div(127.5).sub(1)

        return  {'img':img, 'path':img_path, }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self._img_num