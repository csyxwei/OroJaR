import os
import sys
import ntpath
import time
from . import util
from torchvision.utils import make_grid, save_image


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: Create a tensorboard writer if display_id > 0
        Step 3: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.name = opt.name
        opt.log_file = self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

        if self.display_id > 0:  # create to a tensorboard writer
            from tensorboardX import SummaryWriter
            from datetime import datetime
            now = datetime.now()
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs', now.strftime("%Y%m%d-%H%M"))
            util.mkdirs([self.save_dir])
            self.writer = SummaryWriter(logdir=self.save_dir, flush_secs=1)
            self.epoch = 0
            self.walltime = time.time()
            util.loop_print(self.log_name, 'create log directory %s...' % self.save_dir)

        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.val_dir = os.path.join(opt.checkpoints_dir, opt.name, 'val')
        self.img_dir = os.path.join(self.web_dir, 'images')
        util.loop_print(self.log_name, 'create web directory %s...' % self.web_dir)
        util.loop_print(self.log_name, 'create val directory %s...' % self.val_dir)
        util.mkdirs([self.web_dir, self.img_dir, self.val_dir])
        # create a logging file to store training losses
        now = time.strftime("%c")
        util.loop_print(self.log_name, '================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, epoch):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
        """

        # save images to the disk
        for label, image in visuals.items():
            if 'mask' in label:
                image_numpy = util.tensor2mask(image)
            else:
                image_numpy = util.tensor2im(image)
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            util.save_image(image_numpy, img_path)

    def display_val_results(self, visuals, epoch, name='val', path=None):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        x = util.scaling(visuals, multi=1).clamp_(0, 1)

        if path is None:
            img_path = os.path.join(self.val_dir, 'epoch%.3d_%s.png' % (epoch, name))
        else:
            img_path = path

        util.torch_save_image(x.data.cpu(), img_path, nrow=1, padding=0)


    def display_video_results(self, visuals, epoch, name='val', path=None):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if path is None:
            img_path = os.path.join(self.val_dir, 'epoch%.3d_%s.mov' % (epoch, name))
        else:
            img_path = path

        util.loop_mimsave(img_path, visuals)


    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, total_steps):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        # Tensorboard
        for k, v in losses.items():
            if self.display_id > 0:
                if total_steps == 0:
                    pass
                else:
                    util.tf_write(self.writer, k, v, total_steps)

        util.loop_print(self.log_name, message)  # print the message