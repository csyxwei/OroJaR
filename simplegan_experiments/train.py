"""General-purpose training script for image generation.

This script works for various models (with option '--model': e.g., gan, gan128) and
different datasets (with option '--dataset_mode': e.g., dsprites, celeba).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a GAN model:
        python train.py --dataroot ./datasets/ --name dsprites_orojar --model gan
    Train a GAN128 model:
        python train.py --dataroot ./datasets/CelebA/ --name celeba_dspirtes --model gan128

See options/base_options.py and options/train_options.py for more training options.
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
from util.util import loop_print
from util.vis_tools import make_mp4_video
import os


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()   # get training options
    opt.log_file = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    loop_print(opt.log_file, 'The number of training images = %d' % dataset_size)
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations


    sample_z = torch.randn(8, model.nz).to(model.device)
    sample_z[0, :] = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on Tensorboard and save image
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data, total_iters)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                loop_print(opt.log_file, 'saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('latest')

            if total_iters % opt.display_sample_freq == 0:
                model.set_train(False)
                with torch.no_grad():
                    video = make_mp4_video(model.netG, sample_z, extent=2, interp_steps=40)
                    visualizer.display_video_results(video, epoch)
                loop_print(opt.log_file, 'sampled..................')
                model.set_train(True)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            loop_print(opt.log_file, 'saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        loop_print(opt.log_file, 'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))