from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visualization parameters
        parser.add_argument('--display_freq', type=int, default=100000, help='frequency of showing training results on screen')
        parser.add_argument('--display_sample_freq', type=int, default=100000, help='frequency of showing testing results on screen')
        parser.add_argument('--display_id', type=int, default=1, help='Use the tensorboard?')
        parser.add_argument('--print_freq', type=int, default=10000, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=100000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=600, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='wgangp', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')


        parser.add_argument('--reg_lambda', type=float, default=1e-5, help='initial learning rate for adam')
        parser.add_argument('--reg_type', type=str, default='orojar', help='[orojar | hp | vp] the type of Regularization is applied: OroJaR/ Hessian Penalty/ VP')
        parser.add_argument('--nz', type=int, default=6, help='Number of components in G\'s latent space.')
        parser.add_argument('--nc_out', type=int, default=3, help='Channal number of the output image')
        parser.add_argument('--epsilon', type=float, default=0.1,
                            help='The granularity of the finite differences approximation. '
                                 'When changing this value from 0.1, you will likely need to change '
                                 'hp_lambda as well to get optimal results.')
        parser.add_argument('--num_rademacher_samples', type=int, default=2,
                            help='The number of Rademacher vectors to be sampled per-batch element '
                                 'when estimating the OroJaR. Must be >=2. Setting this '
                                 'parameter larger can result in GPU out-of-memory for smaller GPUs!')
        self.isTrain = True

        return parser
