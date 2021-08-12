import torch
from .base_model import BaseModel
from . import base_networks as net
import torch.nn.functional as F
import torch.autograd as autograd
from .hessian_penalty import hessian_penalty as hp
from .orojar import orojar
import numpy as np
import itertools

class GANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0

        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_reg', 'D_GAN']
        self.visual_names = ['img', 'img_fake']
        self.model_names = ['netG']
        self.reg_type = self.opt.reg_type
        self.k = opt.num_rademacher_samples
        self.epsilon = opt.epsilon
        self.nz = nz = opt.nz
        nc_out = self.opt.nc_out
        self.netG = net.Generator(nz=nz, nc_out=nc_out).to(self.device)

        if opt.isTrain:
            self.model_names += ['netD']
            self.netD = net.Discriminator(nc_in=nc_out).to(self.device)
            self.criterionGAN = net.GANLoss(gan_mode=opt.gan_mode).to(self.device)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if opt.reg_type == 'vp':
                self.model_names += ['netC']
                self.netC = net.Classifier(nc_in=nc_out, out_dim=nz).to(self.device)
                self.optimizer_C = torch.optim.Adam(itertools.chain(self.netC.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_C)

        self.step = 0

    def set_input(self, input):
        self.img = input['img'].to(self.device)
        self.image_paths = input['path']

    def forward(self):

        self.z = z = torch.randn(self.img.size(0), self.nz).to(self.device)

        self.img_fake = self.netG(z)

        if self.opt.reg_type == 'vp':
            delta_dim = np.random.randint(0, self.nz, size=[z.size(0)])
            delta_onehot = np.zeros((z.size(0), self.nz))
            delta_onehot[np.arange(delta_dim.size), delta_dim] = 1

            self.delta_label = torch.from_numpy(delta_onehot.astype('float32')).type_as(z)

            rand_eps = np.random.normal(0, 1, [z.size(0), 1])
            delta_target = delta_onehot * rand_eps
            z_added = torch.from_numpy(delta_target.astype('float32')).type_as(z) + z
            self.img_fake2 = self.netG(z_added)

    def backward_D(self):

        pred_fake = self.netD(self.img_fake.detach())
        pred_real = self.netD(self.img)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_gp = self.gradient_penalty(self.netD, self.img, self.img_fake.detach())

        # Combined loss
        self.loss_D_GAN = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D_gp = loss_D_gp * 10
        self.loss_D = self.loss_D_GAN + self.loss_D_gp
        self.loss_D = self.loss_D
        self.loss_D.backward()

    def backward_G(self):

        pred_fake = self.netD(self.img_fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        loss_G_reg = 0

        layers = [0, 1, 2, 3]

        if self.opt.reg_type == 'hp':
            loss_G_reg = hp(G=self.netG, z=self.z, reg_layer='conv', select_layers=layers, k=self.k, epsilon=self.epsilon)
        elif self.opt.reg_type == 'vp':
            diff = self.img_fake - self.img_fake2
            pred_cls = self.netC(diff)
            loss_G_reg = F.binary_cross_entropy_with_logits(pred_cls, self.delta_label)
        elif self.opt.reg_type == 'orojar':
            loss_G_reg = orojar(G=self.netG, z=self.z, reg_layer='conv', select_layers=layers, k=self.k, epsilon=self.epsilon)

        self.loss_G_reg = loss_G_reg * self.opt.reg_lambda

        self.loss_G_GAN = loss_G_GAN

        self.loss_G = self.loss_G_GAN + self.loss_G_reg

        self.loss_G.backward()

    def optimize_parameters(self):

        self.step = self.step + 1

        for i in range(1):
            self.forward()
            self.optimizer_D.zero_grad()
            self.set_requires_grad([self.netD], True)
            self.backward_D()
            self.optimizer_D.step()

        self.forward()
        self.optimizer_G.zero_grad()
        if self.reg_type == 'vp':
            self.optimizer_C.zero_grad()
        self.set_requires_grad([self.netD], False)
        self.backward_G()
        self.optimizer_G.step()
        if self.reg_type == 'vp':
            self.optimizer_C.step()

    def gradient_penalty(self, f, real, fake=None):
        def interpolate(a, b=None):
            if b is None:  # interpolation in DRAGAN
                beta = torch.rand_like(a)
                b = a + 0.5 * a.var().sqrt() * beta
            alpha = torch.rand(a.size(0), 1, 1, 1)
            alpha = alpha.to(self.device)
            inter = a + alpha * (b - a)
            return inter

        x = interpolate(real, fake).requires_grad_(True)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = autograd.grad(
            outputs=pred, inputs=x,
            grad_outputs=torch.ones_like(pred),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad = grad.view(grad.size(0), -1)
        norm = grad.norm(2, dim=1)
        gp = ((norm - 1.0) ** 2).mean()
        return gp