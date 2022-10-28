import torch
import torch.nn as nn
import torchvision
import os
import pickle
import numpy as np
import scipy
from PIL import Image

from CycleGAN import G, D

class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader):

        self.sampel_path = None
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.g12 = None
        self.g21 = None
        self.d1 = None
        self.d2 = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.use_reconst_loss = config.use_reconst_loss
        self.num_classes = config.num_classes
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.conv_dim = config.conv_dim
        self.num_layers = config.num_layers
        self.train_iter = config.train_iter
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_path = config.model_path
        self.input_dim1 = config.input_dim1
        self.input_dim2 = config.input_dim2
        self.sample_path = config.sample_path
        self.build_model()

    def build_model(self):
        """ Build discriminator and generator"""
        self.g12 = G(self.input_dim1, self.input_dim2,self.conv_dim)
        self.g21 = G(self.input_dim2, self.input_dim1, self.conv_dim)
        self.d1 = D(self.input_dim1, self.conv_dim, self.num_layers)
        self.d2 = D(self.input_dim2, self.conv_dim, self.num_layers)

        g_parameters = list(self.g12.parameters()) + list(self.g21.parameters())
        d_parameters = list(self.d1.parameters()) + list(self.d2.parameters())

        self.g_optimizer = torch.optim.Adam(g_parameters, lr=self.lr, betas=(self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(d_parameters,lr = self.lr, betas = (self.beta1, self.beta2))

    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)

    def to_np(self, x):
        return x.detach().numpy()

    def to_torch(self,x):
        return torch.from_numpy(x)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        iter_per_epoch = min(len(svhn_iter), len(mnist_iter))

        fixed_svhn = next(svhn_iter)[0]
        fixed_mnist = next(mnist_iter)[0]

        criterion = nn.CrossEntropyLoss()

        for step in range(self.train_iter+1):
            if (step+1) % iter_per_epoch == 0:
                mnist_iter = iter(self.mnist_loader)
                svhn_iter = iter(self.svhn_loader)

            svhn, s_labels = next(svhn_iter)
            svhn, s_labels = svhn, s_labels.long().squeeze()
            mnsit, m_labels = next(mnist_iter)
            mnsit, m_labels = mnsit, m_labels.long().squeeze()

            # train
            # train with real images
            self.reset_grad()
            out = self.d1(mnsit)
            d1_loss = torch.mean((out-1)**2)
            out = self.d2(svhn)
            d2_loss = torch.mean((out-1)**2)
            d_mnist_loss = d1_loss
            d_svhn_loss = d2_loss
            d_real_loss = (d1_loss+d2_loss)
            d_real_loss.backward()
            self.d_optimizer.step()

            # train with fake images
            self.reset_grad()
            fake_svhn = self.g12(mnsit)
            out = self.d2(fake_svhn)
            d2_loss = torch.mean(out**2)
            fake_mnist = self.g21(svhn)
            out = self.d1(fake_mnist)
            d1_loss = torch.mean(out**2)
            d_fake_loss = d1_loss+d2_loss
            d_fake_loss.backward()
            self.d_optimizer.step()

            #train G
            #train m-s-m
            self.reset_grad()
            fake_svhn = self.g12(mnsit)
            out = self.d2(fake_svhn)
            g_loss = torch.mean((out-1)**2)
            reconst_mnist = self.g21(fake_svhn)
            if self.use_reconst_loss:
                g_loss += torch.mean((mnsit-reconst_mnist)**2)
            g_loss.backward()
            self.g_optimizer.step()

            # train svhn-m-svhn
            self.reset_grad()
            fake_mnist = self.g21(svhn)
            out = self.d1(fake_mnist)
            reconst_svhn = self.g12(fake_mnist)
            g_loss = torch.mean((out-1)**2)
            if self.use_reconst_loss:
                g_loss += torch.mean((reconst_svhn-svhn)**2)
            g_loss.backward()
            self.g_optimizer.step()

            #print log information
            if (step+1) % self.log_step == 0:
                log = 'Step[{:d}/{:d}], d_real_loss: {:.4f}, d_mnist_loss: {:.4f}, ' \
                      'd_svhn_loss: {:.4f}, d_fake-loss: {:.4f}, g_loss: {:.4f}'
                print(log.format(step+1, self.train_iter, d_real_loss.item(), d_mnist_loss.item(),
                                 d_svhn_loss.item(), d_fake_loss.item(), g_loss.item()))

            if (step+1) % self.sample_step == 0:
                fake_svhn = self.g12(fixed_mnist)
                fake_mnist = self.g21(fixed_svhn)

                mnsit, fake_mnist = self.to_np(mnsit), self.to_np(fake_mnist)
                svhn, fake_svhn = self.to_np(svhn), self.to_np(fake_svhn)

                merged = self.merge_images(mnsit, fake_svhn)
                path = os.path.join(self.sample_path, 'sample-%d-s-m.jpg' % (step + 1))
                image = Image.fromarray(merged, mode ='RGB')
                image.save(path)
                print("Saved %s" % path)

            if (step+1) % 5000 == 0:
                model_state = {}
                model_state['g12_pth'] = self.g12.state_dict()
                model_state['g21_pth'] = self.g21.state_dict()
                model_state['d1'] = self.d1.state_dict()
                model_state['d2'] = self.d2.state_dict()
                path = self.model_path+'-%d.pth' % (step+1)
                print(path)
                torch.save(model_state, path)

































