# modules 
import numpy as np
import pandas as pd
import time
from numpy.random import *
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
from torch import nn, optim


class RBM(nn.Module):
    """Creating RBM with PyTorch
    Attributes:
        b : bias of visible units
        c : bias of hidden units
        w : weight
        k : Sampling Times 
        n_hid (int) : numbers of hidden units
        n_vis (int) : numbers of visible units
        epoch (int) : epoch numbers
        learning_rate : learning rate
        batch_size : batch size
    """
    def __init__(self, n_vis=784, n_hid=128, k=15, epoch=10, learning_rate=0.1, 
                batch_size=100, initial_std=0.01, seed=0, device='cpu'):
        super(RBM, self).__init__()
        self.n_hid = n_hid
        self.n_vis = n_vis
        self.device = device
        self.b = torch.zeros(1, n_vis,  device=device)
        self.c = torch.zeros(1, n_hid,  device=device)
        self.w = torch.empty((n_hid, n_vis), device=device).normal_(mean=0, std=initial_std)
        self.k = k
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def visible_to_hidden(self, v):
        """
        Sampling hidden units from visible units
        """
        p = torch.sigmoid(F.linear(v, self.w, self.c))
        return p.bernoulli()

    def hidden_to_visible(self, h):
        """
        Sampling visible units from hidden units
        """
        p = torch.sigmoid(F.linear(h, self.w.t(), self.b))
        return p.bernoulli()
    
    def com_hiddens(self, v):
        """
        Calculating P(h=1|v)
        """
        return torch.sigmoid(F.linear(v, self.w, self.c))

    def sample_v(self, v, gib_num=1):
        """
        Sampling visible units
        """
        v = v.view(-1, self.n_vis)
        v = v.to(self.device)
        h = self.visible_to_hidden(v)
        # Gibbs Sampling 1 ~ k
        for _ in range(gib_num):
            v_gibb = self.hidden_to_visible(h)
            h = self.visible_to_hidden(v_gibb)
        return v, v_gibb

    def sample_ph(self, v):
        """
        Sampling ph
        """
        v = v.view(-1, self.n_vis)
        v = v.to(self.device)
        ph = torch.sigmoid(F.linear(v, self.w, self.c)) 
        h = ph.bernoulli()
        # Gibbs Sampling 1 ~ k
        for _ in range(self.k):
            v_gibb = self.hidden_to_visible(h)
            ph_gibb =  torch.sigmoid(F.linear(v_gibb, self.w, self.c))
            h = ph_gibb.bernoulli()
        return ph, ph_gibb

    def free_energy(self, v):
        """
        Caluculating Free energy
        """
        v_term = torch.matmul(v, self.b.t())
        w_x_h = torch.matmul(v, self.w.t()) + self.c
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return -h_term - v_term

    # psuedo likelihoodを計算
    def loss(self, v):
        """
        Caluculating psuedo-likelihood
        """
        flip = torch.randint(0, v.size()[1], (1,))
        v_fliped = v.clone()
        v_fliped[:, flip] = 1 - v_fliped[:, flip]
        free_energy = self.free_energy(v)
        free_energy_fliped = self.free_energy(v_fliped)
        return  v.size()[1]*F.softplus(free_energy_fliped - free_energy)

    def batch_fit(self, v_pos):
        """
        Learning Batch Samples
        """
        # positive part
        ph_pos = self.com_hiddens(v_pos)
        # negative part
        v_neg = self.hidden_to_visible(self.h_samples) 
        ph_neg = self.com_hiddens(v_neg)

        lr = (self.learning_rate) / v_pos.size()[0]
        # Update W
        update = torch.matmul(ph_pos.t(), v_pos) - torch.matmul(ph_neg.t(), v_neg)
        self.w += lr * update
        self.b += lr * torch.sum(v_pos - v_neg, dim=0)
        self.c += lr * torch.sum(ph_pos - ph_neg, dim=0)
        # memory of PCD method
        self.h_samples = ph_neg.bernoulli()


    def fit(self, train_loader, train_tensor):
        self.losses = []
        # initialization of hidden units
        self.h_samples = torch.zeros(self.batch_size, self.n_hid, device=device)
        for epoch in range(self.epoch):
            running_loss = 0.0
            start = time.time()
            for id, (data, target) in enumerate(train_loader):
                data = data.view(-1, self.n_vis)
                data = data.to(self.device)
                target = target.to(self.device)
                self.batch_fit(data)
            # Caluculating Psuedo-likelihood 
            for _, (data, target) in enumerate(train_loader):
                data = data.view(-1, self.n_vis)
                data = data.to(self.device)
                target = target.to(self.device)
                running_loss += self.loss(data).mean().item()
            self.losses.append(running_loss)
            end = time.time()
            print(f'Epoch {epoch+1} pseudo-likelihood = {running_loss:.4f}, {end-start:.2f}s')