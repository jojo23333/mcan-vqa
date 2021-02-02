# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch
import torch.optim as Optim


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size, lr_multipliers=[1]):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        assert len(self.optimizer.param_groups) == len(lr_multipliers), "Length of learning rate group do not correspond"
        self.lr_multipliers = lr_multipliers

    def step(self):
        self._step += 1

        rate = self.rate()
        for i, p in enumerate(self.optimizer.param_groups):
            p['lr'] = rate * self.lr_multipliers[i]
        self._rate = rate

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * 1):
            r = self.lr_base * 1/4.
        elif step <= int(self.data_size / self.batch_size * 2):
            r = self.lr_base * 2/4.
        elif step <= int(self.data_size / self.batch_size * 3):
            r = self.lr_base * 3/4.
        else:
            r = self.lr_base

        return r


def get_optim(__C, model, data_size, lr_base=None, param_groups=None, lr_multipliers=[1.]):
    if lr_base is None:
        lr_base = __C.LR_BASE
    if param_groups is None:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    return WarmupOptimizer(
        lr_base,
        Optim.Adam(
            param_groups,
            lr=0,
            betas=__C.OPT_BETAS,
            eps=__C.OPT_EPS,
        ),
        data_size,
        __C.BATCH_SIZE,
        lr_multipliers=lr_multipliers
    )


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r
