import torch
from torch.nn import functional as F


class RBMobj:
    def __init__(self, h_dim: int, v_dim: int, k: int, lr: float = 0.001,
                 max_iter: int = 100, tol: float = 1E-3, epoch: int = 10,
                 trace: bool = False):
        assert h_dim > 0, f'Found zero/negative dimension in latent dimension which is impossible {h_dim}.'
        assert v_dim > 0, f'Found zero/negative dimension in visible dimension which is impossible {v_dim}.'
        assert lr > 0, f'Found zero/negative learning rate which is impossible {lr}.'
        self.h_dim = h_dim
        self.v_dim = v_dim
        self.k = k

        self.lr = torch.tensor(lr, dtype=torch.float64)
        self.tol = tol
        self.epoch = epoch
        self.trace = trace

        # model parameters
        self.W = None
        self.a = None
        self.b = None

    def fit(self, X: torch.Tensor):
        # random initialization
        self.W = torch.randn(self.v_dim, self.h_dim, dtype=torch.float64)
        self.a = torch.randn(self.v_dim, dtype=torch.float64)
        self.b = torch.randn(self.h_dim, dtype=torch.float64)

        # training
        n_sample, x_dim = X.shape

        # epoch loop
        for epoch in range(self.epoch):
            if self.trace:
                print(f'Running epoch: {epoch}')

            # training sample loop
            for i in range(n_sample):
                v = X[i, :]
                h = torch.zeros(self.h_dim, dtype=torch.float64)
                v_next = torch.zeros(self.v_dim, dtype=torch.float64)
                h_next = torch.zeros(self.h_dim, dtype=torch.float64)

                # k-step CD learning
                v_temp = v.clone()
                for t in range(self.k):
                    # sampling
                    h_temp = self._sampling_hidden(v_temp)
                    v_temp = self._sampling_visible(h_temp)

                    # save
                    if t == 0:
                        h = h_temp.clone()
                    if t == self.k - 1:
                        h_next = h_temp.clone()
                        v_next = v_temp.clone()

                # update
                self.W += self.lr * (torch.outer(v, h_next) - torch.outer(v_next, h_next))
                self.a += self.lr * (v - v_next)
                self.b += self.lr * (h - h_next)

    def _sampling_hidden(self, v: torch.Tensor):
        # sample from p(h|v)
        p = torch.matmul(torch.t(self.W), v)
        p += self.b
        p = F.sigmoid(p)
        return torch.bernoulli(p)

    def _sampling_visible(self, h: torch.Tensor):
        # sample from p(v|h)
        p = torch.matmul(self.W, h)
        p += self.a
        p = F.sigmoid(p)
        return torch.bernoulli(p)
