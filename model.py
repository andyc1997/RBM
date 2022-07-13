import torch


# Restricted Boltzmann Machine (Binary data)
class RBMobj:
    def __init__(self, h_dim: int, v_dim: int, k: int, lr: float = 0.001,
                 tol: float = 1E-3, epoch: int = 10, trace: bool = False):
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
        self.pseudo_lik = torch.zeros(epoch, dtype=torch.float64)

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
                v = X[i, :].clone()
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

            # loss update by a random sample, and corrupt by a random bit flip
            idx_sample = torch.randint(0, n_sample, size=(1,)).item()
            idx_bit = torch.randint(0, self.v_dim, size=(1,)).item()

            v_corrpt_rand = X[idx_sample, :].clone()
            d_free_energy = self._free_energy(v_corrpt_rand)
            v_corrpt_rand[idx_bit] = 1 - v_corrpt_rand[idx_bit]
            d_free_energy -= self._free_energy(v_corrpt_rand)

            self.pseudo_lik[epoch] = self.v_dim * torch.log(torch.sigmoid(-d_free_energy))
            if self.trace:
                print(f'Difference in free energy: {d_free_energy}')
                print(f'Pseudo-likelihood: {torch.round(self.pseudo_lik[epoch], decimals=4)}')

    def _sampling_hidden(self, v: torch.Tensor):
        # sample from p(h|v)
        p = torch.matmul(torch.t(self.W), v)
        p += self.b
        p = torch.sigmoid(p)
        return torch.bernoulli(p)

    def _sampling_visible(self, h: torch.Tensor):
        # sample from p(v|h)
        p = torch.matmul(self.W, h)
        p += self.a
        p = torch.sigmoid(p)
        return torch.bernoulli(p)

    def _free_energy(self, v: torch.Tensor):
        E = torch.inner(self.a, v)
        E += torch.sum(1 + torch.exp(torch.matmul(torch.t(self.W), v) + self.b))
        return -E

    def _energy(self, v: torch.Tensor, h: torch.Tensor):
        E = torch.inner(v, torch.matmul(self.W, h))
        E += torch.inner(self.a, v)
        E += torch.inner(self.b, h)
        return -E
