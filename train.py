import numpy as np
import idx2numpy
import torch

from matplotlib import pyplot as plt
from model import RBMobj

## MNIST


# region load images into tensor
X_tr = idx2numpy.convert_from_file(r'.\..\..\deep-clustering-with-conv-autoencoders\train-images.idx3-ubyte')
(N_tr, nx, ny) = X_tr.shape

# add single channel
X_tr = np.expand_dims(X_tr, axis=1)

# binarize image
X_tr = torch.Tensor(X_tr / 255.0).type(torch.float64)
# X_tr = torch.where(X_tr > 0.8, 1, 0).type(torch.float64)

batch_size = 1000
X_tr_batch = X_tr[0:batch_size, :].reshape((batch_size, 28*28))
model = RBMobj(h_dim=100, v_dim=28*28, k=10, lr=1e-4, epoch=50, trace=True)
model.fit(X_tr_batch)
# endregion

plt.plot(model.pseudo_lik)