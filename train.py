import numpy as np
import idx2numpy
import torch

from model import RBM
from torch.utils.data import DataLoader, TensorDataset

## MNIST


# region load images into tensor
X_tr = idx2numpy.convert_from_file(r'train-images.idx3-ubyte')
(N_tr, nx, ny) = X_tr.shape

# add single channel
X_tr = np.expand_dims(X_tr, axis=1)

# normalize into [0, 1]
X_tr = torch.Tensor(X_tr / 255.0)
dataset_tr = TensorDataset(X_tr)

batch_size = 64
dataloader_tr = DataLoader(X_tr, batch_size=batch_size)
# endregion


# region device, model and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# endregion
