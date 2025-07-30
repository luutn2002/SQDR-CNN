# SQDR-CNN

The official repository for "Parameter efficient hybrid spiking-quantum convolutional neural network with surrogate gradient and quantum data-reupload"

## Overview

Source code for experimenting with a spiking-CNN model with [quantum data-reupload](https://quantum-journal.org/papers/q-2020-02-06-226/?utm_source=researcher_app&utm_medium=referral&utm_campaign=RESR_MRKT_Researcher_inbound) using [surrogate gradient](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/surrogate.html).

## Quickstart

This is a quickstart guide on how to use our model as a package 

### Step 1: Environment setup and repo download

To setup the environment testing with this encoder, you will need Pytorch, Pennylane and SpikingJelly. We suggest using conda environment with:

```bash
$ conda create -n env python=3.12.2
$ conda install pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia #As latest pytorch conda guide, change cuda version suitable to your case.
$ pip install spikingjelly
$ pip install pennylane --upgrade
$ pip install git+https://github.com/luutn2002/SQDR-CNN.git
```

or clone and modify locally:

```bash
$ git clone https://github.com/luutn2002/SQDR-CNN.git
$ cd sqdr_cnn
$ pip install -r requirements.txt
```

### Step 2: Usage

To ensure reproducibility, remember to use static random seed:
```python
import torch
import numpy as np
import random

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

To use the model, we can simply import as a normal Pytorch model:

```python
from sqdr_cnn import SQDR_CNN

T = 10
IN_CHANNELS = 1
NUM_CLASS = 10 # Change if needed
N_QUBITS = 9
NUM_BLOCK = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SQDR_CNN(T,
                IN_CHANNELS
                NUM_CLASS,
                n_wires=N_QUBITS,
                qdr_block=NUM_BLOCK).to(DEVICE)

output = model(torch.rand(T, IN_CHANNELS, 32, 32, device=DEVICE))
```

### Step 3: Preprocessing

Trained data are preprocessed with Pytorch predefined datasets, dataloader and transform:

```python
DATADIR = "./Datasets"
BATCH_SIZE = 16

TRANSFORM_TRAIN = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(),
])

TRANSFORM_TEST = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

train_set = torchvision.datasets.MNIST(
        root=DATADIR,
        train=True,
        transform=TRANSFORM_TRAIN,
        download=True)

val_set = torchvision.datasets.MNIST(
        root=DATADIR,
        train=False,
        transform=TRANSFORM_TEST,
        download=True)


if target_digits:
    indices = [i for i, (_, label) in enumerate(train_set) if label in target_digits]
    train_set = torch.utils.data.Subset(train_set, indices)

    indices = [i for i, (_, label) in enumerate(val_set) if label in target_digits]
    val_set = torch.utils.data.Subset(val_set, indices)

train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True,
                                           generator=torch.Generator(device=DEVICE),
                                           collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in torch.utils.data.dataloader.default_collate(x)))
test_loader = torch.utils.data.DataLoader(val_set, 
                                          batch_size=BATCH_SIZE,
                                          generator=torch.Generator(device=DEVICE),
                                          collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in torch.utils.data.dataloader.default_collate(x)))
```

Encoder initializing with SpikingJelly is not needed, as we already include it within the model:

```python
class SQDR_CNN(nn.Module):
        ... # Init fields witin class
        self.encoder = encoding.WeightedPhaseEncoder(K=8)
        
    def forward(self, x):
        ... # Data inference
        self.encoder.reset() # Encoder automatically reset after inference
        return x
```
## Used datasets

All used dataset is included in [torchvision](https://docs.pytorch.org/vision/main/datasets.html).

## License

Source code is licensed under MIT License.

## Contribution guidelines

Please open an issue or pull request if there are bugs or contribution to be made. Thank you.

## Others
Pytorch [guides](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html), Pennylane [guides](https://pennylane.ai/) and SpikingJelly [guides](https://spikingjelly.readthedocs.io/zh-cn/latest/#index-en) are available.

> Note: Aside from Pennylane implementation, we also have [Torchquantum](https://github.com/mit-han-lab/torchquantum) implementation of the PQC modules (around 100x faster but does not have similar noise model implementation in Pennylane). Local cloning for modification is better in this scenario.

## Citations
Paper is under review. Temporarily please cite as:
```bibtex
@misc{luu2025parameter,
  author = {Luu T. Nhan},
  title = {Parameter efficient hybrid spiking-quantum convolutional neural network with surrogate gradient and quantum data-reupload},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/luutn2002/SQDR-CNN}},
}
```
