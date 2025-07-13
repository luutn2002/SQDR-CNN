# SQDR-CNN

The official repository for "Hybrid Spiking-Quantum Data Re-upload Convolutional Neural Network"

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
```

### Step 2: Import and usage

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
Pytorch [guides](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html), Pennylane [guides](https://pennylane.ai/) and SpikingJelly [guides](https://spikingjelly.readthedocs.io/zh-cn/latest/#index-en) are available.

> Note: Aside from Pennylane implementation, we also have [Torchquantum](https://github.com/mit-han-lab/torchquantum) implementation of the PQC modules (around 100x faster but does not have similar noise model implementation in Pennylane). Local cloning for modification is better in this scenario.