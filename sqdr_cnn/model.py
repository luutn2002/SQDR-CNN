from torch import nn
import torch
from spikingjelly.activation_based import encoding
import math

from sqdr_cnn.pqc_modules import QuantumReuploadPQC
from sqdr_cnn.snn_modules import SpikingConvolutionalEncoder, GradientCheckpointingMLP

class SQDR_CNN(nn.Module):
    def __init__(self,
                 T: int,
                 in_channels: int,
                 num_class: int,
                 n_wires: int = 9,
                 qdr_block: int = 2,
                 init_mode: str = "normal",
                 init_range: list[float, float] = [0, 2*math.pi], # type: ignore
                 use_mlp: bool = True):
        super().__init__()
        self.T = T
        self.encoder = encoding.WeightedPhaseEncoder(K=8)
        self.snn = SpikingConvolutionalEncoder(T, 
                                               in_channels, 
                                               n_wires)
        self.qdr = QuantumReuploadPQC(n_wires,
                                      qdr_block,
                                      init_mode=init_mode,
                                      init_range=init_range)
        self.ec = None
        if use_mlp: self.ec = GradientCheckpointingMLP(n_wires, num_class)
        
    def forward(self, x):
        x = (x*255)/256
        x = torch.stack([self.encoder(x) for _ in range(self.T)])
        x = self.snn(x)
        x = self.qdr(x)
        if self.ec: x = self.ec(x)
        self.encoder.reset()
        return x