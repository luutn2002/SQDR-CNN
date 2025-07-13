import random
import numpy as np
import torch
import torchquantum as tq
import math

def gaussian_in_range(lower_bound, 
                      upper_bound,
                      mean=0., 
                      std_dev=1.):
    while True:
        # Sample from the Gaussian distribution
        value = np.random.normal(mean, std_dev)
        # Check if the sampled value is within the specified bounds
        if lower_bound <= value <= upper_bound:
            return value

class QuantumReuploadPQC_TQ(tq.QuantumModule):
    def __init__(self,
                 n_wires: int,
                 num_block: int,
                 init_mode: str = "normal",
                 init_range: list[float, float] = [0, 2*math.pi]): # type: ignore
        super().__init__()
        match init_mode:
            case "normal": 
                init_function = gaussian_in_range
            case "uniform": 
                init_function = random.uniform
            case _:
                raise Exception("Initialization mode not found.")
            
        self.n_wires = n_wires
        self.num_block = num_block
        self.reupload = tq.GeneralEncoder(
            [{"input_idx": [3*i, 3*i+1, 3*i+2], "func": "rot", "wires": [i]} for i in range(n_wires)]
        )
        for i in range(num_block):
            setattr(self, f'rot_{i}', tq.QuantumModule.from_op_history(
                [{'name': 'rot', 'wires': i, 'params': [init_function(init_range[0], 
                                                                      init_range[1]) for _ in range(3)], 'trainable': True} for i in range(n_wires)]))
            setattr(self, f'entanglement_{i}', tq.EntangleCircular(tq.CZ, n_wires))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor):
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, 
                                bsz=bsz, 
                                device=x.device)
        
        for i in range(self.num_block):
            rot_layer = getattr(self, f'rot_{i}')
            entanglement_layer = getattr(self, f'entanglement_{i}')
            self.reupload(qdev, x)
            rot_layer(qdev)
            entanglement_layer(qdev)
            
        x = self.measure(qdev)
        return x