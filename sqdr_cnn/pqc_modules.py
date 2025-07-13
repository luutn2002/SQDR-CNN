import functools
import math
import random
import pennylane as qml
from torch import nn
import torch

class QuantumReuploadPQC(nn.Module):
    def __init__(self,
                 n_wires: int,
                 num_block: int,
                 use_noise: bool = True,
                 p_bitflip: float = 0.02,
                 p_depol: float = 0.03,
                 init_mode: str = "normal",
                 init_range: list[float, float] = [0, 2*math.pi], # type: ignore
                 device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"): # type: ignore
        super().__init__()

        match init_mode:
            case "normal": 
                init_function = torch.nn.init.trunc_normal_
            case "uniform": 
                init_function = torch.nn.init.uniform_
            case _:
                raise Exception("Initialization mode not found.")

        if use_noise:
            dev = qml.device("default.mixed", wires=n_wires)

            def sigmoid(x):
                return 1/(1+torch.exp(-x))
            
            @qml.qnode(dev)
            def qlayer(inputs, weights):
                for n in range(num_block):
                    for i in range(n_wires):
                        qml.Rot(inputs[3*i], inputs[3*i+1], inputs[3*i+2], wires=i)
                        qml.Rot(weights[n][i][0], weights[n][i][1], weights[n][i][2], wires=i)

                        qml.BitFlip(p_bitflip, wires=i)
                        qml.DepolarizingChannel(p_depol, wires=i)
                        qml.AmplitudeDamping(sigmoid(inputs[3*i + random.randint(0, 2)]), wires=i)

                    if num_block > 1 and n < (num_block-1):
                        for j in range(n_wires-1): qml.CZ(wires=[j, j+1])

                        qml.BitFlip(p_bitflip, wires=i) # type: ignore
                        qml.DepolarizingChannel(p_depol, wires=i) # type: ignore
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]
        
        else:
            dev = qml.device("default.qubit", wires=n_wires)

            @qml.qnode(dev)
            def qlayer(inputs, weights):
                for n in range(num_block):
                    for i in range(n_wires):
                        qml.Rot(inputs[3*i], inputs[3*i+1], inputs[3*i+2], wires=i)
                        qml.Rot(weights[n][i][0], weights[n][i][1], weights[n][i][2], wires=i)
                    if num_block > 1 and n < (num_block-1): 
                        for j in range(n_wires-1): qml.CZ(wires=[j, j+1])
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        self.circuit = qml.qnn.TorchLayer(qlayer, 
                                          {"weights": (num_block, n_wires, 3)}, 
                                          init_method=functools.partial(init_function, 
                                                                        a=init_range[0], 
                                                                        b=init_range[1])).to(device)

    def forward(self, x: torch.Tensor):
        return torch.stack([self.circuit(x[i]) for i in range(len(x))])