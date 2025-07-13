import math
import torch.utils.checkpoint as checkpoint
import torch
from torch import nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer

class BaseModule(nn.Module):
    def __init__(self):
        """
        Base class for reusable functionalities.
        """
        super(BaseModule, self).__init__()
        #Dummy tensor to trigger checkpointing
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    # To be used for separated modules in forward, 
    # example: x = checkpoint.checkpoint(self.gradient_checkpoint(self.features), 
    #                                    x, 
    #                                    use_reentrant=True)
    def gradient_checkpoint(self, module):
        def gradient_checkpoint_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return gradient_checkpoint_forward
    
    def forward(self, x):
        """
        A placeholder for forward propagation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

# Note: SpikingJelly does not have gradient checkpointing support 
# due to no TorchScript implementation. No need to inherit BaseModule
class SpikingConvolutionalEncoder(nn.Module):
    def __init__(self, 
                 T: int, 
                 in_channels: int,
                 out_channels: int,
                 conv_kernel_size: int = 3,
                 pool_kernel_size: int = 2,
                 pool_kernel_stride: int = 2):
        super().__init__()
        self.T = T
        self.conv_fc = nn.Sequential(
            layer.Conv2d(in_channels, in_channels*2, kernel_size=conv_kernel_size, padding=1, bias=False),
            layer.BatchNorm2d(in_channels*2, momentum=math.sqrt(0.1)),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(pool_kernel_size, pool_kernel_stride),

            layer.Conv2d(in_channels*2, in_channels*4, kernel_size=conv_kernel_size, padding=1, bias=False),
            layer.BatchNorm2d(in_channels*4, momentum=math.sqrt(0.1)),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(pool_kernel_size, pool_kernel_stride),  # 7 * 7

            layer.Conv2d(in_channels*4, in_channels*8, kernel_size=conv_kernel_size, padding=1, bias=False),
            layer.BatchNorm2d(in_channels*8, momentum=math.sqrt(0.1)),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(pool_kernel_size, pool_kernel_stride),  # 7 * 7

            layer.AdaptiveAvgPool2d((1, 1)),

            layer.Flatten(),
            
            layer.Linear(in_channels*8, out_channels, bias=True),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self, step_mode='m')
    
    def forward(self, x: torch.Tensor):
        return self.conv_fc(x).mean(0)
    
class GradientCheckpointingMLP(BaseModule):
    """
    MLP layer with gradient checkpointing for faster inference.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True),
            nn.ReLU()
        )
            
    def forward(self, x):
        x = checkpoint.checkpoint(self.gradient_checkpoint(self.features), 
                                  x,
                                  self.dummy_tensor, 
                                  use_reentrant=False)
        return x