from ctypes import c_int, c_float, c_bool
from typing import Optional
from .nn import NeuralLayer, C_NeuralLayer
from ..common.common import get_device_by_name

class BatchNorm2DLayer(NeuralLayer):
    def __init__(self, num_channels: int, eps: float = 1e-05,
                 momentum: float = 0.1, use_grad: bool = True,
                 device: Optional[str] = "cpu"):
        _init_batchnorm2d_layer = self.c_lib.init_batchnorm2d_layer
        _init_batchnorm2d_layer.argtypes = [c_int, c_float, c_float, c_bool,
                                            c_int]
        _init_batchnorm2d_layer.restype = C_NeuralLayer

        device = get_device_by_name(device)

        super().__init__(_init_batchnorm2d_layer(num_channels, eps, momentum,
                                                 use_grad, device))
