from ctypes import c_int, c_int, c_bool
from typing import Optional
from .nn import NeuralLayer, C_NeuralLayer
from ..common.common import get_device_by_name

class LinearLayer(NeuralLayer):
    def __init__(self, input_size: int, output_size: int, 
                 use_grad: Optional[bool] = True, 
                 device: Optional[str] = 'cpu'):
        _init_linear_layer = self.c_lib.init_linear_layer
        _init_linear_layer.argtypes = [c_int, c_int, c_bool, c_int]
        _init_linear_layer.restype = C_NeuralLayer

        device = get_device_by_name(device)

        super().__init__(_init_linear_layer(input_size, output_size, use_grad,
                                            device))
