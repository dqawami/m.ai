from ctypes import c_int, c_int, c_bool
from typing import Optional
from .nn import NeuralLayer, C_NeuralLayer
from ..common.common import get_device_by_name

class FlattenLayer(NeuralLayer):
    def __init__(self, dim: int, use_grad: Optional[bool] = True, 
                 device: Optional[str] = "cpu"):
        _init_flatten_layer = self.c_lib.init_flatten_layer
        _init_flatten_layer.argtypes = [c_int, c_bool, c_int]
        _init_flatten_layer.restype = C_NeuralLayer

        device = get_device_by_name(device)

        super().__init__(_init_flatten_layer(dim, use_grad, device))
