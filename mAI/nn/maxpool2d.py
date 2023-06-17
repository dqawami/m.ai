from ctypes import c_int, c_int, c_bool
from typing import Union, List, Optional
from .nn import NeuralLayer, C_NeuralLayer, get_kernel_size
from ..common.common import get_device_by_name

class MaxPool2DLayer(NeuralLayer):
    def __init__(self, kernel: Union[int, List[int]], 
                 stride: Optional[Union[int, List[int]]] = 1,
                 use_grad: Optional[bool] = True, 
                 device: Optional[str] = "cpu"):
        _init_maxpool2d_layer = self.c_lib.init_maxpool2d_layer
        _init_maxpool2d_layer.argtypes = [c_int, c_int, c_int, 
                                          c_int, c_bool, c_int]
        _init_maxpool2d_layer.restype = C_NeuralLayer

        device = get_device_by_name(device)

        kernel_height, kernel_width = get_kernel_size(kernel)
        stride_height, stride_width = get_kernel_size(stride)

        super().__init__(_init_maxpool2d_layer(kernel_height, kernel_width, 
                                               stride_height, stride_width,
                                               use_grad, device))
