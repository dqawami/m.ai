from ctypes import c_int, c_int, c_bool
from typing import Union, List, Optional
from .nn import NeuralLayer, C_NeuralLayer, get_kernel_size
from ..common.common import get_device_by_name

class Conv2DLayer(NeuralLayer):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel: Union[int, List[int]],
                 stride: Union[int, List[int]] = 1, 
                 padding: Union[int, List[int]] = 0,
                 use_grad: bool = True, 
                 device: Optional[str] = "cpu"):
        _init_conv2d_layer = self.c_lib.init_conv2d_layer
        _init_conv2d_layer.argtypes = [c_int, c_int, c_int, c_int,
                                       c_int, c_int, c_int, c_int,
                                       c_bool, c_int]
        _init_conv2d_layer.restype = C_NeuralLayer

        device = get_device_by_name(device)

        kernel_height, kernel_width = get_kernel_size(kernel)
        stride_height, stride_width = get_kernel_size(stride)
        padding_height, padding_width = get_kernel_size(padding)

        super().__init__(_init_conv2d_layer(in_channels, out_channels, 
                                            kernel_height, kernel_width, 
                                            stride_height, stride_width,
                                            padding_height, padding_width,
                                            use_grad, device))
