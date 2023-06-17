from ctypes import *
from typing import List, Union

from ..common.common import Underlying
from ..common.forward_back import ForwardBackUnderlying
from ..tensor.tensor import C_Tensor

class C_NeuralParameters(Structure):
    pass

class C_NeuralLayer(Structure):
    pass

NeuralFunc = CFUNCTYPE(POINTER(C_Tensor), POINTER(C_NeuralLayer), POINTER(C_Tensor))
StepFunc = CFUNCTYPE(None, POINTER(C_NeuralParameters))
ArgFreeFunc = CFUNCTYPE(None, c_void_p)

C_NeuralParameters._fields_ = [
    ("layer", POINTER(C_Tensor)),
    ("bias", POINTER(C_Tensor)),
    ("weights", POINTER(C_Tensor)),
    ("args", c_void_p),
    ("arg_free_func", ArgFreeFunc),
    ("step_func", StepFunc),
    ("prev", POINTER(C_Tensor))
]

C_NeuralLayer._fields_ = [
    ("parameters", POINTER(C_NeuralParameters)),
    ("forward", NeuralFunc),
    ("backward", NeuralFunc),
    ("forward_app", NeuralFunc),
    ("backward_app", NeuralFunc)
]

def get_kernel_size(kernel: Union[int, List[int]]):
    if isinstance(kernel, int):
        kernel_height = kernel
        kernel_width = kernel
    elif isinstance(kernel, list) and len(kernel) == 2:
        assert isinstance(kernel[0], int) and isinstance(kernel[1], int)
        kernel_height = kernel[0]
        kernel_width = kernel[1]
    else:
        raise ArgumentError("Kernel must be either an int or a list of two ints")

    return kernel_height, kernel_width

class NeuralParameters(Underlying):
    def __init__(self, und: POINTER(C_NeuralParameters)):
        self._zero_param_grad = self.c_lib.zero_param_grad
        self._zero_param_grad.argtypes = [POINTER(C_NeuralParameters)]

        super().__init__(und)
    
    def zero_grad(self) -> None:
        self._zero_param_grad(self._underlying())


class NeuralLayer(ForwardBackUnderlying):
    def __init__(self, und: C_NeuralLayer):
        self._destroy_neural_layer = self.c_lib.destroy_neural_layer
        self._destroy_neural_layer.argtypes = [POINTER(C_NeuralLayer)]

        self._grad = self.c_lib.use_grad
        self._grad.argtypes = [POINTER(C_NeuralLayer)]

        self._no_grad = self.c_lib.no_use_grad
        self._no_grad.argtypes = [POINTER(C_NeuralLayer)]

        super().__init__(und, self.c_lib.forward_neural_layer, self.c_lib.backward_neural_layer, C_NeuralLayer)

    def params(self) -> NeuralParameters:
        return NeuralParameters(self._underlying().parameters)
    
    def grad(self):
        self._grad(pointer(self._underlying()))

    def no_grad(self):
        self._no_grad(pointer(self._underlying()))

    def __del__(self):
        if self.und != None:
            self._destroy_neural_layer(pointer(self._underlying()))
