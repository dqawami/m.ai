from ctypes import *
from ..common.forward_back import ForwardBackUnderlying
from ..tensor.tensor import C_Tensor

ActivationFuncCPU = CFUNCTYPE(c_float, c_float)
ActivationFuncGPU = CFUNCTYPE(None, POINTER(C_Tensor))

class C_Activation(Structure):
    _fields_ = [
        ("forward_cpu", ActivationFuncCPU),
        ("backward_cpu", ActivationFuncCPU),
        ("forward_gpu", ActivationFuncGPU),
        ("backward_gpu", ActivationFuncGPU)
    ]


class Activation(ForwardBackUnderlying):
    def __init__(self, und: C_Activation):
        super().__init__(und, self.c_lib.forward_act, self.c_lib.backward_act,
                         C_Activation)
