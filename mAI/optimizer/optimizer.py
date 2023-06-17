from ctypes import *

from ..common.common import Underlying
from ..nn.nn import C_NeuralParameters, ArgFreeFunc
from ..tensor.tensor import C_Tensor

class C_Optimizer(Structure):
    pass

ParamUpdateFunc = CFUNCTYPE(None, POINTER(C_NeuralParameters),
                            POINTER(C_Optimizer))
    
C_Optimizer._fields_ = [
    ("params", POINTER(POINTER(C_NeuralParameters))),
    ("num_params", c_int),
    ("lr", c_float),
    ("args", c_void_p),
    ("param_update_cpu", ParamUpdateFunc),
    ("param_update_gpu", ParamUpdateFunc),
    ("arg_free_func", ArgFreeFunc)
]

class Optimizer(Underlying):
    def __init__(self, und: C_Optimizer):
        self._zero_grad = self.c_lib.zero_grad
        self._zero_grad.argtypes = [POINTER(C_Optimizer)]
        self._zero_grad.restype = c_int

        self._optimizer_step = self.c_lib.optimizer_step
        self._optimizer_step.argtypes = [POINTER(C_Optimizer)]

        self._destroy_optimizer = self.c_lib.destroy_optimizer
        self._destroy_optimizer.argtypes = [POINTER(C_Optimizer)]

        super().__init__(und)

    def __del__(self):
        self._destroy_optimizer(self.und)

    def zero_grad(self) -> None:
        if self._zero_grad(pointer(self._underlying())):
            raise RuntimeError("Parameter is missing, either assigned a missing parameter or layer was deleted")

    def step(self) -> None:
        self._optimizer_step(pointer(self._underlying()))
