from ctypes import *

from ..common.forward_back import ForwardBackUnderlying
from ..tensor.tensor import C_Tensor, Tensor

LossFunc = CFUNCTYPE(c_float, POINTER(C_Tensor), POINTER(C_Tensor))

class C_Loss(Structure):
    _fields_ = [
        ("forward_cpu", LossFunc),
        ("forward_gpu", LossFunc),
        ("backward_cpu", LossFunc),
        ("backward_gpu", LossFunc),
        ("pred", POINTER(C_Tensor)),
        ("exp", POINTER(C_Tensor))
    ]


class Loss(ForwardBackUnderlying):
    def __init__(self, und: C_Loss):
        self._forward_loss = self.c_lib.forward_loss
        self._forward_loss.argtypes = [POINTER(C_Loss), POINTER(C_Tensor), POINTER(C_Tensor)]
        self._forward_loss.restype = c_float

        self._backward_loss = self.c_lib.backward_loss
        self._backward_loss.argtypes = [POINTER(C_Loss)]
        self._backward_loss.restype = POINTER(C_Tensor)

        super().__init__(und)

    def forward(self, pred: Tensor, exp: Tensor) -> float:
        return self._forward_loss(pointer(self._underlying()), pred._underlying(), exp._underlying())

    def backward(self) -> Tensor:
        return self._call_underlying(self._backward_loss(pointer(self._underlying())))
