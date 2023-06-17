from ctypes import *

from .common import Underlying
from ..tensor.tensor import C_Tensor, Tensor

class ForwardBackUnderlying(Underlying):
    def __init__(self, und: Structure, forward = None, backward = None, arg_type = None):
        self._forward = forward
        self._backward = backward

        if arg_type is not None:
            if forward is not None:
                self._forward.argtypes = [POINTER(arg_type), POINTER(C_Tensor)]
                self._forward.restype = POINTER(C_Tensor)
            if backward is not None:
                self._backward.argtypes = [POINTER(arg_type), POINTER(C_Tensor)]
                self._backward.restype = POINTER(C_Tensor)
        else:
            self._forward = None
            self._backward = None
    
        super().__init__(und)

    def _call_underlying(self, output: POINTER(C_Tensor)) -> Tensor:
        if not output:
            raise RuntimeError("Check that gradient is enabled or that the dimensions of the layer and input match")
        
        return Tensor(input_tensor=output)

    def forward(self, x: Tensor) -> Tensor:
        if self._forward:
            return self._call_underlying(self._forward(pointer(self._underlying()), x._underlying()))
        raise NotImplementedError

    def backward(self, x: Tensor) -> Tensor:
        if self._backward:
            return self._call_underlying(self._backward(pointer(self._underlying()), x._underlying()))
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
