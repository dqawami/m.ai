from ctypes import *
from typing import List

from .optimizer import Optimizer, C_Optimizer
from ..nn.nn import NeuralParameters, C_NeuralParameters

class SGDOptimizer(Optimizer):
    def __init__(self, params: List[NeuralParameters], lr: float, 
                 momentum: float = 0.0, weight_decay: float = 0.0):
        num_params = len(params)

        c_params = []
        for param in params:
            c_params.append(param._underlying())

        c_params = (POINTER(C_NeuralParameters)  * num_params)(*c_params)

        _init_sgd_optimizer = self.c_lib.init_sgd_optimizer
        _init_sgd_optimizer.argtypes = [POINTER(POINTER(C_NeuralParameters)),
                                        c_int, c_float, c_float, c_float]
        _init_sgd_optimizer.restype = C_Optimizer

        super().__init__(_init_sgd_optimizer(c_params, num_params, lr,
                                             momentum, weight_decay))