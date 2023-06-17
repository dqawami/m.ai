from .activation import Activation, C_Activation

class ReLU(Activation):
    def __init__(self):
        init_relu = self.c_lib.init_relu
        init_relu.restype = C_Activation

        und = init_relu()

        super().__init__(und)
