from .activation import Activation, C_Activation

class Sigmoid(Activation):
    def __init__(self):
        init_sigmoid = self.c_lib.init_sigmoid
        init_sigmoid.restype = C_Activation

        und = init_sigmoid()

        super().__init__(und)
