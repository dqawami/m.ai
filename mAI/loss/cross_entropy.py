from .loss import Loss, C_Loss

class CrossEntropyLoss(Loss):
    def __init__(self):
        init_cross_entropy_loss = self.c_lib.init_cross_entropy_loss
        init_cross_entropy_loss.restype = C_Loss
        super().__init__(init_cross_entropy_loss())
