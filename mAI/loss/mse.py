from .loss import Loss, C_Loss

class MSELoss(Loss):
    def __init__(self):
        init_mse_loss = self.c_lib.init_mse_loss
        init_mse_loss.restype = C_Loss
        super().__init__(init_mse_loss())
