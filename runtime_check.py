from mAI.loss import MSELoss
from mAI.nn import Conv2DLayer, MaxPool2DLayer
from mAI.optimizer import SGDOptimizer
from mAI.tensor import tensor_random

BATCH_SIZE = 1
IN_CHANNELS = 1
OUT_CHANNELS = 3
INPUT_SIZE = 15
KERNEL_SIZE = 3
HIDDEN_SIZE = INPUT_SIZE - KERNEL_SIZE + 1
OUTPUT_SIZE = HIDDEN_SIZE - KERNEL_SIZE + 1
LR = 0.1
DEVICE = "gpu"

def main():
    dummy_input = tensor_random([BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE], device=DEVICE)
    dummy_out = tensor_random([BATCH_SIZE, OUT_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE], device=DEVICE)
    conv_layer = Conv2DLayer(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, device=DEVICE)
    pool_layer = MaxPool2DLayer(KERNEL_SIZE, device=DEVICE)

    loss = MSELoss()

    optim = SGDOptimizer([conv_layer.params(), pool_layer.params()], LR)

    optim.zero_grad()

    conv_output = conv_layer(dummy_input)

    pred_output = pool_layer(conv_output)

    loss(pred_output, dummy_out)

    loss.backward()
    optim.step()


if __name__ == "__main__":
    main()
