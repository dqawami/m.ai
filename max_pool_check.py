from mAI.loss import MSELoss
from mAI.nn import MaxPool2DLayer
from mAI.optimizer import SGDOptimizer
from mAI.tensor import tensor_random

BATCH_SIZE = 1
CHANNEL = 1
INPUT_SIZE = 4
KERNEL_SIZE = 2
HIDDEN_SIZE = INPUT_SIZE - KERNEL_SIZE + 1
OUTPUT_SIZE = HIDDEN_SIZE - KERNEL_SIZE + 1
LR = 0.1
DEVICE = "gpu"

def main():
    dummy_input = tensor_random([BATCH_SIZE, CHANNEL, INPUT_SIZE, INPUT_SIZE], device=DEVICE)
    dummy_out = tensor_random([BATCH_SIZE, CHANNEL, OUTPUT_SIZE, OUTPUT_SIZE], device=DEVICE)

    pool1 = MaxPool2DLayer(KERNEL_SIZE, device=DEVICE)
    pool2 = MaxPool2DLayer(KERNEL_SIZE, device=DEVICE)

    loss = MSELoss()

    optim = SGDOptimizer([pool1.params(), pool2.params()], LR)

    optim.zero_grad()

    print(dummy_input.tolist())

    hidden_out = pool1(dummy_input)

    print(hidden_out.tolist())

    pred_out = pool2(hidden_out)

    print(pred_out.tolist())

    loss(pred_out, dummy_out)

    loss.backward()
    optim.step()


if __name__ == "__main__":
    main()
