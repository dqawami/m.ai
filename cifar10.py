from mAI.loss import CrossEntropyLoss, MSELoss
from mAI.optimizer import SGDOptimizer
from mAI.tensor import Tensor

from resnet import ResNet18
from toy_model import ToyModel

import pickle

import numpy as np

IN_CHANNELS = 3
INPUT_SIZE = 32
OUTPUT_SIZE = 10
KERNEL_SIZE = 3

BATCH_SIZE = 1
LR = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 50
LR_EPOCH_CHANGE = 20
LOSS_CHECK = 10000 // BATCH_SIZE
TEST_LOSS_CHECK = 2000

DEVICE = "gpu"

def get_cifar10_dataset(filenames):
    out_dict = {'data' : [], 'labels' : []}
    for file in filenames:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        data = dict[b'data']

        data = data / 255 * 2 - 1

        batch_size = data.shape[0]

        out_dict['data'] += data.reshape([batch_size, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE]).tolist()
        out_dict['labels'] += [[int(i == label) for i in range(OUTPUT_SIZE)] for label in dict[b'labels']]
    
    return out_dict

def main():
    folder = "./data/cifar-10-batches-py/"
    files = []
    for i in range(1, 6):
        files.append(folder + "data_batch_" + str(i))

    data = get_cifar10_dataset(files)

    test_data = get_cifar10_dataset([folder + 'test_batch'])

    inputs = Tensor(data['data'], device=DEVICE)
    exp_outputs = Tensor(data['labels'], device=DEVICE)

    test_inputs = Tensor(test_data['data'], device=DEVICE)
    test_exp = [tmp.index(max(tmp)) for tmp in test_data['labels']]

    model = ToyModel(IN_CHANNELS, device=DEVICE)
    # model = ResNet18(device=DEVICE)

    loss = CrossEntropyLoss()
    optim = SGDOptimizer(model.params(), LR, MOMENTUM, WEIGHT_DECAY)

    num_batches = inputs.get_dims()[0] // BATCH_SIZE
    test_size = len(test_exp)

    for e in range(EPOCHS):
        running_loss = 0
        inputs.shuffle(exp_outputs)
        model.grad()

        if (e % LR_EPOCH_CHANGE == LR_EPOCH_CHANGE - 1):
            lr /= 10

        for i in range(num_batches):
            batch_start = i * BATCH_SIZE
            batch_end = batch_start + BATCH_SIZE
            input = inputs[batch_start:batch_end]

            exp_out = exp_outputs[batch_start:batch_end]

            optim.zero_grad()

            pred_out = model(input)

            running_loss += loss(pred_out, exp_out)

            loss.backward()

            optim.step()

            if (i % LOSS_CHECK == LOSS_CHECK - 1):
                print("pred", pred_out.tolist())
                print("exp", exp_out.tolist())
                print(f'[{e + 1}, {i + 1:5d}] loss: {running_loss / LOSS_CHECK:.3f}')
                running_loss = 0.0

        model.no_grad()
        acc = 0
        for i in range(test_size):
            input = test_inputs[i:i + 1]

            pred_out = model(input).tolist()[0]
            pred_class = pred_out.index(max(pred_out))
            acc += int(test_exp[i] == pred_class)

            if (i % TEST_LOSS_CHECK == TEST_LOSS_CHECK - 1):
                print("pred", pred_out)
                print("exp", test_exp[i])

        print(f"Epoch {e + 1} accuracy {acc / test_size}")


if __name__ == "__main__":
    main()
