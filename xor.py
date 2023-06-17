from mAI.activation import Sigmoid
from mAI.loss import MSELoss
from mAI.nn import LinearLayer
from mAI.optimizer import SGDOptimizer
from mAI.tensor import Tensor

NUM_INPUTS = 2
NUM_HIDDEN_NODES = 2
NUM_OUTPUTS = 1
NUM_TRAINING_SETS = 4
BATCH_SIZE = 4
NUM_BATCHES = NUM_TRAINING_SETS // BATCH_SIZE
EPOCHS = 15000
DEVICE = 'gpu'
LR = 0.1

def main():
    '''Main Code'''
    training_inputs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    training_outputs = [[0.0], [1.0], [1.0], [0.0]]

    inputs = Tensor(training_inputs, device=DEVICE)
    outputs = Tensor(training_outputs, device=DEVICE)

    activation = Sigmoid()

    hidden_layer = LinearLayer(NUM_INPUTS, NUM_HIDDEN_NODES, device=DEVICE)
    output_layer = LinearLayer(NUM_HIDDEN_NODES, NUM_OUTPUTS, device=DEVICE)

    loss = MSELoss()

    optim = SGDOptimizer([hidden_layer.params(), output_layer.params()], LR)

    for _ in range(EPOCHS):
        inputs.shuffle(outputs)

        for i in range(NUM_BATCHES):
            input = inputs[i * BATCH_SIZE:i + BATCH_SIZE]

            expected_output = outputs[i * BATCH_SIZE:i + BATCH_SIZE]

            optim.zero_grad()

            hidden_output = hidden_layer(input)
            hidden_output = activation(hidden_output)

            predicted_output = output_layer(hidden_output)
            predicted_output = activation(predicted_output)

            loss(predicted_output, expected_output)

            loss.backward()
            optim.step()

    hidden_layer.no_grad()
    output_layer.no_grad()

    for i in range(NUM_TRAINING_SETS):
        input = inputs[i:i + 1]

        print("Input", input.tolist())

        hidden_output = hidden_layer(input)
        hidden_output = activation(hidden_output)
        predicted_output = output_layer(hidden_output)
        predicted_output = activation(predicted_output)

        print("Predicted output", predicted_output.tolist())


if __name__ == "__main__":
    main()