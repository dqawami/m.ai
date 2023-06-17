from mAI.activation import ReLU
from mAI.nn import Conv2DLayer, MaxPool2DLayer, FlattenLayer, LinearLayer

class ToyModel:
    def __init__(self, input_size, output_size=10, device="cpu"):
        self.conv1 = Conv2DLayer(input_size, 6, 5, device=device)
        self.pool1 = MaxPool2DLayer(2, device=device)
        self.conv2 = Conv2DLayer(6, 16, 5, device=device)
        self.pool2 = MaxPool2DLayer(2, device=device)
        self.fc1 = LinearLayer(16 * 22 * 22, 120, device=device)
        self.fc2 = LinearLayer(120, 84, device=device)
        self.fc3 = LinearLayer(84, output_size, device=device)
        self.relu = ReLU()
        self.flatten = FlattenLayer(1, device=device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def params(self):
        return [self.conv1.params(), self.conv2.params(), self.fc1.params(),
                self.fc2.params(), self.fc3.params(), self.pool1.params(),
                self.pool2.params(), self.flatten.params()]
    
    def grad(self):
        self.conv1.grad()
        self.pool1.grad()
        self.conv2.grad()
        self.pool2.grad()
        self.fc1.grad()
        self.fc2.grad()
        self.fc3.grad()
        self.flatten.grad()

    def no_grad(self):
        self.conv1.no_grad()
        self.pool1.no_grad()
        self.conv2.no_grad()
        self.pool2.no_grad()
        self.fc1.no_grad()
        self.fc2.no_grad()
        self.fc3.no_grad()
        self.flatten.no_grad()
