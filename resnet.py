from mAI.activation import ReLU
from mAI.nn import BatchNorm2DLayer, Conv2DLayer, FlattenLayer, LinearLayer, MaxPool2DLayer

# From https://github.com/dqawami/pytorch-cifar/blob/master/models/resnet.py

class BasicBlock:
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, device="cpu"):
        self.conv1 = Conv2DLayer(in_planes, planes, kernel=3, stride=stride,
                                 padding=1, device=device)
        self.bn1 = BatchNorm2DLayer(planes, device=device)
        self.conv2 = Conv2DLayer(planes, planes, kernel=3, padding=1, device=device)
        self.bn2 = BatchNorm2DLayer(planes, device=device)
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def params(self):
        return [self.conv1.params(), self.bn1.params(), self.conv2.params(),
                self.bn2.params()]
    
    def grad(self):
        self.conv1.grad()
        self.conv2.grad()
        self.bn1.grad()
        self.bn2.grad()

    def no_grad(self):
        self.conv1.no_grad()
        self.conv2.no_grad()
        self.bn1.no_grad()
        self.bn2.no_grad()

class ResNet:
    def __init__(self, block, num_blocks, num_classes=10, device="cpu"):
        self.in_planes = 64

        self.conv1 = Conv2DLayer(3, 64, kernel=3, device=device)
        self.bn1 = BatchNorm2DLayer(64, device=device)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, device=device)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, device=device)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, device=device)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, device=device)
        self.linear = LinearLayer(512 * block.expansion, num_classes, device=device)
        self.pool = MaxPool2DLayer(4, device=device)
        self.relu = ReLU()
        self.flatten = FlattenLayer(1, device=device)

    def _make_layer(self, block, planes, num_blocks, stride, device):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, device))
            self.in_planes = planes * block.expansion
        return layers
    
    def forward(self, x):
        def _forward(layer, x):
            for l in layer:
                x = l(x)
            return x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = _forward(self.layer1, x)
        x = _forward(self.layer2, x)
        x = _forward(self.layer3, x)
        x = _forward(self.layer4, x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def __call__(self, x):
        return self.forward(x)
    
    def params(self):
        def _params(layer):
            params = []
            for l in layer:
                params += l.params()
            return params

        params = [self.conv1.params(), self.bn1.params(), self.pool.params(), 
                  self.flatten.params(), self.linear.params()]
        params += _params(self.layer1)
        params += _params(self.layer2)
        params += _params(self.layer3)
        params += _params(self.layer4)
        return params
    
    def grad(self):
        def _grad(layer):
            for l in layer:
                l.grad()

        self.conv1.grad()
        self.bn1.grad()
        self.pool.grad()
        self.flatten.grad()
        self.linear.grad()
        _grad(self.layer1)
        _grad(self.layer2)
        _grad(self.layer3)
        _grad(self.layer4)

    def no_grad(self):
        def _no_grad(layer):
            for l in layer:
                l.no_grad()

        self.conv1.no_grad()
        self.bn1.no_grad()
        self.pool.no_grad()
        self.flatten.no_grad()
        self.linear.no_grad()
        _no_grad(self.layer1)
        _no_grad(self.layer2)
        _no_grad(self.layer3)
        _no_grad(self.layer4)

def ResNet18(device="cpu"):
    return ResNet(BasicBlock, [2, 2, 2, 2], device=device)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
