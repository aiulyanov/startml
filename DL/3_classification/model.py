import torch
import torch.nn as nn
import torchvision.models as models


class FCNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """
        Fully Connected Neural Network (FCNN) model.

        Parameters:
        - input_size (int): Number of input features.
        - output_size (int): Number of output classes.
        """
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x


class SimpleCNN(nn.Module):
    def __init__(self, output_size: int):
        """
        CNN model for image classification.

        Parameters:
        - output_size (int): Number of output classes.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))                   # 16 x 16
        x = self.pool(self.relu(self.bn2(self.conv2(x))))                   # 8 x 8 

        x = x.view(x.size(0), -1)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x


class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Residual Block for ResNet."""
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: nn.Module, num_blocks: list, num_classes: int = 10) -> None:
        """ResNet CNN model for image classification."""
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block: nn.Module, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

def create_simple_conv_cifar():
    model = SimpleCNN(10)
    return model


def create_resnet18():
    return ResNet(ResBlock, [2, 2, 2, 2])


def get_pretrained_model(model_name: str, num_classes: int, pretrained: bool=True):
    weights = 'DEFAULT' if pretrained else None
    if model_name == 'alexnet':
        model = models.alexnet(weights=weights)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == 'vgg11':
        model = models.vgg11(weights=weights)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == 'googlenet':
        model = models.googlenet(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        if model.aux_logits:
            num_features_aux1 = model.aux1.fc2.in_features
            model.aux1.fc2 = nn.Linear(num_features_aux1, num_classes)
            num_features_aux2 = model.aux2.fc2.in_features
            model.aux2.fc2 = nn.Linear(num_features_aux2, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError("Invalid model name. Choose from 'alexnet', 'vgg11', 'googlenet', 'resnet18'.")
    
    return model
