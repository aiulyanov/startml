import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, output_size: int):
        """
        CNN model for image classification.

        Parameters:
        - output_size (int): Number of output classes.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)                   # 28 x 28
        x = self.relu(x)
        x = self.pool1(x)                   # 14 x 14

        x = self.conv2(x)                   # 14 x 14
        x = self.relu(x)
        x = self.pool2(x)                   # 7 x 7

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


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
