from torch import nn
from Tools.utils import get_out_layer


class AlexNet(nn.Module):
    def __init__(self, classes, shape=(224, 224)):
        super().__init__()
        self.name = 'AlexNet'

        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                   nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                   nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2))

        out = get_out_layer(nn.Sequential(self.conv1, self.conv2, self.conv3), 3, shape)

        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(256 * out, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                nn.Linear(4096, classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)
