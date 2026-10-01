import torch.nn as nn


class DNN_DAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=16, padding='same'), nn.ReLU(inplace=True))
        self.pool1 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=16, stride=4, padding=6), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=16, padding='same'), nn.ReLU(inplace=True))
        self.pool2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=16, stride=4, padding=6), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose1d(64, 64, kernel_size=16, stride=4, padding=6)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=16, padding='same'), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose1d(32, 32, kernel_size=16, stride=4, padding=6)
        self.output = nn.Conv1d(32, 1, kernel_size=16, padding='same')

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.up1(x)
        x = self.conv3(x)
        x = self.up2(x)
        return self.output(x)
