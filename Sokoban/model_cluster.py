import torch
import torch.nn as nn
import numpy as np


class HeuristicNetworkPHS(nn.Module):
    def __init__(self, size, channels, kernel_size=2, filters=32, number_actions=4):
        super(HeuristicNetworkPHS, self).__init__()
        self.size = size
        self.channels = channels
        self.kernel_size = kernel_size
        self.filters = filters
        self.num_actions = number_actions
        self.conv1 = nn.Conv2d(self.channels, self.filters, self.kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.size = self.size - self.kernel_size + 1
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.size = np.floor(self.size / 2 - 1) + 1
        self.conv2 = nn.Conv2d(self.filters, self.filters, self.kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.size = self.size - self.kernel_size + 1
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.size = np.floor(self.size / 2 - 1) + 1
        self.dense1 = nn.Linear(self.size * self.size * self.channels, 128)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pooling1(nn.ReLU(self.conv1(x)))
        x = self.pooling2(nn.ReLU(self.conv2(x)))
        x = x.view(-1, self.size * self.size * self.channels)
        x = nn.ReLU(self.dense1(x))
        x = self.dense2(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channel=32):
        super(ResBlock, self).__init__()
        self.channel = channel
        self.block = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU(),
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNet(nn.Module):
    def __init__(self, channel, numBlock):
        super(ResNet, self).__init__()
        self.blocks = []
        self.channel = channel
        self.numBlock = numBlock
        for _ in range(self.numBlock):
            self.blocks.append(ResBlock(channel=self.channel))
        self.blocks = nn.ModuleList(self.blocks)
        self.resnet = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.resnet(x)


class PVNetwork(nn.Module):
    def __init__(self, input_size, actions, in_channel=4, channel=64, numBlock=3):
        super(PVNetwork, self). __init__()
        self.input_size = input_size
        self.actions = actions
        self.channel = channel
        self.in_channel = in_channel
        self.numBlock = numBlock
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )
        self.resnet = ResNet(channel=self.channel, numBlock=self.numBlock)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.policy_fc = nn.Sequential(
            nn.Linear(self.input_size * self.input_size, self.actions),
            nn.LogSoftmax(dim=1)
        )
        self.value_fc = nn.Sequential(
            nn.Linear(self.input_size * self.input_size, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.head(x)
        feature = self.resnet(x)
        policy = self.policy_conv(feature)
        policy = policy.view(-1, self.input_size * self.input_size)
        policy = self.policy_fc(policy)
        value = self.value_conv(feature)
        feature = value.view(-1, self.input_size * self.input_size)
        value = self.value_fc(feature)
        return policy, value, torch.sigmoid(feature)


class PVModel():
    def __init__(self, input_size, actions, in_channel, model_path=None, device='cuda:0'):
        self.model = PVNetwork(input_size, actions, in_channel)
        self.model = self.model.to(device)
        self.device = device
        if model_path is not None:
            parameters = torch.load(model_path, map_location={'cuda:1': self.device})
            self.model.load_state_dict(parameters)
        self.model.eval()

    def predict(self, states, policy=False):
        states = torch.FloatTensor(np.array(states, dtype=np.float32)).to(self.device)
        log_probs, values, feature = self.model(states)
        probs = np.exp(log_probs.data.cpu().numpy())
        values = values.data.cpu().numpy()
        feature = feature.data.cpu().numpy()
        if policy:
            return probs, values, feature
        else:
            return values, feature