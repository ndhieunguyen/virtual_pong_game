from torch import nn


class DeepQNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(36864, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, n_actions),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


# Test model
if __name__ == "__main__":
    import torch

    shape = (4, 3, 224, 224)
    n_actions = 6
    model = DeepQNet(input_shape=shape[1:], n_actions=n_actions)
    tensor = torch.zeros((shape))
    print(model(tensor))
