import torch
import torch.nn as nn

from src.models.layers import ControlActivationLayer


class Codevilla(nn.Module):
    """
    Based on "End-to-end Driving via Conditional Imitation Learning", specifically a TensorFlow implementation that
    can be found at: https://www.github.com/merantix/imitation-learning
    """

    def __init__(self, config=None):
        super().__init__()

        # image network, convolutional layers
        self.image_conv_0 = Codevilla.conv_block(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.image_conv_1 = Codevilla.conv_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.image_conv_2 = Codevilla.conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.image_conv_3 = Codevilla.conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.image_conv_4 = Codevilla.conv_block(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.image_conv_5 = Codevilla.conv_block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)

        self.image_conv_6 = Codevilla.conv_block(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.image_conv_7 = Codevilla.conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)

        # image network, fully connected layers
        self.image_fc_0 = Codevilla.fc_block(7168, 512)
        self.image_fc_1 = Codevilla.fc_block(512, 512)

        # measurement/state network
        self.state_fc_0 = Codevilla.fc_block(9, 128)
        self.state_fc_1 = Codevilla.fc_block(128, 128)

        # control network
        self.control_fc_0 = Codevilla.fc_block(512 + 128, 256)
        self.control_fc_1 = Codevilla.fc_block(256, 256)
        self.control_fc_2 = Codevilla.fc_block(256, 4)

        self.final_activation = ControlActivationLayer()

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        batch_norm = nn.BatchNorm2d(out_channels)
        dropout = nn.Dropout(0.2)
        activation = nn.ReLU()
        return nn.Sequential(conv, batch_norm, dropout, activation)

    @staticmethod
    def fc_block(in_features, out_features):
        fc = nn.Linear(in_features, out_features)
        dropout = nn.Dropout(0.5)
        activation = nn.ReLU()
        return nn.Sequential(fc, dropout, activation)

    def forward(self, x):
        image_x = self.image_conv_0(x["input_image_0"])
        image_x = self.image_conv_1(image_x)
        image_x = self.image_conv_2(image_x)
        image_x = self.image_conv_3(image_x)
        image_x = self.image_conv_4(image_x)
        image_x = self.image_conv_5(image_x)
        image_x = self.image_conv_6(image_x)
        image_x = self.image_conv_7(image_x)
        image_x = image_x.reshape(image_x.size(0), -1)

        image_x = self.image_fc_0(image_x)
        image_x = self.image_fc_1(image_x)

        state_x = self.state_fc_0(x["input_state"])
        state_x = self.state_fc_1(state_x)

        combined_x = torch.cat([image_x, state_x], dim=-1)

        control_x = self.control_fc_0(combined_x)
        control_x = self.control_fc_1(control_x)
        logits = self.control_fc_2(control_x)
        probabilities = self.final_activation(logits)

        out = {"output_control": probabilities}
        return out


class Codevilla300(Codevilla):

    def __init__(self, config=None):
        super().__init__(config)

        self.image_conv_8 = Codevilla.conv_block(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.image_conv_9 = Codevilla.conv_block(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)

        self.image_fc_0 = Codevilla.fc_block(14336, 512)

    def forward(self, x):
        image_x = self.image_conv_0(x["input_image_0"])
        image_x = self.image_conv_1(image_x)
        image_x = self.image_conv_2(image_x)
        image_x = self.image_conv_3(image_x)
        image_x = self.image_conv_4(image_x)
        image_x = self.image_conv_5(image_x)
        image_x = self.image_conv_6(image_x)
        image_x = self.image_conv_7(image_x)
        image_x = self.image_conv_8(image_x)
        image_x = self.image_conv_9(image_x)
        image_x = image_x.reshape(image_x.size(0), -1)

        image_x = self.image_fc_0(image_x)
        image_x = self.image_fc_1(image_x)

        state_x = self.state_fc_0(x["input_state"])
        state_x = self.state_fc_1(state_x)

        combined_x = torch.cat([image_x, state_x], dim=-1)

        control_x = self.control_fc_0(combined_x)
        control_x = self.control_fc_1(control_x)
        logits = self.control_fc_2(control_x)
        probabilities = self.final_activation(logits)

        out = {"output_control": probabilities}
        return out


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    image = torch.zeros((1, 3, 300, 400)).to(device)
    state = torch.zeros((1, 9)).to(device)
    X = {
        "input_image_0": image,
        "input_state": state
    }

    net = Codevilla300().to(device)
    result = net(X)
    print(result)
