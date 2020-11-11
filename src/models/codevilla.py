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

        self.image_net_conv = nn.Sequential(
            self.image_conv_0,
            self.image_conv_1,
            self.image_conv_2,
            self.image_conv_3,
            self.image_conv_4,
            self.image_conv_5,
            self.image_conv_6,
            self.image_conv_7,
        )

        self.image_net_fc = nn.Sequential(
            self.image_fc_0,
            self.image_fc_1
        )

        # measurement/state network
        self.state_fc_0 = Codevilla.fc_block(len(config["drone_state_names"]), 128)
        self.state_fc_1 = Codevilla.fc_block(128, 128)

        self.state_net = nn.Sequential(
            self.state_fc_0,
            self.state_fc_1
        )

        # control network
        self.control_fc_0 = Codevilla.fc_block(512 + 128, 256)
        self.control_fc_1 = Codevilla.fc_block(256, 256)
        self.control_fc_2 = Codevilla.fc_block(256, 4)

        self.control_net = nn.Sequential(
            self.control_fc_0,
            self.control_fc_1,
            self.control_fc_2
        )

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
        image_x = self.image_net_conv(x["input_image_0"])
        image_x = image_x.reshape(image_x.size(0), -1)
        image_x = self.image_net_fc(image_x)

        state_x = self.state_net(x["input_state"])

        combined_x = torch.cat([image_x, state_x], dim=-1)

        control_x = self.control_net(combined_x)
        probabilities = self.final_activation(control_x)

        out = {"output_control": probabilities}
        return out


class CodevillaMultiHead(Codevilla):

    def __init__(self, config=None):
        super().__init__()

        # image network, convolutional layers
        self.image_net_conv = nn.Sequential(
            Codevilla.conv_block(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0),
            Codevilla.conv_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            Codevilla.conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
            Codevilla.conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            Codevilla.conv_block(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            Codevilla.conv_block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            Codevilla.conv_block(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0),
            Codevilla.conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
        )

        # image network, fully connected layers
        self.image_net_fc = nn.Sequential(
            Codevilla.fc_block(7168, 512),
            Codevilla.fc_block(512, 512)
        )

        # measurement/state network
        self.state_net = nn.Sequential(
            Codevilla.fc_block(len(config["drone_state_names"]), 128),
            Codevilla.fc_block(128, 128)
        )

        # control network input vector
        self.control_input_vector = Codevilla.fc_block(512 + 128, 512)

        # branches of control network
        # TODO: should maybe be a dictionary (depending on what the batch input will look like)
        self.branches = nn.ModuleList()
        for b in range(5):
            branch = nn.Sequential(
                Codevilla.fc_block(512, 256),
                Codevilla.fc_block(256, 256),
                Codevilla.fc_block(256, 4)
            )
            self.branches.append(branch)

        self.final_activation = ControlActivationLayer()

    def forward(self, x):
        image_x = self.image_net_conv(x["input_image_0"])
        image_x = image_x.reshape(image_x.size(0), -1)
        image_x = self.image_net_fc(image_x)

        state_x = self.state_net(x["input_state"])

        combined_x = torch.cat([image_x, state_x], dim=-1)
        combined_x = self.control_input_vector(combined_x)

        """
        dummy = torch.autograd.Variable(torch.ones_like(combined_x), requires_grad=True).to(combined_x.device)
        combined_x = dummy.clone()
        """

        # use different branches depending on label of each sample in batch
        samples = []
        for b_idx, branch in enumerate(self.branches):
            # get subset of batch where label matches branch index
            subset = (x["label_high_level"] == b_idx)

            # vector of all indices in the batch, reshaped to match the dimensions of the input vector
            index = torch.arange(end=combined_x.size(0)).to(combined_x.device)[subset]

            # select the correct samples
            branch_batch = torch.index_select(combined_x, 0, index)

            # pass them through the branch head
            branch_batch = branch(branch_batch)

            # put them in the samples list
            for s_idx, sample in zip(index, branch_batch):
                samples.append((int(s_idx), sample.unsqueeze(0)))

        # combine samples again
        samples = [sample[1] for sample in sorted(samples, key=lambda s: s[0])]
        combined_x = torch.cat(samples, 0)

        # final activation
        # probabilities = self.final_activation(combined_x)

        out = {"output_control": combined_x}
        return out


class CodevillaDualBranch(CodevillaMultiHead):

    def __init__(self, config=None):
        super().__init__(config)

        # image network, convolutional layers
        self.branch_0_image_net_conv = self.image_net_conv
        self.branch_1_image_net_conv = nn.Sequential(
            Codevilla.conv_block(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0),
            Codevilla.conv_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            Codevilla.conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
            Codevilla.conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            Codevilla.conv_block(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            Codevilla.conv_block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            Codevilla.conv_block(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0),
            Codevilla.conv_block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
        )

        # image network(s), fully connected layers
        self.image_net_fc = nn.Sequential(
            Codevilla.fc_block(7168 * 2, 512),
            Codevilla.fc_block(512, 512)
        )

        # measurement/state network
        self.state_net = nn.Sequential(
            Codevilla.fc_block(len(config["drone_state_names"]), 128),
            Codevilla.fc_block(128, 128)
        )

        # control network input vector
        self.control_input_vector = Codevilla.fc_block(512 + 128, 512)

    def forward(self, x):
        branch_0_image_x = self.branch_0_image_net_conv(x["input_image_0"])
        branch_0_image_x = branch_0_image_x.reshape(branch_0_image_x.size(0), -1)
        # branch_0_image_x = self.branch_0_image_net_fc(branch_0_image_x)

        branch_1_image_x = self.branch_1_image_net_conv(x["input_image_1"])
        branch_1_image_x = branch_1_image_x.reshape(branch_1_image_x.size(0), -1)
        # branch_1_image_x = self.branch_1_image_net_fc(branch_1_image_x)

        combined_x = torch.cat([branch_0_image_x, branch_1_image_x], dim=-1)
        combined_x = self.image_net_fc(combined_x)

        state_x = self.state_net(x["input_state"])

        combined_x = torch.cat([combined_x, state_x], dim=-1)
        combined_x = self.control_input_vector(combined_x)

        # use different branches depending on label of each sample in batch
        samples = []
        for b_idx, branch in enumerate(self.branches):
            # get subset of batch where label matches branch index
            subset = (x["label_high_level"] == b_idx)

            # vector of all indices in the batch, reshaped to match the dimensions of the input vector
            index = torch.arange(end=combined_x.size(0)).to(combined_x.device)[subset]

            # select the correct samples
            branch_batch = torch.index_select(combined_x, 0, index)

            # pass them through the branch head
            branch_batch = branch(branch_batch)

            # put them in the samples list
            for s_idx, sample in zip(index, branch_batch):
                samples.append((int(s_idx), sample.unsqueeze(0)))

        # combine samples again
        samples = [sample[1] for sample in sorted(samples, key=lambda s: s[0])]
        combined_x = torch.cat(samples, 0)

        # final activation
        # probabilities = self.final_activation(combined_x)

        out = {"output_control": combined_x}
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


class CodevillaSkip(Codevilla):
    """
    Based on "End-to-end Driving via Conditional Imitation Learning", specifically a TensorFlow implementation that
    can be found at: https://www.github.com/merantix/imitation-learning
    """

    def __init__(self, config=None):
        super().__init__()

    def forward(self, x):
        prev_image_x = x["input_image_0"]
        image_x = self.image_conv_0(x["input_image_0"])
        image_x = self.image_conv_1(image_x)
        image_x += prev_image_x

        prev_image_x = image_x
        image_x = self.image_conv_2(image_x)
        image_x = self.image_conv_3(image_x)
        image_x += prev_image_x

        prev_image_x = image_x
        image_x = self.image_conv_4(image_x)
        image_x = self.image_conv_5(image_x)
        image_x += prev_image_x

        prev_image_x = image_x
        image_x = self.image_conv_6(image_x)
        image_x = self.image_conv_7(image_x)
        image_x += prev_image_x

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

    image = torch.ones((2, 3, 150, 200)).to(device)
    state = torch.ones((2, 9)).to(device)
    label = torch.LongTensor([0, 1]).to(device)
    X = {
        "input_image_0": image,
        "input_image_1": image.clone(),
        "input_state": state,
        "label_high_level": label,
        "drone_state_names": ["?"] * 9
    }

    net = CodevillaDualBranch().to(device)
    result = net(X)
    print(result)
