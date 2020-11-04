import torch
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary
from src.models.layers import ControlActivationLayer


class ResNet18BaseModel(nn.Module):

    def __init__(self, module_transfer_depth=7, transfer_weights=True):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300
        # shape will be [-1, 256, 10, 13] after this with module_transfer_depth 7 and input height 150

        # defining the upscaling layers to get out the original image size again
        self.upscaling = nn.Sequential(
            nn.Upsample(size=(37, 50)),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(75, 100)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(150, 200)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(300, 400)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upscaling(x)
        return x


class ResNet18BaseModelSimple(nn.Module):

    def __init__(self, module_transfer_depth=6, transfer_weights=True):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 128, 38, 50] after this with module_transfer_depth 6 and input height 300

        # defining the upscaling layers to get out the original image size again
        self.upscaling = nn.Sequential(
            # nn.Upsample(size=(37, 50)),
            # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            nn.Upsample(size=(75, 100)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(150, 200)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(300, 400)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upscaling(x)
        return x


class ResNet18Regressor(nn.Module):

    def __init__(self, module_transfer_depth=7, transfer_weights=True):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # pooling layer, using the same as ResNet for now
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        # defining the upscaling layers to get out the original image size again
        self.regressor = nn.Sequential(
            nn.Linear(256, 4),
            ControlActivationLayer()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.not_sure(x)
        # print(x.shape)
        x = self.pooling(x)
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.regressor(x)
        return x


class ResNet18SimpleRegressor(nn.Module):

    def __init__(self, module_transfer_depth=6, transfer_weights=True):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 128, 38, 50] after this with module_transfer_depth 6 and input height 300

        # pooling layer, using the same as ResNet for now
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        # defining the upscaling layers to get out the original image size again
        self.regressor = nn.Sequential(
            nn.Linear(128, 4),
            ControlActivationLayer()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = x.reshape(x.size(0), -1)
        x = self.regressor(x)
        return x


class ResNet18DualBranchRegressor(nn.Module):

    def __init__(self, module_transfer_depth=6, transfer_weights=True):
        super().__init__()

        # get the layers form ResNet
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        # creating the feature extractors/two branches
        self.features_0 = nn.Sequential(*modules)
        self.features_1 = nn.Sequential(*modules)

        # pooling layer, using the same as ResNet for now
        # should be able to apply this to features from both branches
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # regressor with double the input size as simple ResNet regressor
        self.regressor = nn.Sequential(
            nn.Linear(256, 4),
            ControlActivationLayer()
        )

    def forward(self, x):
        # pass both images through the branches
        x_0 = self.features_0(x[0])
        x_1 = self.features_1(x[1])

        # pooling
        x_0 = self.pooling(x_0)
        x_1 = self.pooling(x_1)

        # flatten the pooled features
        x_0 = x_0.reshape(x_0.size(0), -1)
        x_1 = x_1.reshape(x_1.size(0), -1)

        # concatenate the features and pass them through the regressor
        x = torch.cat((x_0, x_1), 1)
        x = self.regressor(x)

        return x


class ResNetStateRegressor(nn.Module):

    def __init__(self, config=None):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(True)
        modules = list(resnet18.children())[:7]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # pooling layer, using the same as ResNet for now
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # state "transformation" layer
        self.state = nn.Linear(9, 256)

        # defining the upscaling layers to get out the original image size again
        self.regressor = nn.Sequential(
            nn.Linear(512, 4),
            ControlActivationLayer()
        )

    def forward(self, x):
        image_x = self.features(x["input_image_0"])
        image_x = self.pooling(image_x)
        image_x = image_x.reshape(image_x.size(0), -1)

        state_x = self.state(x["input_state"])

        combined_x = torch.cat([image_x, state_x], dim=-1)

        probabilities = self.regressor(combined_x)

        out = {"output_control": probabilities}
        return out


class ResNetRegressor(nn.Module):

    def __init__(self, config=None):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(True)
        modules = list(resnet18.children())[:7]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # pooling layer, using the same as ResNet for now
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # defining the upscaling layers to get out the original image size again
        self.regressor = nn.Sequential(
            nn.Linear(256, 4),
            ControlActivationLayer()
        )

    def forward(self, x):
        image_x = self.features(x["input_image_0"])
        image_x = self.pooling(image_x)
        image_x = image_x.reshape(image_x.size(0), -1)

        probabilities = self.regressor(image_x)

        out = {"output_control": probabilities}
        return out


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # tensor = torch.zeros((1, 3, 300, 400)).to(device)
    tensor = [torch.zeros((1, 3, 300, 400)).to(device), torch.zeros((1, 3, 300, 400)).to(device)]

    net = ResNet18DualBranchRegressor().to(device)
    out = net(tensor)
    print(out.shape)

    # summary(net, (3, 300, 400))
