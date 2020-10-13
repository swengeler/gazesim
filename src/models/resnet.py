import torch
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary


class ResNet18BaseModel(nn.Module):

    def __init__(self, module_transfer_depth=7, transfer_weights=True):
        super(ResNet18BaseModel, self).__init__()

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


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    tensor = torch.zeros((1, 3, 300, 400)).to(device)

    net = ResNet18BaseModel().to(device)
    out = net(tensor)
    print(out.shape)

    # summary(net, (3, 300, 400))
