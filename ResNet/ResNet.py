import torch
from torch import nn

class BottleNeck1(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels, stride=1, skip_connection = False):
        super().__init__()
        self.skip_connection = skip_connection
        self.model1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1, stride=1)
        self.model2 = torch.nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=3, stride=stride, padding=1)
        self.model3 = torch.nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):

        input_x = x
        x = self.model1(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.model2(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.model3(x)

        # print(x.size(), input_x.size())
        if self.skip_connection == True:
            input_x = nn.Conv2d(in_channels=input_x.size(1), out_channels=x.size(1), kernel_size=1, stride=1)(input_x)

        x = input_x + x
        # x = nn.BatchNorm2d(x.size(1))
        x = nn.ReLU(inplace=True)(x)
        return x


class BottleNeck2(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels, stride=1, skip_connection = False):
        super().__init__()
        self.skip_connection = skip_connection
        self.model1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1, stride=1)
        self.model2 = torch.nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=3, stride=stride, padding=1)
        self.model3 = torch.nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):

        input_x = x
        x = self.model1(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.model2(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.model3(x)

        if self.skip_connection:

            input_x = nn.Conv2d(in_channels=input_x.size(1), out_channels=x.size(1), kernel_size=1, stride=2)(input_x)
        # print(x.size(), input_x.size())
        x = x + input_x
        # x = nn.BatchNorm2d(x.size(1))

        x = nn.ReLU(inplace=True)(x)
        return x



class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = nn.Sequential(BottleNeck1(in_channels=64, middle_channels = 64, out_channels=256, skip_connection = True),
                                     BottleNeck1(in_channels=256, middle_channels = 64,out_channels=256, skip_connection = True),
                                     BottleNeck1(in_channels=256, middle_channels = 64,out_channels=256, skip_connection = True))

        self.conv3_x = nn.Sequential(BottleNeck2(in_channels=256, middle_channels = 128, out_channels=512, stride=2, skip_connection=True),
                                     BottleNeck1(in_channels=512, middle_channels = 128, out_channels=512, skip_connection = True),
                                     BottleNeck1(in_channels=512, middle_channels = 128, out_channels=512, skip_connection = True),
                                     BottleNeck1(in_channels=512, middle_channels = 128, out_channels=512, skip_connection = True))
        self.conv4_x = nn.Sequential(BottleNeck2(in_channels=512, middle_channels = 256, out_channels=1024, stride=2, skip_connection=True),
                                     BottleNeck1(in_channels=1024, middle_channels = 256, out_channels=1024, skip_connection = True),
                                     BottleNeck1(in_channels=1024, middle_channels = 256, out_channels=1024, skip_connection = True),
                                     BottleNeck1(in_channels=1024, middle_channels = 256, out_channels=1024, skip_connection = True),
                                     BottleNeck1(in_channels=1024, middle_channels = 256, out_channels=1024, skip_connection = True),
                                     BottleNeck1(in_channels=1024, middle_channels = 256, out_channels=1024, skip_connection = True))
        self.conv5_x = nn.Sequential(BottleNeck2(in_channels=1024, middle_channels = 512, out_channels=2048, stride=2, skip_connection=True),
                                     BottleNeck1(in_channels=2048, middle_channels = 512, out_channels=2048, skip_connection = True),
                                     BottleNeck1(in_channels=2048, middle_channels = 512, out_channels=2048, skip_connection = True))
        self.Avgpool1 = nn.AvgPool2d(kernel_size=7, stride=1)


        self.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        # print(x.size())
        x = self.Maxpool1(x)
        # print(x.size())
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.Avgpool1(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        print(x)
        x = nn.Softmax()(x)

        return x


x = torch.randn((10, 3, 224, 224))


model = ResNet()
print(x, torch.max(model(x), 1)[1])
print(model(x))

