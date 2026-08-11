import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, input_channel,num_channel,use_1conv=False,strides=1):
        super(Residual,self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=num_channel, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.bn2 = nn.BatchNorm2d(num_channel)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channel,out_channels=num_channel,kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):

        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y = self.ReLU(y+x)
        return y

class VIS_Encoder(nn.Module):
    def __init__(self,Residual):
        super(VIS_Encoder,self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1))

        self.b2 = nn.Sequential(
            Residual(64,64,use_1conv=False,strides=1),
            Residual(64,64,use_1conv=False,strides=1))

        self.b3 = nn.Sequential(
            Residual(64,128,use_1conv=True,strides=2),
            Residual(128,128,use_1conv=False,strides=1))

        self.b4 = nn.Sequential(
            Residual(128,256,use_1conv=True,strides=2),
            Residual(256,256,use_1conv=False,strides=1))

        self.b5 = nn.Sequential(
            Residual(256,512,use_1conv=True,strides=2),
            Residual(512,512,use_1conv=False,strides=1))

    def forward(self, x):
        features = []

        x = self.b1(x)
        features.append(x)
        x = self.b2(x)
        features.append(x)
        x = self.b3(x)
        features.append(x)
        x = self.b4(x)
        features.append(x)
        x = self.b5(x)
        features.append(x)

        return features

class IR_Encoder(nn.Module):
    def __init__(self,Residual):
        super(IR_Encoder,self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1))

        self.b2 = nn.Sequential(
            Residual(64,64,use_1conv=False,strides=1),
            Residual(64,64,use_1conv=False,strides=1))

        self.b3 = nn.Sequential(
            Residual(64,128,use_1conv=True,strides=2),
            Residual(128,128,use_1conv=False,strides=1))

        self.b4 = nn.Sequential(
            Residual(128,256,use_1conv=True,strides=2),
            Residual(256,256,use_1conv=False,strides=1))

        self.b5 = nn.Sequential(
            Residual(256,512,use_1conv=True,strides=2),
            Residual(512,512,use_1conv=False,strides=1))

    def forward(self, x):
        features = []

        x = self.b1(x)
        features.append(x)
        x = self.b2(x)
        features.append(x)
        x = self.b3(x)
        features.append(x)
        x = self.b4(x)
        features.append(x)
        x = self.b5(x)
        features.append(x)

        return features

if __name__ == "__main__":
    from torchsummary import summary

    model_VIS = VIS_Encoder(Residual)
    model_IR = IR_Encoder(Residual)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_VIS = model_VIS.to(device)
    model_IR = model_IR.to(device)

    print(summary(model_VIS,input_size=(3,224,224)))
    print(summary(model_IR,input_size=(1,224,224)))

