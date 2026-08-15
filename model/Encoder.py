import torch
from torch import nn


IR_MODES = ("gray", "learned_gray")


class GrayEnhancer(nn.Module):
    """对单通道灰度 IR 做有界、恒等初始化的可学习增强。"""

    def __init__(self, branch_channels=8):
        super(GrayEnhancer, self).__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=branch_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(4, branch_channels),
                    nn.SiLU(),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        feature_channels = branch_channels * len(self.branches)
        self.residual_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.gate_head = nn.Conv2d(feature_channels, 1, kernel_size=1)

        # residual=0 使增强器初始时严格退化为恒等映射；gate=0.5 保持中性。
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(self, gray):
        if gray.ndim != 4 or gray.shape[1] != 1:
            raise ValueError(
                f"GrayEnhancer 输入必须是 [B,1,H,W]，实际为 {tuple(gray.shape)}"
            )
        features = torch.cat([branch(gray) for branch in self.branches], dim=1)
        residual = torch.tanh(self.residual_head(features))
        gate = torch.sigmoid(self.gate_head(features))
        # 对 gray∈[0,1]，gray + r*gray*(1-gray) 在 r∈[-1,1] 时仍位于 [0,1]。
        return gray + gate * residual * gray * (1.0 - gray)


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
    def __init__(self,Residual,ir_mode="gray"):
        super(IR_Encoder,self).__init__()
        if ir_mode not in IR_MODES:
            raise ValueError(f"不支持的 IR 模式: {ir_mode!r}，可选值为 {IR_MODES}")
        self.ir_mode = ir_mode
        self.enhancer = GrayEnhancer() if ir_mode == "learned_gray" else nn.Identity()
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

        x = self.enhancer(x)
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

