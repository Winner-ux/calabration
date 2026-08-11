import torch
from torch import nn


class DecoderBlock(nn.Module):
    """
    解码器基础块: 转置卷积上采样 + 可选跳跃连接。

    无跳跃连接 (skip_channels=0):
        x -> ConvTranspose2d -> BN -> ReLU

    有跳跃连接 (skip_channels>0):
        x -> ConvTranspose2d -> Concat(skip) -> 1x1Conv降维 -> BN -> ReLU

    参数:
        in_channels:   转置卷积的输入通道数
        out_channels:  转置卷积的输出通道数
        kernel_size:   转置卷积核大小 (通常 4)
        stride:        上采样倍率 (通常 2)
        skip_channels: 跳跃连接特征的通道数, 0 表示无跳跃连接
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, skip_channels=0):
        super(DecoderBlock, self).__init__()

        # ==================================================
        # 转置卷积: 将特征图尺寸放大
        # 输出尺寸公式:
        #   H_out = (H_in - 1) * stride - 2 * padding + kernel_size
        # 当 kernel=4, stride=2, padding=1 时: H_out = 2 * H_in
        # ==================================================
        self.ct = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=1)

        # ==================================================
        # 跳跃连接处理
        # 如果有跳跃连接: 拼接后通道数 = out_channels + skip_channels
        # 需要额外 1x1 卷积将拼接结果降维回 out_channels
        # ==================================================
        if skip_channels > 0:
            concat_channels = out_channels + skip_channels
            self.skip_conv = nn.Conv2d(concat_channels, out_channels,kernel_size=1, stride=1)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            # 无跳跃连接: 使用普通的 BN
            self.skip_conv = None
            self.bn = nn.BatchNorm2d(out_channels)

        self.ReLU = nn.ReLU()

    def forward(self, x, skip=None):
        """
        x:    (B, in_channels, H_in, W_in)  -- 上一层解码器的输出
        skip: (B, skip_channels, H_out, W_out) 或 None -- 编码器的同分辨率特征

        返回: (B, out_channels, H_out, W_out)
        """
        # 第1步: 转置卷积上采样
        x = self.ct(x)

        # 第2步: 跳跃连接融合 (如果有)
        if skip is not None and self.skip_conv is not None:
            # 将解码器上采样后的特征 + 编码器同层特征在通道维拼接
            x = torch.cat([x, skip], dim=1)
            # 1x1 卷积融合降维
            x = self.skip_conv(x)
            x = self.skip_bn(x)
        else:
            x = self.bn(x)

        # 第3步: ReLU 激活
        x = self.ReLU(x)

        return x


class Decoder(nn.Module):
    """
    图像重建解码器, 带跳跃连接。

    上采样路径与跳跃连接映射:
        decoder1: (512, 7x7)  -> (256, 14x14)   skip <- ff4 (256ch, 14x14)
        decoder2: (256, 14x14)-> (128, 28x28)   skip <- ff3 (128ch, 28x28)
        decoder3: (128, 28x28)-> (64,  56x56)   skip <- ff2 (64ch,  56x56)
        decoder4: (64,  56x56)-> (32, 112x112)  无跳跃连接
        decoder5: (32, 112x112)->(16, 224x224)  无跳跃连接
        decoder6: (16, 224x224)->(3,  224x224)  输出 RGB
        sigmoid:  归一化到 [0, 1]

    说明:
        b1 (64, 56x56) 不用于跳跃连接, 因为 decoder4 上采样后是 112x112,
        空间尺寸不匹配。b1 的信息通过深层特征 (b2->b5) 间接传递到解码器。

    输入:
        x:              (B, 512, 7, 7)   -- 融合后的最深特征 (ff5)
        fused_features: list[5]           -- 5层融合特征 [ff1,ff2,ff3,ff4,ff5]

    输出:
        (B, 3, 224, 224) -- 融合图像
    """
    def __init__(self, DecoderBlock):
        super(Decoder, self).__init__()

        # decoder1: 512->256, 7->14, skip来自ff4(256ch)
        self.decoder1 = DecoderBlock(512, 256, 4, 2, skip_channels=256)

        # decoder2: 256->128, 14->28, skip来自ff3(128ch)
        self.decoder2 = DecoderBlock(256, 128, 4, 2, skip_channels=128)

        # decoder3: 128->64, 28->56, skip来自ff2(64ch)
        self.decoder3 = DecoderBlock(128, 64, 4, 2, skip_channels=64)

        # decoder4: 64->32, 56->112, 无skip (编码器无112x112特征)
        self.decoder4 = DecoderBlock(64, 32, 4, 2)

        # decoder5: 32->16, 112->224, 无skip
        self.decoder5 = DecoderBlock(32, 16, 4, 2)

        # decoder6: 16->3, 保持224x224 (kernel=3,stride=1,padding=1)
        self.decoder6 = nn.ConvTranspose2d(16, 3, kernel_size=3,
                                            stride=1, padding=1)

        # 输出归一化到 [0, 1]
        self.s7 = nn.Sigmoid()

    def forward(self, x, fused_features):
        """
        x:              (B, 512, 7, 7)   -- 最深融合特征 ff5
        fused_features: [ff1, ff2, ff3, ff4, ff5]
                        ff1: (B, 64,  56x56)  -- 不用于skip
                        ff2: (B, 64,  56x56)  -- skip -> decoder3
                        ff3: (B, 128, 28x28)  -- skip -> decoder2
                        ff4: (B, 256, 14x14)  -- skip -> decoder1
                        ff5: (B, 512,  7x7)   -- 初始输入 (即 x)

        跳跃连接需从深层到浅层对应解码器从浅到深的顺序:
            decoder1 (最靠近瓶颈) <- ff4 (较深层特征)
            decoder2               <- ff3
            decoder3               <- ff2
            decoder4               <- 无
        """
        # fused_features 索引: [0]=ff1, [1]=ff2, [2]=ff3, [3]=ff4, [4]=ff5
        x = self.decoder1(x, fused_features[3])   # skip: ff4 (256ch, 14x14)
        x = self.decoder2(x, fused_features[2])   # skip: ff3 (128ch, 28x28)
        x = self.decoder3(x, fused_features[1])   # skip: ff2 (64ch,  56x56)
        x = self.decoder4(x)                       # 无 skip
        x = self.decoder5(x)                       # 无 skip
        x = self.decoder6(x)                       # 16->3 (修复原代码遗漏)
        x = self.s7(x)                             # 归一化 [0,1]

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Decoder(DecoderBlock).to(device)

    # 构造测试输入
    x_test = torch.randn(2, 512, 7, 7).to(device)
    fused_test = [
        torch.randn(2, 64,  56, 56).to(device),  # ff1
        torch.randn(2, 64,  56, 56).to(device),  # ff2
        torch.randn(2, 128, 28, 28).to(device),  # ff3
        torch.randn(2, 256, 14, 14).to(device),  # ff4
        x_test,                                    # ff5
    ]

    output = model(x_test, fused_test)
    print(f"解码器输出: {output.shape}")  # 预期: (2, 3, 224, 224)
