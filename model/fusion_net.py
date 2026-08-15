import torch
from torch import nn
from .attention import CrossAttention
from .Encoder import VIS_Encoder, IR_Encoder, Residual
from .decoder import DecoderBlock, Decoder


# ==================================================================
# FusionBlock: 单层 VIS-IR 特征融合块
# ==================================================================
class FusionBlock(nn.Module):
    """
    对同一层级的 VIS 和 IR 特征图进行融合。

    融合流程:
        1) 将 2D 特征图展平为序列 (B, H*W, C), 适配 CrossAttention
        2) VIS 作为 Query, IR 作为 Key/Value 做交叉注意力
        3) 将注意力输出恢复为 2D 特征图
        4) 原始VIS + 原始IR + 注意力输出 -> 通道维拼接 -> 1x1Conv降维

    参数:
        channels:      该层特征图的通道数 (如 64, 128, 256, 512)
        attention_dim: 注意力内部的投影维度 (通常取与 channels 相同)
    """
    def __init__(self, channels, attention_dim):
        super(FusionBlock, self).__init__()

        # ---------------------------------------------------------
        # CrossAttention: VIS 去 "关注" IR 中有价值的信息
        # VIS -> Query, IR -> Key/Value
        # 含义: "VIS 的每个空间位置, 去 IR 的哪些位置寻找互补信息"
        # ---------------------------------------------------------
        self.cross_attn = CrossAttention(input_dim=channels,attention_dim=attention_dim)

        # ---------------------------------------------------------
        # 融合卷积: 将三路信息合并
        # 输入 3*C 通道: [原始VIS | 原始IR | 注意力输出]
        # 输出 C   通道: 融合后的紧凑表达
        # 使用 1x1 卷积实现逐点融合, 不引入空间混叠
        # ---------------------------------------------------------
        self.fusion_conv = nn.Conv2d(in_channels=channels * 3,out_channels=channels,kernel_size=1,stride=1)
        self.bn = nn.BatchNorm2d(channels)
        self.ReLU = nn.ReLU()

    def forward(self, vis, ir):
        """
        vis: (B, C, H, W) -- VIS 编码器某层输出
        ir:  (B, C, H, W) -- IR  编码器同层输出

        返回: (B, C, H, W) -- 融合后的特征图
        """
        B, C, H, W = vis.shape

        # ==================================================
        # 第1步: 将 2D 特征图 "展平" 为序列
        #
        # 原始形状: (B, C, H, W)   例如 (2, 256, 14, 14)
        # 展平:     (B, C, H*W)    例如 (2, 256, 196)
        # 转置:     (B, H*W, C)    例如 (2, 196, 256)
        #
        # 含义: 每个空间位置变成一个 "token",
        #       其向量维度 = 通道数 C
        #       例如 14x14 特征图 -> 196 个 token
        # ==================================================
        vis_flat = vis.view(B, C, -1).transpose(1, 2)   # (B, N, C), N=H*W
        ir_flat  = ir.view(B, C, -1).transpose(1, 2)    # (B, N, C), N=H*W

        # ==================================================
        # 第2步: CrossAttention 融合
        #
        # VIS 作为 Query, IR 作为 Key 和 Value
        # 输出 attn_out:     (B, N, C) -- VIS 吸收了 IR 信息后的特征
        # 输出 attn_weight:  (B, N, N) -- 注意力权重矩阵
        #   attn_weight[i][p][q] 表示第i个样本中,
        #   VIS 的位置 p 对 IR 的位置 q 关注了多少
        # ==================================================
        attn_out, attn_weight = self.cross_attn(vis_flat, ir_flat)

        # ==================================================
        # 第3步: 恢复为 2D 特征图
        #
        # (B, N, C) -> 转置 -> (B, C, N) -> 重塑 -> (B, C, H, W)
        # ==================================================
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

        # ==================================================
        # 第4步: 三路拼接 + 1x1 卷积降维
        #
        # 拼接: [VIS原始 | IR原始 | 注意力输出]
        #   VIS原始:    保留了可见光自身的纹理/颜色信息
        #   IR原始:     保留了红外自身的热辐射信息
        #   注意力输出:  VIS 从 IR 中 "挑选" 出的互补信息
        # 三者在通道维拼接 -> (B, 3C, H, W)
        # 然后 1x1 卷积压缩回 (B, C, H, W)
        # ==================================================
        fused = torch.cat([vis, ir, attn_out], dim=1)       # (B, 3C, H, W)
        fused = self.ReLU(self.bn(self.fusion_conv(fused)))  # (B, C, H, W)

        return fused


# ==================================================================
# FusionNet: 多层级融合聚合器
# ==================================================================
class FusionNet(nn.Module):
    """
    在编码器的 5 个层级分别执行 VIS-IR 融合。

    每层使用独立的 FusionBlock, 参数不共享,
    因为不同层级的语义差异很大:
        b1/b2 (56x56):  浅层纹理、边缘信息
        b3   (28x28):   中层形状信息
        b4   (14x14):   深层语义信息
        b5   (7x7):     最深层全局语义

    输入:  vis_features  -- VIS_Encoder 返回的 5 层特征列表
          ir_features    -- IR_Encoder  返回的 5 层特征列表

    输出:  fused_features -- 5 层融合特征列表
    """
    def __init__(self, FusionBlock, CrossAttention):
        super(FusionNet, self).__init__()

        # 5 个层级, 通道数对应编码器的 b1~b5
        # 注意力维度取与通道数相同 (保持特征维度一致)
        self.fuse1 = FusionBlock(channels=64,  attention_dim=64)   # b1
        self.fuse2 = FusionBlock(channels=64,  attention_dim=64)   # b2
        self.fuse3 = FusionBlock(channels=128, attention_dim=128)  # b3
        self.fuse4 = FusionBlock(channels=256, attention_dim=256)  # b4
        self.fuse5 = FusionBlock(channels=512, attention_dim=512)  # b5

    def forward(self, vis_features, ir_features):
        """
        vis_features: list[5], 每个元素 (B, C_i, H_i, W_i)
        ir_features:  list[5], 每个元素 (B, C_i, H_i, W_i)

        列表索引对应关系:
            [0]=b1, [1]=b2, [2]=b3, [3]=b4, [4]=b5
        """
        fused_features = []

        fused_features.append(self.fuse1(vis_features[0], ir_features[0]))
        fused_features.append(self.fuse2(vis_features[1], ir_features[1]))
        fused_features.append(self.fuse3(vis_features[2], ir_features[2]))
        fused_features.append(self.fuse4(vis_features[3], ir_features[3]))
        fused_features.append(self.fuse5(vis_features[4], ir_features[4]))

        return fused_features


# ==================================================================
# ResNetFusion: 完整的 IR-VIS 图像融合模型
# ==================================================================
class ResNetFusion(nn.Module):
    """
    IR-VIS 图像融合完整模型。

    数据流:
        VIS(3,224,224) --> VIS_Encoder --> [v1,v2,v3,v4,v5] --+
                                                                |
        IR (1,224,224) --> IR_Encoder  --> [i1,i2,i3,i4,i5] --+
                                                                |
                                        FusionNet (5xCrossAttention)
                                                                |
                                     [ff1,ff2,ff3,ff4,ff5] ----+
                                                                |
                  ff5(512,7x7) --> Decoder --> (3,224,224)
                                    +     +     +
                        ff4 --skip--+     |     |
                        ff3 --skip--------+     |
                        ff2 --skip--------------+

    使用方法:
        model = ResNetFusion(Residual, DecoderBlock, FusionBlock, CrossAttention)
        fused = model(vis_image, ir_image)
    """
    def __init__(self, Residual, DecoderBlock, FusionBlock, CrossAttention, ir_mode="gray"):
        super(ResNetFusion, self).__init__()

        # ==================================================
        # 双路编码器: VIS 和 IR 各自独立提取特征
        # VIS: 3通道 RGB -> 5层特征
        # IR:  1通道 灰度 -> 5层特征
        # 两者结构完全相同, 仅首层输入通道数不同
        # 权重不共享 (因为 VIS 和 IR 的统计特性差异很大)
        # ==================================================
        self.vis_encoder = VIS_Encoder(Residual)
        self.ir_mode = ir_mode
        self.ir_encoder  = IR_Encoder(Residual, ir_mode=ir_mode)

        # ==================================================
        # 融合网络: 在 5 个层级将 VIS 和 IR 特征融合
        # ==================================================
        self.fusion = FusionNet(FusionBlock, CrossAttention)

        # ==================================================
        # 解码器: 从最深特征逐步上采样重建融合图像
        # 通过跳跃连接接收 ff2/ff3/ff4 补充空间细节
        # ==================================================
        self.decoder = Decoder(DecoderBlock)

    def forward(self, vis, ir):
        """
        vis: (B, 3, 224, 224) -- 可见光图像
        ir:  (B, 1, 224, 224) -- 红外图像

        返回: (B, 3, 224, 224) -- 融合后的 RGB 图像
        """
        # 第1步: 双路编码
        vis_features = self.vis_encoder(vis)
        ir_features  = self.ir_encoder(ir)

        # 第2步: 多层级融合
        fused_features = self.fusion(vis_features, ir_features)

        # 第3步: 解码重建
        # fused_features[-1] 即 ff5 (512,7,7) 作为解码器初始输入
        # 整个 fused_features 列表用于跳跃连接
        output = self.decoder(fused_features[-1], fused_features)

        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建完整模型
    model = ResNetFusion(Residual, DecoderBlock, FusionBlock, CrossAttention)
    model = model.to(device)

    # 构造测试输入
    vis_test = torch.randn(2, 3, 224, 224).to(device)
    ir_test  = torch.randn(2, 1, 224, 224).to(device)

    # 前向传播测试
    output = model(vis_test, ir_test)
    print(f"VIS 输入:   {vis_test.shape}")
    print(f"IR  输入:   {ir_test.shape}")
    print(f"融合输出:   {output.shape}")
    print(f"输出范围:   [{output.min().item():.4f}, {output.max().item():.4f}]")

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量:   {total_params / 1e6:.2f} M")
    print(f"可训练参数: {train_params / 1e6:.2f} M")
