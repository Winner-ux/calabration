"""
红外-可见光图像融合 无监督损失函数模块

设计思想：
    由于融合任务没有Ground Truth，损失函数需要同时约束：
    1. 红外目标信息保持（通过强度损失）
    2. 可见光纹理细节保持（通过梯度损失）
    3. 自然结构保持（通过SSIM结构损失）
    4. 边缘强化（通过边缘损失）

核心策略：Max-Selection
    对每个像素位置，取红外和可见光中"更强"的响应作为目标。
    红外擅长捕获热辐射目标（高亮区域），
    可见光擅长捕获反射纹理（边缘、细节），
    max策略确保融合图像保留两模态各自优势。

理论依据：
    红外图像反映的是场景的热辐射分布，亮度值直接关联目标温度。
    可见光图像反映的是场景的反射率分布，纹理和边缘信息丰富。
    两者在统计特性上互补——红外的高亮区域往往对应可见光中
    纹理稀疏的区域（如人体、车辆），而可见光的纹理区域
    在红外中可能亮度均匀。max操作天然地选择了两者中
    "更显著"的响应，实现了信息互补保留。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# SSIMLoss: 结构相似性损失子模块
# ==================================================================
class SSIMLoss(nn.Module):
    """
    自实现SSIM（结构相似性）计算模块。
    不依赖第三方库，纯PyTorch实现，支持GPU和混合精度训练。

    SSIM公式（简化形式）：
        SSIM(x,y) = (2*μx*μy + C1)*(2*σxy + C2)
                  / ((μx² + μy² + C1)*(σx² + σy² + C2))

    其中：
        μx, μy    — 局部均值（高斯加权）
        σx², σy² — 局部方差
        σxy       — 局部协方差
        C1, C2    — 稳定性常数，防止除零

    实现方式：
        使用11×11高斯窗口作为滑动窗口，
        通过 F.conv2d 的 groups=3 参数逐通道高效计算局部统计量。
        三通道分别计算SSIM后取均值，与原始SSIM论文一致。

    参数说明：
        window_size: 高斯窗口大小，默认11（SSIM论文标准值）
        sigma:       高斯核标准差，默认1.5（论文推荐值）
        C1:          亮度稳定性常数，默认(0.01)²
                     含义：当像素值范围为[0,1]时，0.01的动态范围视为可忽略
        C2:          对比度稳定性常数，默认(0.03)²
                     含义：当像素值范围为[0,1]时，0.03的对比度变化视为可忽略
    """
    def __init__(self, window_size=11, sigma=1.5, C1=0.0001, C2=0.0009):
        super(SSIMLoss, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.window_size = window_size

        # ==================================================
        # 生成 11×11 二维高斯窗口
        #
        # 步骤：
        #   1) 生成一维高斯向量 g_1d，长度为 window_size
        #   2) 外积得到二维高斯核: g_2d = g_1d^T × g_1d
        #      （利用高斯函数的可分离性：二维高斯 = 一维高斯的乘积）
        #   3) 归一化使窗口内所有权重之和 = 1
        #   4) reshape 为 (1,1,11,11) 再 expand 为 (3,1,11,11)
        #      适配 F.conv2d 的 groups=3 模式：
        #        - 3 组 = RGB 三个通道各自独立计算
        #        - 每组 1 个输入通道、1 个输出通道
        #        - 核形状 (3, 1, 11, 11) 表示 3 个组各有 1 个 11×11 核
        # ==================================================

        # 第1步：生成一维高斯分布
        # coords: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        # g_1d: 高斯权重，中心最大，向两边衰减
        g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g_1d = g_1d / g_1d.sum()  # 归一化：所有权重之和 = 1

        # 第2步：外积 → 二维高斯核
        # g_1d.unsqueeze(0): (1, 11)  列向量
        # g_1d.unsqueeze(1): (11, 1)  行向量
        # 外积结果: (11, 11)  二维高斯核
        g_2d = g_1d.unsqueeze(0) * g_1d.unsqueeze(1)
        g_2d = g_2d / g_2d.sum()  # 再次归一化，确保二维核权重和为1

        # 第3步：reshape + expand → 适配 groups=3 卷积
        # (11, 11) → (1, 1, 11, 11) → (3, 1, 11, 11)
        window = g_2d.view(1, 1, window_size, window_size)
        window = window.expand(3, 1, window_size, window_size).contiguous()

        # 注册为 buffer：
        #   - 不参与梯度更新（无需优化）
        #   - 随 model.to(device) 自动迁移 CPU/GPU
        #   - 随 model.float()/half() 不改变（内核强制 float32）
        self.register_buffer('window', window)

    def forward(self, img1, img2):
        """
        计算两张图像的 SSIM（结构相似性）。

        输入：
            img1: (B, 3, H, W) — 图像1（如融合图像）
            img2: (B, 3, H, W) — 图像2（如可见光图像）

        返回：
            ssim_val: 标量 Tensor — 三通道 SSIM 均值
                      取值范围约 [0, 1]，值越大表示越相似

        计算流程：
            1) 用高斯窗口 F.conv2d 计算局部均值 μ
            2) 通过二次矩减一次矩平方计算局部方差 σ² 和协方差 σ12
            3) 代入 SSIM 简化公式得到逐像素 SSIM 图
            4) 对所有维度（batch、通道、空间）取平均
        """
        # -------------------------------------------------
        # 确保在 float32 下计算
        # 原因：SSIM 的 C1=(0.01)²=0.0001 在 float16
        #       下可能被截断为 0，导致数值不稳定。
        #       混合精度训练时模型输出可能是 float16，
        #       这里显式转换为 float32 保证精度。
        # -------------------------------------------------
        img1 = img1.float()
        img2 = img2.float()

        # -------------------------------------------------
        # 第1步：计算局部均值 μ（高斯加权平均）
        #
        # F.conv2d 对每个像素位置执行：
        #   output[pixel] = Σ(input[neighbor] × gaussian[neighbor])
        #
        # groups=3：R/G/B 三通道各自独立计算
        # padding=窗口半径(5)：保持输入输出尺寸一致
        #
        # 形状变化：img: (B,3,H,W) → mu: (B,3,H,W)
        # -------------------------------------------------
        pad = self.window_size // 2  # 11//2 = 5
        mu1 = F.conv2d(img1, self.window, groups=3, padding=pad)
        mu2 = F.conv2d(img2, self.window, groups=3, padding=pad)

        # 预计算均值的平方和乘积（后续复用）
        mu1_sq = mu1 ** 2        # μx²
        mu2_sq = mu2 ** 2        # μy²
        mu1_mu2 = mu1 * mu2      # μx·μy

        # -------------------------------------------------
        # 第2步：计算局部方差和协方差
        #
        # 利用公式：σ² = E[X²] - (E[X])²
        #           σxy = E[XY] - E[X]·E[Y]
        #
        # E[X²] 通过 F.conv2d(img², window) 计算
        # （高斯加权平均等价于局部期望）
        # -------------------------------------------------
        sigma1_sq = F.conv2d(img1 * img1, self.window, groups=3, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, groups=3, padding=pad) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, self.window, groups=3, padding=pad) - mu1_mu2

        # -------------------------------------------------
        # 第3步：SSIM 简化公式
        #
        # 分子：(2·μx·μy + C1) × (2·σxy + C2)
        #   - 第一项衡量亮度相似度
        #   - 第二项衡量结构相似度（协方差反映结构关系）
        #
        # 分母：(μx² + μy² + C1) × (σx² + σy² + C2)
        #   - 第一项归一化亮度差异
        #   - 第二项归一化对比度差异
        #
        # C1, C2 的作用：
        #   当局部区域亮度/对比度接近零时
        #   （如纯黑区域），防止分母为零导致数值爆炸
        # -------------------------------------------------
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        # -------------------------------------------------
        # 第4步：全局平均
        #
        # ssim_map 形状: (B, 3, H, W)
        # .mean() 对 batch、通道、空间所有维度取平均
        # 得到单个标量 SSIM 值
        # -------------------------------------------------
        return ssim_map.mean()


# ==================================================================
# FusionLoss: 红外-可见光融合总损失
# ==================================================================
class FusionLoss(nn.Module):
    """
    IR-VIS 图像融合无监督损失函数。

    总损失公式：
        L_total = λ1·L_intensity + λ2·L_gradient + λ3·L_SSIM + λ4·L_edge

    四个分量分别约束：
        L_intensity  — 红外目标强度保持（确保热目标不丢失）
        L_gradient   — 纹理细节保持（确保可见光纹理不模糊）
        L_SSIM       — 结构相似性保持（确保融合图看起来"自然"）
        L_edge       — 边缘强化（确保目标边界和纹理边界清晰）

    Max-Selection 策略：
        每个分量中，使用 max(IR响应, VIS响应) 作为目标值。
        背后逻辑：融合图像的目的不是"平均"，而是"取优"——
        哪个模态在某位置信息更强，就该保留那个模态的信息。

    参数说明：
        lambda_intensity: 强度损失权重，默认 1
        lambda_gradient:  梯度损失权重，默认 10
                          （梯度损失通常权重较大，因为纹理保持对视觉质量影响大）
        lambda_ssim:      SSIM 损失权重，默认 5
        lambda_edge:      边缘损失权重，默认 2
                          （由于梯度损失已部分覆盖边缘，边缘权重较小）

    使用示例：
        criterion = FusionLoss()
        fused = model(vis, ir)
        total_loss, loss_dict = criterion(fused, ir, vis)
        # loss_dict: {'intensity_loss': ..., 'gradient_loss': ...,
        #              'ssim_loss': ..., 'edge_loss': ...}
    """
    def __init__(
        self,
        lambda_intensity=1,
        lambda_gradient=10,
        lambda_ssim=5,
        lambda_edge=2
    ):
        super(FusionLoss, self).__init__()

        # ==================================================
        # 损失权重注册
        # 注册为 buffer 以便保存到 checkpoint
        # 权重可随训练阶段动态调整
        # ==================================================
        self.register_buffer('lambda_intensity', torch.tensor(lambda_intensity, dtype=torch.float32))
        self.register_buffer('lambda_gradient',  torch.tensor(lambda_gradient, dtype=torch.float32))
        self.register_buffer('lambda_ssim',      torch.tensor(lambda_ssim, dtype=torch.float32))
        self.register_buffer('lambda_edge',      torch.tensor(lambda_edge, dtype=torch.float32))

        # ==================================================
        # 梯度卷积核（固定，不参与训练）
        #
        # 水平梯度核: [[-1, 1]]
        #   作用：I(x+1, y) - I(x, y)
        #   即：右侧像素 − 当前像素 = x方向差分
        #
        # 垂直梯度核: [[-1], [1]]
        #   作用：I(x, y+1) - I(x, y)
        #   即：下方像素 − 当前像素 = y方向差分
        #
        # 形状说明：
        #   (3, 1, kH, kW) 用于 groups=3 卷积
        #   3 = RGB 三通道各自独立
        #   1 = 每组 1 个输入通道
        # ==================================================
        grad_kernel_x = torch.tensor([[[[-1., 1.]]]], dtype=torch.float32)  # (1,1,1,2)
        grad_kernel_y = torch.tensor([[[[-1.], [1.]]]], dtype=torch.float32)  # (1,1,2,1)
        self.register_buffer('grad_kernel_x', grad_kernel_x.expand(3, 1, 1, 2).contiguous())
        self.register_buffer('grad_kernel_y', grad_kernel_y.expand(3, 1, 2, 1).contiguous())

        # ==================================================
        # Laplacian 边缘检测核（4-邻域）
        #
        # 核定义：
        #     [[ 0,  1,  0],
        #      [ 1, -4,  1],
        #      [ 0,  1,  0]]
        #
        # 作用：二阶导数，检测像素强度的突变区域（边缘）
        #   正/负值表示边缘的上升/下降沿
        #
        # 为什么用4-邻域而非8-邻域？
        #   8-邻域 [[1,1,1],[1,-8,1],[1,1,1]] 对噪声更敏感，
        #   4-邻域在保持边缘检测能力的同时更稳定
        # ==================================================
        laplacian_kernel = torch.tensor(
            [[[[0., 1., 0.],
               [1., -4., 1.],
               [0., 1., 0.]]]],
            dtype=torch.float32
        )  # (1,1,3,3)
        self.register_buffer('laplacian_kernel', laplacian_kernel.expand(3, 1, 3, 3).contiguous())

        # ==================================================
        # SSIM 计算器
        # ==================================================
        self.ssim = SSIMLoss()

    # ==================================================================
    # 梯度幅值计算
    # ==================================================================
    def _compute_gradient_magnitude(self, img):

        # ??1???padding
        img_pad_x = F.pad(img, (0, 1, 0, 0), mode="replicate")
        img_pad_y = F.pad(img, (0, 0, 0, 1), mode="replicate")

        # ??2?????
        grad_x = F.conv2d(img_pad_x, self.grad_kernel_x, groups=3)

        # ??3?????
        grad_y = F.conv2d(img_pad_y, self.grad_kernel_y, groups=3)

        # ??4???????
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        return grad_mag
    # ==================================================================
    # Laplacian 边缘响应计算
    # ==================================================================
    def _compute_laplacian(self, img):
        """
        使用4-邻域Laplacian算子计算边缘响应图。

        输入：
            img: (B, 3, H, W) — RGB图像

        返回：
            edge_map: (B, 3, H, W) — 边缘响应图（取绝对值）
                      值越大表示该位置越可能是边缘

        原理：
            Laplacian = ∂²I/∂x² + ∂²I/∂y²
            在边缘处（强度突变）Laplacian绝对值大；
            在平滑区域Laplacian接近0。
        """
        # F.conv2d + padding=1 保持输入输出尺寸一致
        lap = F.conv2d(img.float(), self.laplacian_kernel, groups=3, padding=1)

        # 取绝对值：边缘可能是正（暗→亮）或负（亮→暗），我们都关心其强度
        return torch.abs(lap)

    # ==================================================================
    # 1) 强度损失
    # ==================================================================
    def intensity_loss(self, fused, ir, vis):
        """
        强度保持损失。
        确保融合图像的红外目标和可见光亮区信息不丢失。

        目标构造：
            I_target = max(IR, VIS)
            为什么取max？
            - 红外图像中的热目标（如人体、车辆）通常亮度很高
            - 可见光图像中的亮区（如天空、灯光）也包含重要信息
            - max确保融合图保留"任一模态中更亮"的区域

        损失：
            L = mean(|fused - I_target|)  [L1距离]

        返回：
            标量 — 强度损失值
        """
        # 构造目标强度：逐元素取最大值
        I_target = torch.max(ir, vis)

        # L1 距离：fused 与目标强度之差的绝对值的均值
        return torch.mean(torch.abs(fused - I_target))

    # ==================================================================
    # 2) 梯度损失
    # ==================================================================
    def gradient_loss(self, fused, ir, vis):
        """
        梯度保持损失。
        确保融合图像保留两模态中更清晰的纹理和边缘。

        计算流程：
            1) 分别计算 fused、ir、vis 的梯度幅值图
            2) 目标梯度 = max(梯度_ir, 梯度_vis)
               含义：对每个像素，保留两模态中"梯度更强"的响应
            3) L = mean(|梯度_fused - 梯度_target|)  [L1距离]

        为什么梯度损失重要？
            人类视觉系统对边缘和纹理敏感。
            如果只优化强度损失，融合图可能模糊（类似均值滤波效果）。
            梯度损失显式地约束边缘清晰度。

        返回：
            标量 — 梯度损失值
        """
        # 步骤1：计算三个图像的梯度幅值
        grad_fused = self._compute_gradient_magnitude(fused)
        grad_ir    = self._compute_gradient_magnitude(ir)
        grad_vis   = self._compute_gradient_magnitude(vis)

        # 步骤2：构造目标梯度（逐元素取最大）
        grad_target = torch.max(grad_ir, grad_vis)

        # 步骤3：L1 距离
        return torch.mean(torch.abs(grad_fused - grad_target))

    # ==================================================================
    # 3) SSIM 结构损失
    # ==================================================================
    def ssim_loss(self, fused, ir, vis):
        """
        结构相似性损失。
        确保融合图像具有"自然"的图像结构。

        策略（无Ground Truth情况下的折中方案）：
            SSIM_target = max( SSIM(fused, vis), SSIM(fused, ir) )
            L_SSIM = 1 - SSIM_target

        为什么用max而非平均？
            - 融合图像的每个局部区域可能更像可见光或更像红外
            - 如果某个区域的红外和可见光结构冲突，max策略允许
              融合图选择与其中一个模态的结构对齐
            - 这比强制与两者同时对齐（min/avg）更灵活

        为什么是 1 - SSIM？
            - SSIM 取值范围 [0,1]，1表示完全相同
            - 1 - SSIM 将"最大化相似性"转换为"最小化损失"

        返回：
            标量 — SSIM 损失值
        """
        # 计算 fused 分别与 vis 和 ir 的结构相似度
        ssim_fused_vis = self.ssim(fused, vis)   # 标量
        ssim_fused_ir  = self.ssim(fused, ir)    # 标量

        # 取最大结构相似度作为目标
        ssim_target = torch.max(ssim_fused_vis, ssim_fused_ir)

        # 转换为损失：1 - SSIM
        return 1 - ssim_target

    # ==================================================================
    # 4) 边缘损失
    # ==================================================================
    def edge_loss(self, fused, ir, vis):
        """
        边缘强化损失。
        基于Laplacian二阶导数，显式约束融合图像的边缘保真度。

        计算流程：
            1) 用4-邻域Laplacian核计算三张图的边缘响应
            2) 目标边缘 = max(边缘_ir, 边缘_vis)
            3) L = mean(|边缘_fused - 边缘_target|)  [L1距离]

        与梯度损失的区别：
            - 梯度损失（一阶导数）：关注"变化量"
              适合捕获纹理、渐变等缓慢变化
            - 边缘损失（二阶导数）：关注"变化率的变化"
              适合精确定位物体边界和突变边缘
            两者互补：一阶保证纹理清晰，二阶保证边界锐利

        返回：
            标量 — 边缘损失值
        """
        # 步骤1：计算三张图的Laplacian边缘响应（已取绝对值）
        edge_fused = self._compute_laplacian(fused)
        edge_ir    = self._compute_laplacian(ir)
        edge_vis   = self._compute_laplacian(vis)

        # 步骤2：构造目标边缘（逐元素取最强响应）
        edge_target = torch.max(edge_ir, edge_vis)

        # 步骤3：L1 距离
        return torch.mean(torch.abs(edge_fused - edge_target))

    # ==================================================================
    # forward: 计算总损失
    # ==================================================================
    def forward(self, fused, ir, vis):
        """
        计算总损失和各分量损失。

        输入：
            fused: (B, 3, H, W) — 网络输出的融合图像，值范围 [0, 1]
            ir:    (B, 1, H, W) — 红外输入图像，值范围 [0, 1]
            vis:   (B, 3, H, W) — 可见光输入图像，值范围 [0, 1]

        返回：
            total_loss: 标量 Tensor — 加权总损失
            loss_dict:  dict — 各分量损失值（用于监控训练）
                {
                    'intensity_loss': 标量,
                    'gradient_loss':  标量,
                    'ssim_loss':      标量,
                    'edge_loss':      标量
                }

        处理流程：
            0) IR 通道扩展：若 IR 是单通道 (B,1,H,W)，扩展为 (B,3,H,W)
            1) 计算四个分量的损失值
            2) 加权求和得到总损失
            3) 返回总损失和分量字典
        """
        # ==================================================
        # 步骤0：IR 通道扩展
        #
        # expand 与 repeat 的区别：
        #   - expand: 零拷贝，只修改 tensor 的 stride 元数据
        #             不会真正复制内存 → 高效但要求原始维度为1
        #   - repeat: 真正复制数据 → 内存开销大
        #
        # 此处 IR 通道数=1，适合用 expand。
        # 扩展后 IR 三通道共享同一份数据（灰度图重复三次），
        # 等价于在 RGB 三个通道上施加相同的约束。
        # ==================================================
        if ir.shape[1] == 1:
            ir = ir.expand(-1, 3, -1, -1)  # (B,1,H,W) → (B,3,H,W)

        # ==================================================
        # 步骤1：计算四个分量的损失值
        # 每个都返回标量 Tensor
        # ==================================================
        L_intensity = self.intensity_loss(fused, ir, vis)
        L_gradient  = self.gradient_loss(fused, ir, vis)
        L_ssim      = self.ssim_loss(fused, ir, vis)
        L_edge      = self.edge_loss(fused, ir, vis)

        # ==================================================
        # 步骤2：加权求和 — 总损失
        #
        # 权重设计逻辑：
        #   λ_gradient=10（最大）：纹理清晰度对视觉效果影响最大
        #   λ_ssim=5：         结构保真保证图像"看着像真的"
        #   λ_edge=2：         精细边缘约束（梯度损失已覆盖大部分）
        #   λ_intensity=1（最小）：强度约束相对简单，过度约束会
        #                         导致融合图偏暗（趋向红外）
        # ==================================================
        total_loss = (
            self.lambda_intensity * L_intensity +
            self.lambda_gradient  * L_gradient +
            self.lambda_ssim      * L_ssim +
            self.lambda_edge      * L_edge
        )

        # ==================================================
        # 步骤3：构造分量字典（用于训练监控和日志记录）
        # .item() 提取 Python 浮点数，避免保留计算图
        # ==================================================
        loss_dict = {
            'intensity_loss': L_intensity.item(),
            'gradient_loss':  L_gradient.item(),
            'ssim_loss':      L_ssim.item(),
            'edge_loss':      L_edge.item(),
        }

        return total_loss, loss_dict


# ==================================================================
# 自测代码
# ==================================================================
if __name__ == "__main__":
    """
    自测目的：
        1) 验证各损失分量前向传播正常
        2) 验证反向传播梯度流正常
        3) 验证 CPU/GPU 兼容性
        4) 验证 IR 单通道自动扩展
        5) 观察损失分量数值范围
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")
    print("=" * 60)

    # 初始化损失函数
    criterion = FusionLoss(
        lambda_intensity=1,
        lambda_gradient=10,
        lambda_ssim=5,
        lambda_edge=2
    )
    criterion = criterion.to(device)

    # 构造测试输入
    # fused/vis: (B,3,H,W) — RGB三通道
    # ir:        (B,1,H,W) — 单通道灰度
    B, H, W = 2, 224, 224
    fused_test = torch.rand(B, 3, H, W).to(device)
    vis_test   = torch.rand(B, 3, H, W).to(device)
    ir_test    = torch.rand(B, 1, H, W).to(device)

    print(f"fused 形状: {fused_test.shape}")
    print(f"ir    形状: {ir_test.shape}")
    print(f"vis   形状: {vis_test.shape}")
    print("=" * 60)

    # -------------------------------------------------
    # 测试1：前向传播
    # -------------------------------------------------
    print("\n[测试1] 前向传播...")
    total_loss, loss_dict = criterion(fused_test, ir_test, vis_test)

    print(f"  total_loss     = {total_loss.item():.6f}")
    print(f"  intensity_loss = {loss_dict['intensity_loss']:.6f}")
    print(f"  gradient_loss  = {loss_dict['gradient_loss']:.6f}")
    print(f"  ssim_loss      = {loss_dict['ssim_loss']:.6f}")
    print(f"  edge_loss      = {loss_dict['edge_loss']:.6f}")

    # 验证损失值为正
    assert total_loss.item() > 0, "总损失应为正值"
    for name, val in loss_dict.items():
        assert val > 0, f"{name} 应为正值"

    # -------------------------------------------------
    # 测试2：反向传播
    # -------------------------------------------------
    print("\n[测试2] 反向传播（需要 requires_grad=True）...")
    fused_grad = torch.rand(B, 3, H, W).to(device).requires_grad_(True)
    vis_grad   = torch.rand(B, 3, H, W).to(device)
    ir_grad    = torch.rand(B, 1, H, W).to(device)

    total_loss, _ = criterion(fused_grad, ir_grad, vis_grad)
    total_loss.backward()

    # 验证梯度存在且不为零
    assert fused_grad.grad is not None, "fused 应有梯度"
    print(f"  fused.grad 范数: {fused_grad.grad.norm().item():.6f}")
    print("  反向传播通过！")

    # -------------------------------------------------
    # 测试3：IR 3通道输入（不应再扩展）
    # -------------------------------------------------
    print("\n[测试3] IR 三通道输入（skip expand）...")
    ir_3ch = torch.rand(B, 3, H, W).to(device)
    total_loss_3ch, _ = criterion(fused_test, ir_3ch, vis_test)
    print(f"  total_loss (IR 3ch): {total_loss_3ch.item():.6f}")

    # -------------------------------------------------
    # 测试4：SSIM 边界值测试
    # -------------------------------------------------
    print("\n[测试4] SSIM 边界值测试...")
    ssim_test = SSIMLoss().to(device)
    # 相同图像的 SSIM 应接近 1
    same_img = torch.rand(B, 3, H, W).to(device)
    ssim_same = ssim_test(same_img, same_img)
    print(f"  SSIM(相同图像) = {ssim_same.item():.6f}  (期望 ≈ 1)")

    print("\n" + "=" * 60)
    print("所有测试通过！")


# ==================================================================
# 训练集成说明
# ==================================================================
"""
============================================================
一、训练循环示例
============================================================

```python
import torch
from model import ResNetFusion
from loss import FusionLoss

# ------------------- 初始化 -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetFusion(...).to(device)
criterion = FusionLoss(
    lambda_intensity=1,
    lambda_gradient=10,
    lambda_ssim=5,
    lambda_edge=2
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------------------- 训练循环 -------------------
for epoch in range(num_epochs):
    for vis, ir in dataloader:
        vis = vis.to(device)        # (B, 3, H, W)
        ir  = ir.to(device)         # (B, 1, H, W)

        # 1. 前向传播
        optimizer.zero_grad()
        fused = model(vis, ir)      # (B, 3, H, W)

        # 2. 计算损失
        total_loss, loss_dict = criterion(fused, ir, vis)

        # 3. 反向传播
        total_loss.backward()
        optimizer.step()

        # 4. 日志记录（可用 TensorBoard / WandB）
        # writer.add_scalars('loss', loss_dict, global_step)
```

============================================================
二、混合精度训练支持
============================================================

```python
scaler = torch.cuda.amp.GradScaler()

for vis, ir in dataloader:
    optimizer.zero_grad()

    # 模型前向用 autocast（自动 float16）
    with torch.cuda.amp.autocast():
        fused = model(vis, ir)

    # loss 计算在 float32 下（SSIMLoss 内部 .float() 保证）
    total_loss, loss_dict = criterion(fused, ir, vis)

    # 混合精度反向传播
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

============================================================
三、训练过程中各 Loss 分量的变化趋势解读
============================================================

| 分量            | 训练初期 | 正常训练中       | 异常信号                  |
|----------------|---------|-----------------|--------------------------|
| intensity_loss | 快速下降 | 缓慢下降至稳定    | 不下降 → 模型未学习        |
| gradient_loss  | 较快下降 | 持续缓慢下降      | 震荡剧烈 → λ_gradient 过大 |
| ssim_loss      | 较高     | 逐渐下降          | 不降反升 → 融合图失真      |
| edge_loss      | 中等     | 缓慢波动下降      | 快速趋零 → 边缘过平滑      |

训练良好的表现：
    - 四个分量均呈下降趋势
    - intensity_loss 和 ssim_loss 最先收敛（约前30%的epoch）
    - gradient_loss 和 edge_loss 在整个训练中持续缓慢下降
    - 最终 fused 图像在视觉上兼具红外热目标和可见光纹理

============================================================
四、理论解释：为什么 max 策略适用于 IR-VIS 融合？
============================================================

1. 成像机制的互补性
   - 红外传感器：接收目标的自身热辐射
     → 热目标（人体、发动机）亮度高，但纹理信息少
   - 可见光传感器：接收目标的反射光
     → 纹理细节丰富，但暗处/夜间目标不可见

2. max 策略的物理含义
   - 对每个像素位置，红外和可见光的响应反映了不同的物理量
     （热辐射 vs 反射率）
   - max(I_ir, I_vis) 选择"在当前像素位置上信息更显著"的模态
   - 这与融合的目标一致：不是平均，而是择优

3. 与其他策略的对比
   - 平均策略 (I_ir + I_vis)/2：
     → 导致两种信息相互"稀释"，融合图既不够热也不够清晰
   - 加权和策略 w1*I_ir + w2*I_vis：
     → 需要手工调节权重，不具备自适应性
   - max 策略：
     → 自适应选择，不需要额外参数
     → 在很多文献中被验证为简单且有效的策略
       (DenseFuse, FusionGAN, U2Fusion 等均使用类似方法)

4. 四个损失分量的协同作用
   - intensity_loss 保证"该亮的区域亮起来"（热目标）
   - gradient_loss 保证"该清晰的纹理清晰"（可见光细节）
   - ssim_loss 保证"结构看起来自然"（避免伪影）
   - edge_loss 保证"边界锐利"（目标轮廓不模糊）
   四者联合约束，从不同角度引导网络学习最优融合策略。
"""
