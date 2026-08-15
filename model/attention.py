import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    最基础的Self Attention实现
    输入:
        x:
        batch_size × 序列长度 × 特征维度
    输出:
        attention后的特征

    """
    def __init__(self,input_dim,attention_dim):
        super(SelfAttention,self).__init__()
        # ==============================
        # 将输入映射成Query
        # ==============================

        self.query = nn.Linear(input_dim,attention_dim)
        # ==============================
        # 将输入映射成Key
        # ==============================
        self.key = nn.Linear(input_dim,attention_dim)
        # ==============================
        # 将输入映射成Value
        # ==============================

        self.value = nn.Linear(input_dim,attention_dim)

    def forward(self,x):
        """
        x:

        B × N × C
        B:
        batch size
        N:
        有多少个元素
        C:
        每个元素多少维特征
        """
        # --------------------------------
        # 1.生成Q,K,V
        # --------------------------------
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        """
        此时：
        Q:
        B × N × D
        K:
        B × N × D
        V:
        B × N × D
        D:
        attention维度
        """
        # --------------------------------
        # 2.计算Q和K的相似程度
        # --------------------------------
        # 注意力 logits 的动态范围会随训练增大；在 AMP 下强制 FP32，
        # 避免 FP16 matmul 溢出后 softmax 产生 NaN。
        attention_score = torch.matmul(Q.float(), K.float().transpose(-2, -1))
        """
        Q:
        B × N × D
        K转置:
        B × D × N
        相乘:
        B × N × N
        含义：
        每个位置和其他所有位置的关系
        例如:
        attention_score[0][3][5]
        表示:
        第3个元素关注第5个元素多少
        """
        # --------------------------------
        # 3.缩放
        # --------------------------------
        attention_score = (attention_score/(Q.shape[-1] ** 0.5))
        """
        为什么除sqrt(D)?
        因为：
        D越大
        点积结果越大
        softmax容易饱和
        导致梯度消失
        所以进行缩放
        """
        # 4.softmax计算注意力权重
        attention_weight = F.softmax(attention_score,dim=-1)
        """
        softmax作用：
        将关系分数转换为概率
        例如：
        原始:
        [2,1,0]
        softmax:
        [0.66,0.24,0.10]
        表示关注比例
        """
        # 5.加权求和Value
        output = torch.matmul(attention_weight, V.float())
        """
        attention_weight:
        B × N × N
        V:
        B × N × D
        输出:
        B × N × D
        每个位置:
        根据注意力大小
        从其他位置提取信息
        """
        return output + x.float(), attention_weight

class CrossAttention(nn.Module):
    """
    最基础的Self Attention实现
    输入:
        x:
        batch_size × 序列长度 × 特征维度
    输出:
        attention后的特征

    """
    def __init__(self,input_dim,attention_dim):
        super(CrossAttention,self).__init__()
        # ==============================
        # 将输入映射成Query
        # ==============================

        self.query = nn.Linear(input_dim,attention_dim)
        # ==============================
        # 将输入映射成Key
        # ==============================
        self.key = nn.Linear(input_dim,attention_dim)
        # ==============================
        # 将输入映射成Value
        # ==============================

        self.value = nn.Linear(input_dim,attention_dim)

    def forward(self,vis,ir):
        """
        x:

        B × N × C
        B:
        batch size
        N:
        有多少个元素
        C:
        每个元素多少维特征
        """
        # --------------------------------
        # 1.生成Q,K,V
        # --------------------------------
        Q = self.query(vis)
        K = self.key(ir)
        V = self.value(ir)
        """
        此时：
        Q:
        B × N × D
        K:
        B × N × D
        V:
        B × N × D
        D:
        attention维度
        """
        # PyTorch 原生 SDPA 与上述 scaled QK^T -> softmax -> V 等价，
        # 但 CUDA 可选择数值稳定、内存高效的 kernel，不物化巨大的 B×N×N
        # 权重矩阵。当前模型从未消费 attention_weight，因此返回 None。
        output = F.scaled_dot_product_attention(
            Q.unsqueeze(1),
            K.unsqueeze(1),
            V.unsqueeze(1),
            dropout_p=0.0,
            is_causal=False,
        ).squeeze(1)
        return output + vis, None


if __name__=="__main__":
    # batch数量
    B = 2
    # 有多少个token
    N = 64
    # 每个token特征维度
    C = 512
    x = torch.randn(B,N,C)
    print("输入:")
    print(x.shape)
    attention = SelfAttention(
        input_dim=512,
        attention_dim=64
    )
    output,weight = attention(x)
    print("输出:")
    print(output.shape)
    print("Attention矩阵:")
    print(weight.shape)
