"""
SignalEncoder (trae/arvpNoise 新增)
================================================
对围捕者 obs 中追加的 16 维时域声呐采样进行特征分离:
    1) 用 torch.fft.rfft 取复数谱 → 取对数幅度谱 (稳定 + 紧凑)
    2) 把每个频点视作一个时间步,经 1×Conv 投影到 hidden 维
    3) 双层 GRU 在频点序列上聚合
    4) 自注意力 (Self-Attention) 跨频点加权聚合
    5) 输出 context 向量, 与原 obs 编码向量拼接送入后续 FC

设计理由:
    - 真实飞行器辐射噪声在频域上有显著谐波结构 (主频 + 谐波)
    - 目标特征1 (2/5/8 Hz) 与障碍特征2 (3/7/11 Hz) 在频域完全分离
    - 频点序列上的 GRU+Attention 能自适应学习对哪几个频点敏感
      → 网络"对围捕目标/动态障碍的信号特征越来越敏感"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalEncoder(nn.Module):
    def __init__(self, n_samples=16, hidden_dim=64, n_gru_layers=2, n_heads=4):
        super(SignalEncoder, self).__init__()
        self.n_samples = n_samples
        self.n_bins = n_samples // 2 + 1      # rfft 输出长度
        self.hidden_dim = hidden_dim
        self.n_gru_layers = n_gru_layers

        # 把每个频点的对数幅度投影到 hidden_dim
        self.freq_proj = nn.Linear(1, hidden_dim)

        # 频点序列上的 GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
        )

        # 单层多头自注意力 (跨频点)
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.attn_scale = (self.head_dim ** -0.5)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, signal):
        """
        signal: [B, n_samples]  时域声呐采样 (含多源叠加 + 噪声)
        return: [B, hidden_dim]  分离后的信号上下文向量
        """
        # 1) FFT 幅度谱 (支持反向传播)
        freq = torch.fft.rfft(signal, dim=-1)         # [B, n_bins] complex
        mag = torch.abs(freq)                          # [B, n_bins]
        mag = torch.log1p(mag)                         # log(1+|X|) 稳定化
        # 2) 视作 [B, n_bins, 1] 序列
        seq = mag.unsqueeze(-1)                        # [B, n_bins, 1]
        seq = self.freq_proj(seq)                      # [B, n_bins, hidden_dim]
        # 3) GRU 聚合频点序列
        gru_out, _ = self.gru(seq)                      # [B, n_bins, hidden_dim]
        # 4) 多头自注意力
        B, L, H = gru_out.shape
        q = self.attn_q(gru_out).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, h, L, d]
        k = self.attn_k(gru_out).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.attn_v(gru_out).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale  # [B, h, L, L]
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)            # [B, h, L, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H)  # [B, L, H]
        attn_out = self.attn_out(attn_out)             # [B, L, H]
        # 残差 + LayerNorm
        attn_out = self.ln(attn_out + gru_out)
        # 5) 频点维度池化 → 单一上下文向量
        context = attn_out.mean(dim=1)                 # [B, hidden_dim]
        return self.fc_out(context)
