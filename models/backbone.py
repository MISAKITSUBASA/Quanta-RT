import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """将输入图像划分为固定大小的补丁，并映射到指定的嵌入维度"""
        super().__init__()
        # 支持 img_size 为 int 或 (H,W) 元组
        if isinstance(img_size, tuple):
            self.img_height, self.img_width = img_size
        else:
            self.img_height = self.img_width = img_size
        self.patch_size = patch_size
        # 计算垂直和水平方向上的 patch 数量
        self.grid_size = (self.img_height // patch_size, self.img_width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 使用一个卷积层将图像切分为 patch 并映射到 embed_dim 维度
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 输入 x: [B, in_chans, H, W]
        B, C, H, W = x.shape
        # 卷积映射: 输出 [B, embed_dim, H_patch, W_patch]
        x = self.proj(x)  # 切分 patch 并映射到 embed_dim 维度
        # 展平成序列 [B, embed_dim, N_patches]
        x = x.flatten(2)  # 将 H_patch 和 W_patch 两个维度展平
        x = x.transpose(1, 2)  # 转换为 [B, N_patches, embed_dim]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        """多头自注意力模块"""
        super().__init__()
        assert dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim) 缩放因子
        # 查询、键、值的线性变换层 (输出总维度仍为 dim)
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        # 注意力和输出的 dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # 输入 x: [B, N, D] (N = 序列长度, 包含CLS token, D = dim)
        B, N, D = x.shape
        # 线性变换得到 Q, K, V
        q = self.Wq(x)  # [B, N, D]
        k = self.Wk(x)  # [B, N, D]
        v = self.Wv(x)  # [B, N, D]
        # 拆分多头，将 D 维度拆成 [num_heads, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        # 计算注意力分数: Q * K^T / sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_probs = F.softmax(attn_scores, dim=-1)  # 对最后一维做 softmax
        attn_probs = self.attn_drop(attn_probs)
        # 基于注意力权重加权求和 V
        attn_out = torch.matmul(attn_probs, v)  # [B, num_heads, N, head_dim]
        # 合并多头: 转置并reshape回 [B, N, D]
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        # 输出线性投影
        out = self.proj(attn_out)  # [B, N, D]
        out = self.proj_drop(out)
        return out

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        """Transformer FFN层: 包含两层全连接和GELU激活"""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 第一层将维度从 in_features 扩展到 hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 使用 GELU 作为激活函数
        self.act = nn.GELU()
        # Dropout
        self.drop = nn.Dropout(drop)
        # 第二层将维度缩回 out_features
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        """Transformer 编码器层: 多头注意力 + 前馈网络"""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=hidden_dim, out_features=dim, drop=drop)

    def forward(self, x):
        # 自注意力子层 (预归一化 + 残差连接)
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residual + x  # 残差相加
        # 前馈网络子层 (预归一化 + 残差连接)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0):
        """标准 Vision Transformer 模型"""
        super().__init__()
        # 图像补丁嵌入模块
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        # 分类 token (可学习参数)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置嵌入 (包含CLS token位置，总长度 num_patches+1)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        # 多层 Transformer 编码器
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        # 最终分类前的 LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # 参数初始化
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        # 注: PyTorch默认Linear/Conv已用Kaiming初始化，这里简化未重新初始化其他参数

    def forward(self, x):
        # 输入 x: [B, in_chans, H, W]
        B = x.shape[0]
        # Patch Embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        # 拼接分类 token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+num_patches, embed_dim]
        # 加入位置嵌入
        x = x + self.pos_embed  # (假定输入尺寸不变; 若不同, 需插值位置嵌入)
        x = self.pos_drop(x)
        # 通过所有 Transformer 编码层
        for layer in self.layers:
            x = layer(x)
        # 最终 LayerNorm
        x = self.norm(x)
        # 取出分类 token 对应的表示并通过分类头
        cls_token_final = x[:, 0]  # [B, embed_dim]
        out = self.head(cls_token_final)  # [B, num_classes]
        return out
