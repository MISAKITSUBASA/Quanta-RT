import torch
import torch.nn as nn
from models.backbone import VisionTransformer
from models.token_prune import prune_tokens
from models.token_quant import quantize_tensor

class QuantaVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0,
                 prune_ratio=0.0, quant_bits=0):
        """
        扩展的 Vision Transformer，支持动态 token 剪枝和量化。
        prune_ratio: 保留 token 的比例 (0表示不剪枝).
        quant_bits: 量化位宽 (0或>=32表示不量化, 如8表示8比特量化).
        """
        super().__init__(img_size, patch_size, in_chans, num_classes,
                         embed_dim, depth, num_heads, mlp_ratio,
                         drop_rate, attn_drop_rate)
        # 保存剪枝和量化配置
        self.prune_ratio = prune_ratio
        self.quant_bits = quant_bits
        # 确定在哪一层后执行剪枝，这里选择在中间层进行剪枝
        if prune_ratio > 0:
            self.prune_layer = depth // 2  # 在第 prune_layer 层后剪枝
        else:
            self.prune_layer = None

    def forward(self, x):
        B = x.shape[0]
        # Patch embedding 和位置编码 (与父类相同)
        x = self.patch_embed(x)  # [B, num_patches, dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展 CLS token
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接 CLS token
        x = x + self.pos_embed  # 加入位置嵌入
        x = self.pos_drop(x)
        # 通过 Transformer 层，按照配置执行剪枝和量化
        if self.prune_layer is None:
            # 无剪枝情况
            for layer in self.layers:
                if self.quant_bits > 0:
                    x = quantize_tensor(x, bits=self.quant_bits)
                x = layer(x)
            if self.quant_bits > 0:
                x = quantize_tensor(x, bits=self.quant_bits)
        else:
            # 前半部分层
            for i, layer in enumerate(self.layers):
                if self.quant_bits > 0:
                    x = quantize_tensor(x, bits=self.quant_bits)
                x = layer(x)
                # 在指定层后执行剪枝
                if i == self.prune_layer - 1:
                    x, keep_indices = prune_tokens(x, keep_ratio=self.prune_ratio)
                    # 保存保留的 token 索引用于后续分析/可视化
                    self.last_keep_indices = keep_indices
            # 剩余层
            for j in range(self.prune_layer, len(self.layers)):
                if self.quant_bits > 0:
                    x = quantize_tensor(x, bits=self.quant_bits)
                x = self.layers[j](x)
            if self.quant_bits > 0:
                x = quantize_tensor(x, bits=self.quant_bits)
        # 最终 LayerNorm 和分类输出
        x = self.norm(x)
        cls_token_final = x[:, 0]
        out = self.head(cls_token_final)
        return out
