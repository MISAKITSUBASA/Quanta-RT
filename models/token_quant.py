import torch

def quantize_tensor(tensor, bits=8):
    """
    将张量近似量化为指定bit宽度 (fake quantization)，返回量化后的张量 (浮点表示)。
    """
    if bits <= 0 or bits >= 32:
        # bits<=0表示不量化, bits>=32视为使用32位全精度
        return tensor
    # 对称量化范围
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    # 计算张量绝对值的最大值用于缩放
    max_val = float(tensor.abs().max())
    if max_val == 0:
        return tensor
    scale = max_val / qmax
    # 假量化: 将张量除以scale后四舍五入到整数并截断，然后再乘回scale
    tensor_int = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    tensor_quantized = tensor_int * scale
    # 注: 直接 round 会阻断梯度, 实际训练中可用 straight-through estimator 保持梯度传递
    return tensor_quantized

def quantize_model_weights(model, bits=8):
    """
    将模型的权重参数量化到指定 bit 宽度 (直接修改模型参数值)。
    """
    # 遍历模型的各个模块
    for name, module in model.named_modules():
        # 针对 Linear 和 Conv2d 层进行权重量化
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            module.weight.data = quantize_tensor(module.weight.data, bits)
            if module.bias is not None:
                module.bias.data = quantize_tensor(module.bias.data, bits)
    # 单独处理位置嵌入和分类token等参数
    if hasattr(model, 'pos_embed'):
        model.pos_embed.data = quantize_tensor(model.pos_embed.data, bits)
    if hasattr(model, 'cls_token'):
        model.cls_token.data = quantize_tensor(model.cls_token.data, bits)
    # 注意: 未对 LayerNorm 等层的参数执行量化，以保证数值稳定
