import torch

def prune_tokens(tokens, keep_ratio=0.5):
    """
    基于重要性分数对输入 token 序列进行剪枝。
    参数:
        tokens (Tensor): [B, N, D] 输入序列 (包含CLS token在 index 0).
        keep_ratio (float): 保留的 token 比例 (0~1).
    返回:
        pruned_tokens (Tensor): 剪枝后的 token 序列 [B, N_keep, D].
        keep_indices (List[List[int]]): 每个样本保留的 token 索引列表.
    """
    B, N, D = tokens.shape
    if keep_ratio <= 0 or keep_ratio >= 1:
        # 不剪枝，返回原始 tokens
        keep_indices = [list(range(N)) for _ in range(B)]
        return tokens, keep_indices
    # 排除 CLS token，仅考虑 patch tokens
    patch_count = N - 1
    # 计算需要保留的 patch token 数量
    keep_count = max(1, int(patch_count * keep_ratio))
    # 计算每个 token 的重要性 (这里使用 token 向量的 L2 范数作为重要性度量)
    importance = torch.norm(tokens[:, 1:, :], dim=2)  # [B, patch_count]
    # 获取每个样本中分数最高的 keep_count 个 token 的索引
    topk_indices = torch.topk(importance, k=keep_count, dim=1, largest=True, sorted=False).indices  # [B, keep_count]
    # 将索引升序排序（恢复为原序出现顺序）
    topk_indices, _ = torch.sort(topk_indices, dim=1)
    # 添加 CLS token 索引0到保留列表前部，并提取保留索引的 tokens
    keep_indices = []
    pruned_list = []
    for b in range(B):
        idx = topk_indices[b] + 1  # 将相对索引转换为绝对索引(+1跳过CLS位置)
        idx = torch.cat([torch.zeros(1, dtype=torch.long, device=idx.device), idx])
        idx_list = idx.tolist()
        keep_indices.append(idx_list)
        # 提取对应的 tokens 子集 (保留顺序)
        pruned_list.append(tokens[b:b+1, idx_list, :])
    # 拼接剪枝后的 batch 序列
    pruned_tokens = torch.cat(pruned_list, dim=0)  # [B, 1+keep_count, D]
    # 注: 简单使用L2范数估计重要性，可替换为轻量预测模块以获得更优性能
    return pruned_tokens, keep_indices
