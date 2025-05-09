import torch
import time
import argparse

from models.quanta_vit import QuantaVisionTransformer
from models.backbone import VisionTransformer

def analyze():
    parser = argparse.ArgumentParser(description="Quanta-RT 模型分析脚本")
    parser.add_argument('--model_path', type=str, default='quanta_rt_model.pt', help='模型文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='使用设备 (cpu 或 cuda)')
    parser.add_argument('--iterations', type=int, default=100, help='测试推理速度的迭代次数')
    args = parser.parse_args()

    # 加载模型
    model = torch.load(args.model_path, map_location='cpu')
    model.to(args.device)
    model.eval()
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params} (可训练参数: {trainable_params})")
    # 准备 dummy 输入 (匹配模型输入大小)
    in_c = model.patch_embed.proj.in_channels if hasattr(model.patch_embed, 'proj') else 3
    H = model.patch_embed.img_height if hasattr(model.patch_embed, 'img_height') else 224
    W = model.patch_embed.img_width if hasattr(model.patch_embed, 'img_width') else H
    dummy_input = torch.randn(1, in_c, H, W).to(args.device)
    # 测量推理速度
    with torch.no_grad():
        # 预热
        model(dummy_input)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.iterations):
            model(dummy_input)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
    avg_time = (end - start) / args.iterations
    fps = float('inf') if avg_time == 0 else (1.0 / avg_time)
    if args.device == 'cuda':
        print(f"平均推理时间: {avg_time*1000:.2f} ms/张 (GPU)")
    else:
        print(f"平均推理时间: {avg_time*1000:.2f} ms/张 (CPU)")
    print(f"推理吞吐量: {fps:.2f} 张/秒")

if __name__ == "__main__":
    analyze()
