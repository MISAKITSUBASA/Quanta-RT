import torch
import argparse
import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from models.quanta_vit import QuantaVisionTransformer
from models.token_quant import quantize_model_weights

def export():
    parser = argparse.ArgumentParser(description="Quanta-RT 模型导出脚本")
    parser.add_argument('--model_path', type=str, default='quanta_rt_model.pt', help='模型文件路径')
    parser.add_argument('--out_path', type=str, default='quanta_rt_model_quantized.onnx', help='导出文件路径')
    parser.add_argument('--bits', type=int, default=8, help='量化位宽')
    args = parser.parse_args()

    # 加载模型
    try:
        model = torch.load(args.model_path, map_location='cpu')
    except Exception as e:
        print("模型加载失败:", e)
        return
    model.eval()
    # 模型权重量化 (如果指定 bits<32)
    if 0 < args.bits < 32:
        quantize_model_weights(model, bits=args.bits)
        print(f"模型权重已量化为 {args.bits} 比特.")
    else:
        print(f"未应用模型权重量化 (bits = {args.bits}).")
    # 准备 dummy 输入 (与模型输入尺寸匹配)
    in_c = model.patch_embed.proj.in_channels if hasattr(model.patch_embed, 'proj') else 3
    H = model.patch_embed.img_height if hasattr(model.patch_embed, 'img_height') else 224
    W = model.patch_embed.img_width if hasattr(model.patch_embed, 'img_width') else H
    dummy_input = torch.randn(1, in_c, H, W)
    # 导出模型为 ONNX 文件
    try:
        torch.onnx.export(model, dummy_input, args.out_path,
                          input_names=['input'], output_names=['output'], opset_version=11)
        print("模型已导出为 ONNX 文件:", args.out_path)
    except Exception as e:
        print("ONNX 导出失败:", e)
        # 尝试导出为 TorchScript
        try:
            scripted = torch.jit.trace(model, dummy_input)
            ts_path = args.out_path.replace('.onnx', '.pt')
            scripted.save(ts_path)
            print("模型已导出为 TorchScript 文件:", ts_path)
        except Exception as e2:
            print("模型导出失败:", e2)

if __name__ == "__main__":
    export()
