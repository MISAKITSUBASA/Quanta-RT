import torch
import argparse
import numpy as np
from PIL import Image

from models.quanta_vit import QuantaVisionTransformer

def visualize():
    parser = argparse.ArgumentParser(description="Quanta-RT 模型可视化脚本")
    parser.add_argument('--model_path', type=str, default='quanta_rt_model.pt', help='模型文件路径')
    parser.add_argument('--image_path', type=str, default=None, help='输入图像路径 (可选)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='若未提供图像则使用的数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集目录')
    parser.add_argument('--output', type=str, default='visualization.png', help='输出可视化图像文件名')
    args = parser.parse_args()

    # 加载训练好的模型
    model = torch.load(args.model_path, map_location='cpu')
    model.eval()

    # 获取一张示例图像
    if args.image_path:
        # 从路径加载指定图像
        image = Image.open(args.image_path).convert('RGB')
        # 调整图像尺寸与模型输入匹配
        img_h = model.patch_embed.img_height if hasattr(model.patch_embed, 'img_height') else 224
        img_w = model.patch_embed.img_width if hasattr(model.patch_embed, 'img_width') else 224
        image_resized = image.resize((img_w, img_h))
    else:
        # 未提供图像路径，从数据集加载第一张测试图像
        from scripts.download_dataset import download_data
        _, test_set = download_data(args.dataset, args.data_dir)
        if test_set is None or len(test_set) == 0:
            print("无法获取示例图像")
            return
        # 对于有 .data 属性的数据集 (如 CIFAR/MNIST)，直接读取原始图像数据
        image = None
        if hasattr(test_set, 'data'):
            img_arr = test_set.data[0]
            # 将 numpy 数组或 Tensor 转换为 PIL 图像
            if hasattr(img_arr, 'numpy'):
                img_arr = img_arr.numpy()
            if img_arr is not None:
                image = Image.fromarray(img_arr)
        if image is None:
            # 对于 ImageFolder 等数据集，通过索引获取 (假定 transform=None 时返回 PIL)
            sample = test_set[0]
            image = sample[0] if isinstance(sample, tuple) else sample
        image = image.convert('RGB')
        img_h = model.patch_embed.img_height if hasattr(model.patch_embed, 'img_height') else 224
        img_w = model.patch_embed.img_width if hasattr(model.patch_embed, 'img_width') else 224
        image_resized = image.resize((img_w, img_h))

    # 将图像转换为张量并归一化 (使用与训练相同的均值和方差)
    from torchvision import transforms
    in_c = model.patch_embed.proj.in_channels if hasattr(model.patch_embed, 'proj') else 3
    if in_c == 1:
        # 模型期望单通道输入
        image_resized = image_resized.convert('L')
        mean = [0.5]; std = [0.5]
    else:
        if args.dataset.lower() in ['cifar10', 'cifar100']:
            mean = [0.4914, 0.4822, 0.4465]; std = [0.2470, 0.2435, 0.2616]
        else:
            mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img_tensor = transform(image_resized).unsqueeze(0)  # [1, C, H, W]

    # 前向传播 (无需关心输出，只为获取剪枝信息)
    with torch.no_grad():
        _ = model(img_tensor)
    # 获取模型保存的保留 token 索引
    if hasattr(model, 'last_keep_indices'):
        keep_indices = model.last_keep_indices
        if isinstance(keep_indices, list):
            keep_indices = keep_indices[0]  # 仅单张图像，取第0个
    else:
        keep_indices = None
        print("模型未产生保留索引 (可能未启用剪枝)")

    # 构建可视化结果图像
    vis_img = np.array(image_resized).copy()
    if keep_indices is not None:
        # 计算 patch 网格大小和patch尺寸
        if hasattr(model.patch_embed, 'grid_size'):
            grid_h, grid_w = model.patch_embed.grid_size
        else:
            patch = model.patch_embed.patch_size if hasattr(model.patch_embed, 'patch_size') else 16
            grid_h = img_h // patch
            grid_w = img_w // patch
        patch_size = model.patch_embed.patch_size if hasattr(model.patch_embed, 'patch_size') else (img_h // grid_h)
        # 计算被移除的 patch 索引集合 (转换为0-based的patch索引)
        total_patches = grid_h * grid_w
        kept_patches = [(idx - 1) for idx in keep_indices if idx != 0]
        removed_patches = [p for p in range(total_patches) if p not in kept_patches]
        # 将移除的 patch 区域变暗以指示被剪枝
        for p in removed_patches:
            y = p // grid_w
            x = p % grid_w
            top = y * patch_size
            left = x * patch_size
            bottom = top + patch_size
            right = left + patch_size
            # 以灰色 (降低亮度) 填充该 patch 区域
            vis_img[top:bottom, left:right, :] = (vis_img[top:bottom, left:right, :] * 0.2).astype(np.uint8)
    # 保存可视化图像
    vis_image = Image.fromarray(vis_img)
    vis_image.save(args.output)
    print("可视化结果已保存:", args.output)

if __name__ == "__main__":
    visualize()
