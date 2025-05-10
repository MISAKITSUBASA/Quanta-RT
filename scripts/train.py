import sys
import os
import time
from tqdm import tqdm
import datetime

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse

from models.quanta_vit import QuantaVisionTransformer
from scripts.download_dataset import download_data
from torchvision import transforms

# 尝试导入logger模块（如果存在）
try:
    from utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

def train():
    parser = argparse.ArgumentParser(description="Quanta-RT 模型训练脚本")
    parser.add_argument('--dataset', type=str, default='cifar10', help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据存储路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练周期数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--prune_ratio', type=float, default=0.0, help='token 剪枝保留比例 (0表示不剪枝)')
    parser.add_argument('--quant_bits', type=int, default=0, help='量化位宽 (0表示不量化)')
    parser.add_argument('--embed_dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--depth', type=int, default=6, help='Transformer 层数')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--patch_size', type=int, default=16, help='图像补丁大小')
    parser.add_argument('--img_size', type=int, default=None, help='输入图像尺寸 (默认自动推断)')
    args = parser.parse_args()

    # 下载并加载数据集
    train_set, test_set = download_data(args.dataset, args.data_dir)
    if train_set is None or test_set is None:
        print(f"数据集 {args.dataset} 无法加载，请检查路径或名称。")
        return

    # 若未指定 img_size，则根据数据集选择默认值
    if args.img_size is None:
        if args.dataset.lower() in ['cifar10', 'cifar100']:
            args.img_size = 32
        elif args.dataset.lower() == 'mnist':
            args.img_size = 28
        else:
            args.img_size = 224

    # 定义数据增强和归一化
    if args.dataset.lower() in ['cifar10', 'cifar100']:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        train_transform = transforms.Compose([
            transforms.RandomCrop(args.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif args.dataset.lower() == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        mean = [0.5]
        std = [0.5]
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    # 更新数据集的 transform
    train_set.transform = train_transform
    test_set.transform = test_transform

    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    model = QuantaVisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=1 if args.dataset.lower() == 'mnist' else 3,
        num_classes=len(train_set.classes) if hasattr(train_set, 'classes') else 10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        prune_ratio=args.prune_ratio,
        quant_bits=args.quant_bits
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 初始化训练参数
    start_time = time.time()
    best_acc = 0.0
    history = {
        'train_loss': [],
        'val_acc': [],
        'epoch_times': []
    }

    # 如果有剪枝或量化，记录相关指标
    if args.prune_ratio > 0.0 or args.quant_bits > 0:
        history['token_keep_ratio'] = []
        history['bitwidth_avg'] = []

    print("\n" + "="*50)
    print(f"开始训练 Quanta-RT 模型 ({args.dataset.upper()})")
    print(f"设备: {device}")
    print(f"批次大小: {args.batch_size}, 学习率: {args.learning_rate}")
    print(f"训练集大小: {len(train_set)}, 验证集大小: {len(test_set)}")
    if args.prune_ratio > 0.0:
        print(f"Token 剪枝保留比例: {args.prune_ratio}")
    if args.quant_bits > 0:
        print(f"量化位宽: {args.quant_bits} 位")
    print("="*50 + "\n")

    # 训练循环
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        # 使用tqdm创建进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", 
                         leave=False, ncols=100)
        
        batch_count = 0
        for images, labels in train_bar:
            batch_count += 1
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            running_loss += loss.item()
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{running_loss/batch_count:.4f}"
            })
        
        # 计算平均训练损失
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # 验证模型在测试集上的性能
        model.eval()
        correct = 0
        total = 0
        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]", 
                      leave=False, ncols=100)
        
        with torch.no_grad():
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新验证进度条
                current_acc = 100.0 * correct / total
                val_bar.set_postfix({'acc': f"{current_acc:.2f}%"})
        
        # 计算验证准确率
        acc = 100.0 * correct / total if total > 0 else 0.0
        history['val_acc'].append(acc)
        
        # 记录剪枝和量化指标（如果有）
        prune_info = ""
        quant_info = ""
        if hasattr(model, 'get_keep_ratio') and args.prune_ratio > 0.0:
            keep_ratio = model.get_keep_ratio()
            history['token_keep_ratio'].append(keep_ratio)
            prune_info = f", 保留比例: {keep_ratio:.2f}"
            
        if hasattr(model, 'get_avg_bitwidth') and args.quant_bits > 0:
            avg_bits = model.get_avg_bitwidth()
            history['bitwidth_avg'].append(avg_bits)
            quant_info = f", 平均位宽: {avg_bits:.2f}"
        
        # 记录每个epoch的耗时
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        # 更新最佳模型
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            torch.save(model.state_dict(), "quanta_rt_best.pt")
        
        # 打印本轮结果
        print(f"Epoch {epoch+1}/{args.epochs} - 耗时: {epoch_time:.1f}秒")
        print(f"  训练损失: {avg_loss:.4f}, 验证准确率: {acc:.2f}%{prune_info}{quant_info}")
        if is_best:
            print(f"  【新的最佳模型】验证准确率: {acc:.2f}%")
            
        # 每5个epoch或训练结束时保存检查点
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }
            torch.save(checkpoint, f"checkpoint_epoch{epoch+1}.pt")
            print(f"  保存检查点: checkpoint_epoch{epoch+1}.pt")
            
    # 计算总训练时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"训练完成！总耗时: {int(hours)}小时 {int(minutes)}分 {seconds:.1f}秒")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"最终验证准确率: {acc:.2f}%")
    
    # 保存最终模型
    torch.save(model, "quanta_rt_model.pt")
    print("最终模型已保存至 quanta_rt_model.pt")
    print("最佳模型已保存至 quanta_rt_best.pt")
    print("="*50)
    
    # 返回训练历史记录（可用于分析和可视化）
    return history

if __name__ == "__main__":
    train()
