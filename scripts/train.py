import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse

from models.quanta_vit import QuantaVisionTransformer
from scripts.download_dataset import download_data
from torchvision import transforms

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

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        # 验证模型在测试集上的性能
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"Validation Accuracy: {acc:.2f}%")
    # 保存训练好的模型
    torch.save(model, "quanta_rt_model.pt")
    print("模型已保存至 quanta_rt_model.pt")

if __name__ == "__main__":
    train()
