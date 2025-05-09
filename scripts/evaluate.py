import torch
from torch.utils.data import DataLoader
import argparse

from models.quanta_vit import QuantaVisionTransformer  # 确保定义模型类
from scripts.download_dataset import download_data
from torchvision import transforms

def evaluate():
    parser = argparse.ArgumentParser(description="Quanta-RT 模型评估脚本")
    parser.add_argument('--dataset', type=str, default='cifar10', help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--model_path', type=str, default='quanta_rt_model.pt', help='模型文件路径')
    args = parser.parse_args()

    # 加载数据集 (这里只需要测试集)
    _, test_set = download_data(args.dataset, args.data_dir)
    if test_set is None:
        print(f"无法加载数据集 {args.dataset}")
        return

    # 设置与训练相同的归一化
    if args.dataset.lower() in ['cifar10', 'cifar100']:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    elif args.dataset.lower() == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5]
        std = [0.5]
    test_set.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # 加载模型
    try:
        model = torch.load(args.model_path, map_location='cpu')
    except Exception as e:
        print("模型加载失败:", e)
        return
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 测试集上评估准确率
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
    print(f"Test Accuracy on {args.dataset}: {acc:.2f}%")

if __name__ == "__main__":
    evaluate()
