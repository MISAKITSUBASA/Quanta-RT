import os
from torchvision import datasets, transforms

def download_data(dataset_name, data_dir):
    """
    下载指定数据集并返回训练集和测试集 Dataset 对象。
    支持: CIFAR10, CIFAR100, MNIST 等。对于 ImageNet 和 Tiny ImageNet，需要手动下载数据。
    """
    dataset_name = dataset_name.lower()
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name == 'cifar10':
        transform = transforms.ToTensor()
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set
    elif dataset_name == 'cifar100':
        transform = transforms.ToTensor()
        train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set
    elif dataset_name == 'mnist':
        transform = transforms.ToTensor()
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set
    elif dataset_name == 'imagenet':
        print(f"请先从ImageNet官网获取数据，并将其解压到目录: {data_dir}")
        return None, None
    elif dataset_name == 'tinyimagenet':
        # 首先尝试标准路径
        train_dir = os.path.join(data_dir, 'tinyimagenet', 'train')
        val_dir = os.path.join(data_dir, 'tinyimagenet', 'val')
        
        # 如果标准路径不存在，尝试官方路径结构
        if not os.path.exists(train_dir):
            train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
        
        if not os.path.exists(val_dir):
            val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
            # 尝试备用验证目录
            if not os.path.exists(val_dir):
                val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val_processed')
        
        # 检查训练目录是否存在
        if not os.path.exists(train_dir):
            print(f"错误: Tiny ImageNet 训练数据目录未找到: {train_dir}")
            print("请确保数据已正确下载和解压。")
            return None, None
        
        # 加载训练集
        try:
            transform = transforms.ToTensor() 
            train_set = datasets.ImageFolder(root=train_dir, transform=transform)
        except Exception as e:
            print(f"错误: 加载 Tiny ImageNet 训练集失败: {e}")
            return None, None

        # 加载验证集（如果存在）
        test_set = None
        if os.path.exists(val_dir):
            try:
                test_set = datasets.ImageFolder(root=val_dir, transform=transform)
            except Exception as e:
                print(f"警告: 加载 Tiny ImageNet 验证集失败: {e}")
                # 验证集加载失败不阻止训练

        return train_set, test_set
    else:
        print(f"未支持的数据集: {dataset_name}")
        return None, None

if __name__ == "__main__":
    train_set, test_set = download_data('cifar10', './data')
    print("CIFAR10 训练集大小:", len(train_set) if train_set else 0)
    print("CIFAR10 测试集大小:", len(test_set) if test_set else 0)
