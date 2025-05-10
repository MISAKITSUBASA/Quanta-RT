import os
from torchvision import datasets, transforms

def download_data(dataset_name, data_dir):
    """
    下载指定数据集并返回训练集和测试集 Dataset 对象。
    支持: CIFAR10, CIFAR100, MNIST, TinyImageNet 等。对于 ImageNet，需要手动下载数据。
    """
    dataset_name = dataset_name.lower()
    os.makedirs(data_dir, exist_ok=True)
    if dataset_name == 'cifar10':
        # CIFAR10 数据集
        transform = transforms.ToTensor()  # 仅转换为张量，归一化在训练中进行
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
        # ImageNet 需要用户自行下载数据
        print(f"请先从ImageNet官网获取数据，并将其解压到目录: {data_dir}")
        return None, None
    elif dataset_name == 'tinyimagenet':
        # TinyImageNet 数据集处理
        train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
        val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
        
        # 检查目录是否存在
        if not os.path.exists(train_dir):
            print(f"TinyImageNet 训练数据目录未找到: {train_dir}")
            print("请确保已经解压 tiny-imagenet-200.zip 文件")
            return None, None
            
        # 检查验证集目录结构是否已处理
        if os.path.exists(os.path.join(val_dir, 'images')):
            print("请先处理 TinyImageNet 验证集目录结构")
            print("验证集图像需要按类别组织到子文件夹中")
            return None, None
            
        # 加载数据集
        transform = transforms.ToTensor()
        try:
            train_set = datasets.ImageFolder(root=train_dir, transform=transform)
            test_set = datasets.ImageFolder(root=val_dir, transform=transform)
            return train_set, test_set
        except Exception as e:
            print(f"加载 TinyImageNet 数据集失败: {e}")
            return None, None
    else:
        print(f"未支持的数据集: {dataset_name}")
        return None, None

if __name__ == "__main__":
    # 脚本直接运行示例: 下载 CIFAR10 数据集
    train_set, test_set = download_data('cifar10', './data')
    print("训练集大小:", len(train_set) if train_set else 0)
    print("测试集大小:", len(test_set) if test_set else 0)
