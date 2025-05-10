import os
from torchvision import datasets, transforms

def download_data(dataset_name, data_dir):
    """
    下载指定数据集并返回训练集和测试集 Dataset 对象。
    支持: CIFAR10, CIFAR100, MNIST 等。对于 ImageNet 和 Tiny ImageNet，需要手动下载数据。
    """
    print(f"[DEBUG] download_data received dataset_name: '{dataset_name}', data_dir: '{data_dir}'")
    original_dataset_name = dataset_name  # Store original for user messages
    dataset_name_lower = dataset_name.lower()
    print(f"[DEBUG] lowercased dataset_name: '{dataset_name_lower}'")
    
    # Ensure absolute path for data_dir for clearer debug messages
    abs_data_dir = os.path.abspath(data_dir)
    print(f"[DEBUG] Absolute data_dir: '{abs_data_dir}'")
    os.makedirs(abs_data_dir, exist_ok=True)

    if dataset_name_lower == 'cifar10':
        print("[DEBUG] Matched 'cifar10'")
        transform = transforms.ToTensor()
        train_set = datasets.CIFAR10(root=abs_data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=abs_data_dir, train=False, download=True, transform=transform)
        return train_set, test_set
    elif dataset_name_lower == 'cifar100':
        print("[DEBUG] Matched 'cifar100'")
        transform = transforms.ToTensor()
        train_set = datasets.CIFAR100(root=abs_data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root=abs_data_dir, train=False, download=True, transform=transform)
        return train_set, test_set
    elif dataset_name_lower == 'mnist':
        print("[DEBUG] Matched 'mnist'")
        transform = transforms.ToTensor()
        train_set = datasets.MNIST(root=abs_data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=abs_data_dir, train=False, download=True, transform=transform)
        return train_set, test_set
    elif dataset_name_lower == 'imagenet':
        print("[DEBUG] Matched 'imagenet'")
        print(f"请先从ImageNet官网获取数据，并将其解压到目录: {abs_data_dir}")
        return None, None
    elif dataset_name_lower == 'tinyimagenet':
        print("[DEBUG] Matched 'tinyimagenet'")
        
        # 尝试多种可能的路径格式
        possible_paths = [
            # 标准官方路径结构
            os.path.join(abs_data_dir, 'tiny-imagenet-200', 'train'),
            # 自定义路径结构（如之前尝试的）
            os.path.join(abs_data_dir, 'tinyimagenet', 'train'),
            # 如果直接将train和val放在data_dir下
            os.path.join(abs_data_dir, 'train'),
        ]
        
        train_dir = None
        # 查找存在的训练目录
        for path in possible_paths:
            print(f"[DEBUG] 检查训练目录是否存在: {path}")
            if os.path.exists(path):
                train_dir = path
                print(f"[DEBUG] 找到有效的训练目录: {train_dir}")
                break
        
        # 类似地，尝试多种可能的验证集路径
        possible_val_paths = [
            os.path.join(abs_data_dir, 'tiny-imagenet-200', 'val'),
            os.path.join(abs_data_dir, 'tiny-imagenet-200', 'val_processed'),
            os.path.join(abs_data_dir, 'tinyimagenet', 'val'),
            os.path.join(abs_data_dir, 'val'),
        ]
        
        val_dir = None
        # 查找存在的验证目录
        for path in possible_val_paths:
            print(f"[DEBUG] 检查验证目录是否存在: {path}")
            if os.path.exists(path):
                val_dir = path
                print(f"[DEBUG] 找到有效的验证目录: {val_dir}")
                break
        
        # 如果没有找到训练目录
        if train_dir is None:
            paths_str = "\n - ".join(possible_paths)
            print(f"错误: 无法找到 Tiny ImageNet 训练目录。已检查以下路径:\n - {paths_str}")
            print("请确保您已下载并解压 Tiny ImageNet 数据集，并且 --data_dir 参数指向正确的父目录。")
            return None, None
        
        train_set = None
        try:
            transform = transforms.ToTensor() 
            train_set = datasets.ImageFolder(root=train_dir, transform=transform)
            print(f"[DEBUG] 成功加载训练集，共 {len(train_set)} 样本")
        except Exception as e:
            print(f"错误: 加载 Tiny ImageNet 训练集失败: {e}")
            return None, None

        test_set = None
        if val_dir is None:
            print(f"警告: 未找到 Tiny ImageNet 验证目录。测试集将为 None。")
        else:
            try:
                transform = transforms.ToTensor()
                test_set = datasets.ImageFolder(root=val_dir, transform=transform)
                print(f"[DEBUG] 成功加载验证集，共 {len(test_set)} 样本")
            except Exception as e:
                print(f"错误: 加载 Tiny ImageNet 验证集失败: {e}")
                # 验证集加载失败不阻止训练，但会输出警告

        if train_set is None:
             print("错误: Tiny ImageNet 训练集最终未能加载。")
             return None, None

        return train_set, test_set
    else:
        print(f"[DEBUG] No dataset match found. Falling into final else for input: '{original_dataset_name}' (lowercased: '{dataset_name_lower}')")
        print(f"未支持的数据集: {original_dataset_name}")
        return None, None

if __name__ == "__main__":
    print("[DEBUG] Running download_dataset.py directly for testing.")
    train_set, test_set = download_data('cifar10', './data_cifar_test')
    print("CIFAR10 训练集大小:", len(train_set) if train_set else 0)
    print("CIFAR10 测试集大小:", len(test_set) if test_set else 0)
