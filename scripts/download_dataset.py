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
        
        train_dir = os.path.join(abs_data_dir, 'tinyimagenet', 'train')
        val_processed_dir = os.path.join(abs_data_dir, 'tinyimagenet', 'val')

        print(f"[DEBUG] Expected Tiny ImageNet train_dir: {train_dir}")
        print(f"[DEBUG] Expected Tiny ImageNet val_processed_dir (for test_set): {val_processed_dir}")

        if not os.path.exists(train_dir):
            print(f"错误: Tiny ImageNet 训练数据目录未找到: {train_dir}")
            print(f"请确保 tiny-imagenet-200/train 存在于 '{abs_data_dir}' 下。")
            print("并且 --data_dir 参数指向 'tiny-imagenet-200' 的父目录。")
            return None, None
        
        train_set = None
        try:
            transform = transforms.ToTensor() 
            train_set = datasets.ImageFolder(root=train_dir, transform=transform)
            print(f"[DEBUG] Successfully loaded train_set from {train_dir}. Size: {len(train_set)}")
        except Exception as e:
            print(f"错误: 加载 Tiny ImageNet 训练集失败从 '{train_dir}': {e}")
            return None, None

        test_set = None
        if not os.path.exists(val_processed_dir):
            print(f"警告: Tiny ImageNet 处理后的验证数据目录 {val_processed_dir} 未找到。")
            print("测试集将为 None。如果需要验证/测试，请确保已运行预处理脚本并将验证图像分类。")
        else:
            try:
                transform = transforms.ToTensor()
                test_set = datasets.ImageFolder(root=val_processed_dir, transform=transform)
                print(f"[DEBUG] Successfully loaded test_set from {val_processed_dir}. Size: {len(test_set)}")
            except Exception as e:
                print(f"错误: 加载 Tiny ImageNet 验证集 (作为测试集) 失败从 '{val_processed_dir}': {e}")

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
