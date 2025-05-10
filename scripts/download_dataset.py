import os
from torchvision import datasets, transforms
import glob

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
        
        # 诊断目录结构问题
        print(f"[DEBUG] 诊断训练目录结构: {train_dir}")
        # 检查一些子目录
        subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        if not subdirs:
            print(f"错误: 训练目录 {train_dir} 不包含任何子目录!")
            return None, None
        
        print(f"[DEBUG] 找到 {len(subdirs)} 个类别子目录")
        # 检查前5个子目录的内容
        for i, subdir in enumerate(subdirs[:5]):
            subdir_path = os.path.join(train_dir, subdir)
            files = glob.glob(os.path.join(subdir_path, "**/*.*"), recursive=True)
            valid_exts = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
            valid_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
            print(f"[DEBUG] 类别 {subdir}: 共 {len(files)} 个文件, 其中 {len(valid_files)} 个有效图像")
            if not valid_files and i == 0:  # 只显示第一个空子目录的详细内容
                print(f"[DEBUG] 子目录 {subdir} 内容: {os.listdir(subdir_path)[:10]} ...")
        
        # 检查图像放置位置是否有问题 (图像应该直接放在类别子目录下或者特定模式的子子目录下)
        problem_subdirs = []
        for subdir in subdirs[:20]:  # 只检查前20个子目录，避免输出过多
            subdir_path = os.path.join(train_dir, subdir)
            # 检查是否存在images子目录 (Tiny ImageNet官方格式)
            images_subdir = os.path.join(subdir_path, 'images')
            if os.path.isdir(images_subdir):
                print(f"[DEBUG] 检测到 '{subdir}/images/' 子目录，这可能是Tiny ImageNet原始格式")
                problem_subdirs.append(subdir)
        
        if problem_subdirs:
            print("[WARN] 检测到以下类别文件夹内有 'images' 子目录，可能需要调整结构适应ImageFolder:")
            print(", ".join(problem_subdirs))
            print("[WARN] 建议执行以下调整命令:")
            print("cd [数据目录] && for cls in */; do if [ -d \"$cls/images\" ]; then mv \"$cls/images\"/* \"$cls/\"; fi; done")
        
        # 尝试加载数据集
        train_set = None
        try:
            transform = transforms.ToTensor() 
            train_set = datasets.ImageFolder(root=train_dir, transform=transform)
            print(f"[DEBUG] 成功加载训练集，共 {len(train_set)} 样本")
        except Exception as e:
            print(f"错误: 加载 Tiny ImageNet 训练集失败: {e}")
            
            # 额外尝试：如果检测到images子目录问题，尝试自动修复
            if problem_subdirs and "Found no valid file" in str(e):
                print("[DEBUG] 尝试自动修复目录结构...")
                
                # 创建临时修复目录
                fixed_train_dir = train_dir + "_fixed"
                if not os.path.exists(fixed_train_dir):
                    os.makedirs(fixed_train_dir)
                
                # 复制并修复目录结构
                import shutil
                fixed_any = False
                for subdir in subdirs:
                    src_dir = os.path.join(train_dir, subdir)
                    dst_dir = os.path.join(fixed_train_dir, subdir)
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    
                    # 如果有images子目录，从那里复制
                    images_dir = os.path.join(src_dir, 'images')
                    if os.path.isdir(images_dir):
                        image_files = []
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG']:  # 支持的扩展名
                            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
                        
                        if image_files:
                            fixed_any = True
                            for img in image_files:
                                shutil.copy2(img, dst_dir)
                            print(f"[DEBUG] 修复类别 {subdir}: 从images子目录复制了 {len(image_files)} 个文件")
                    else:
                        # 否则，直接从类别根目录复制
                        image_files = []
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG']:
                            image_files.extend(glob.glob(os.path.join(src_dir, ext)))
                        
                        if image_files:
                            for img in image_files:
                                shutil.copy2(img, dst_dir)
                
                # 如果成功修复了任何目录，尝试加载修复后的数据集
                if fixed_any:
                    print(f"[DEBUG] 尝试从修复的目录加载: {fixed_train_dir}")
                    try:
                        train_set = datasets.ImageFolder(root=fixed_train_dir, transform=transform)
                        print(f"[SUCCESS] 从修复的目录成功加载训练集，共 {len(train_set)} 样本")
                        # 更新训练目录为修复后的目录
                        train_dir = fixed_train_dir
                    except Exception as e2:
                        print(f"[ERROR] 修复后仍然无法加载训练集: {e2}")
            
            if train_set is None:
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
