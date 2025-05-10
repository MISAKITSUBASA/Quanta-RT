#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import zipfile
import shutil
from pathlib import Path

def prepare_tinyimagenet(data_dir='/content/drive/MyDrive/Quanta-RT/data'):
    """
    解压和处理 TinyImageNet 数据集，包括训练集和验证集。
    适用于 Google Colab 环境。
    
    Args:
        data_dir: 数据目录路径，应包含 tiny-imagenet-200.zip 文件。
                 默认为Google Drive中的路径。
    """
    data_dir = Path(data_dir)
    print(f"正在使用数据目录: {data_dir}")

    if not data_dir.exists():
        print(f"错误: 数据目录 {data_dir} 不存在")
        print("请确保已正确挂载 Google Drive 并创建相应目录")
        return False

    zip_path = data_dir / 'tiny-imagenet-200.zip'
    extract_path = data_dir # 解压到数据目录本身
    dataset_root = extract_path / 'tiny-imagenet-200'

    print(f"正在检查 {zip_path} 是否存在...")
    if not zip_path.exists():
        print(f"错误: 未找到 {zip_path} 文件")
        print(f"请确保 {zip_path.name} 文件位于 {data_dir} 目录中")
        return False

    # 解压缩文件
    if not dataset_root.exists():
        print(f"正在解压 {zip_path} 到 {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("解压完成")
        except Exception as e:
            print(f"解压失败: {e}")
            return False
    else:
        print(f"数据集目录 {dataset_root} 已存在，跳过解压。")

    # 1. 处理验证集 (val set)
    val_dir = dataset_root / 'val'
    val_img_dir = val_dir / 'images'
    annotations_file = val_dir / 'val_annotations.txt'

    if not val_dir.exists():
        print(f"错误: 验证集目录 {val_dir} 未找到。")
        return False

    if val_img_dir.exists() and annotations_file.exists():
        print(f"正在处理验证集目录结构 ({val_dir})...")
        img_to_class = {}
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name, class_id = parts[0], parts[1]
                    img_to_class[img_name] = class_id

        for class_id in set(img_to_class.values()):
            (val_dir / class_id).mkdir(exist_ok=True)

        moved_count = 0
        for img_name, class_id in img_to_class.items():
            src = val_img_dir / img_name
            dst = val_dir / class_id / img_name
            if src.exists():
                try:
                    shutil.move(str(src), str(dst))
                    moved_count += 1
                except Exception as e:
                    print(f"移动验证集图像 {img_name} 失败: {e}")
        print(f"成功移动了 {moved_count}/{len(img_to_class)} 个验证集图像")

        if val_img_dir.exists():
            try:
                # 确保 images 目录为空再删除
                if not any(val_img_dir.iterdir()): 
                    val_img_dir.rmdir()
                    print(f"原始验证集 images 目录 ({val_img_dir}) 已清理")
                else:
                    print(f"警告: 原始验证集 images 目录 ({val_img_dir}) 不为空，未删除。") 
            except Exception as e:
                print(f"清理原始验证集 images 目录失败: {e}")
        print("验证集处理完成。")
    elif not val_img_dir.exists() and not any((val_dir / c).is_dir() for c in os.listdir(val_dir) if (val_dir / c).is_dir()):
        print(f"错误: 验证集目录 {val_dir} 结构不符合预期，既没有 images 子目录，也没有类别子目录。")
        return False
    else:
        print(f"验证集目录 ({val_dir}) 似乎已处理或结构不标准，跳过处理。")

    # 2. 处理训练集 (train set)
    train_dir = dataset_root / 'train'
    if not train_dir.exists():
        print(f"错误: 训练集目录 {train_dir} 未找到。")
        return False

    print(f"正在处理训练集目录结构 ({train_dir})...")
    processed_train_classes = 0
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith('n')]

    for class_dir in class_dirs:
        images_subdir = class_dir / 'images'
        if images_subdir.exists() and images_subdir.is_dir():
            # 将图像从 class/images/ 移动到 class/
            image_files = list(images_subdir.glob('*.JPEG')) # TinyImageNet uses .JPEG
            moved_class_images = 0
            for img_file in image_files:
                dst = class_dir / img_file.name
                try:
                    shutil.move(str(img_file), str(dst))
                    moved_class_images +=1
                except Exception as e:
                    print(f"移动训练集图像 {img_file} 失败: {e}")
            
            if moved_class_images > 0:
                 print(f"  处理类别 {class_dir.name}: 移动了 {moved_class_images} 张图片。")

            # 移除空的images目录
            try:
                if not any(images_subdir.iterdir()):
                    images_subdir.rmdir()
                else:
                    print(f"  警告: 训练集类别 {class_dir.name} 的 images 子目录不为空，未删除。")
            except Exception as e:
                print(f"  删除训练集类别 {class_dir.name} 的 images 子目录失败: {e}")
            processed_train_classes += 1
    
    if processed_train_classes > 0:
        print(f"处理了 {processed_train_classes}/{len(class_dirs)} 个训练集类别的 images 子目录。")
    else:
        print("训练集目录结构似乎已处理或无需处理。")

    print("\n数据集准备脚本执行完毕。")
    print(f"请确保 {train_dir} 和 {val_dir} 目录结构正确后再进行训练。")
    print(f"预期的训练命令 (根据您的Colab路径调整data_dir):")
    print(f"python scripts/train.py --dataset tinyimagenet --data_dir {data_dir} --img_size 64 --patch_size 8")
    return True

if __name__ == "__main__":
    # 默认使用Google Drive中的路径，与函数默认值一致
    default_colab_data_dir = '/content/drive/MyDrive/Quanta-RT/data'
    
    # 允许通过命令行参数指定不同的数据目录
    # 如果在本地运行，可以传入如 './data'
    data_directory = sys.argv[1] if len(sys.argv) > 1 else default_colab_data_dir
    
    prepare_tinyimagenet(data_directory)