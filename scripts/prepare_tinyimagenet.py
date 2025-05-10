#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import zipfile
import shutil
from pathlib import Path

def prepare_tinyimagenet(data_dir='/content/drive/MyDrive/Quanta-RT/data'):
    """
    解压和处理 TinyImageNet 数据集。
    适用于 Google Colab 环境。
    
    Args:
        data_dir: 数据目录路径，包含tiny-imagenet-200.zip文件
                 默认为Google Drive中的路径
    """
    # 确保data_dir目录存在
    data_dir = Path(data_dir)
    
    print(f"正在使用数据目录: {data_dir}")
    
    if not data_dir.exists():
        print(f"错误: 数据目录 {data_dir} 不存在")
        print("请确保已正确挂载 Google Drive 并创建相应目录")
        return False
    
    zip_path = data_dir / 'tiny-imagenet-200.zip'
    extract_path = data_dir
    
    print(f"正在检查 {zip_path} 是否存在...")
    if not zip_path.exists():
        print(f"错误: 未找到 {zip_path} 文件")
        print("请确保 tiny-imagenet-200.zip 文件位于指定的数据目录中")
        return False
    
    # 解压缩文件
    print(f"正在解压 {zip_path} 到 {extract_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("解压完成")
    except Exception as e:
        print(f"解压失败: {e}")
        return False
    
    # 处理验证集目录结构
    val_dir = extract_path / 'tiny-imagenet-200' / 'val'
    val_img_dir = val_dir / 'images'
    annotations_file = val_dir / 'val_annotations.txt'
    
    if not val_img_dir.exists() or not annotations_file.exists():
        print("警告: 验证集目录结构不符合预期")
        print(f"请检查 {val_dir} 目录是否包含 images 文件夹和 val_annotations.txt 文件")
        return False
    
    print("正在处理验证集目录结构...")
    
    # 读取annotations文件，获取图像对应的类别
    img_to_class = {}
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                class_id = parts[1]
                img_to_class[img_name] = class_id
    
    # 为每个类别创建目录
    for class_id in set(img_to_class.values()):
        os.makedirs(val_dir / class_id, exist_ok=True)
    
    # 移动图像到对应的类别目录
    moved_count = 0
    for img_name, class_id in img_to_class.items():
        src = val_img_dir / img_name
        dst = val_dir / class_id / img_name
        if src.exists():
            try:
                shutil.move(str(src), str(dst))
                moved_count += 1
            except Exception as e:
                print(f"移动图像 {img_name} 失败: {e}")
    
    print(f"成功移动了 {moved_count}/{len(img_to_class)} 个验证图像")
    
    # 移动完成后删除images目录
    if val_img_dir.exists():
        remaining = list(val_img_dir.iterdir())
        if len(remaining) > 0:
            print(f"警告: images 目录中仍有 {len(remaining)} 个文件未处理")
        else:
            try:
                val_img_dir.rmdir()
                print("验证集 images 目录已清理")
            except Exception as e:
                print(f"清理 images 目录失败: {e}")
    
    # 处理训练集中的images子目录 (Tiny ImageNet 标准结构)
    train_dir = extract_path / 'tiny-imagenet-200' / 'train'
    class_dirs = list(train_dir.glob('n*'))
    
    if not class_dirs:
        print("警告: 训练集目录结构不符合预期")
        return False
    
    print(f"正在处理训练集 {len(class_dirs)} 个类别的 images 子目录...")
    processed_count = 0
    
    for class_dir in class_dirs:
        images_dir = class_dir / 'images'
        if images_dir.exists() and images_dir.is_dir():
            # 将图像从 class/images/ 移动到 class/
            image_files = list(images_dir.glob('*.JPEG'))
            for img_file in image_files:
                dst = class_dir / img_file.name
                try:
                    shutil.move(str(img_file), str(dst))
                except Exception as e:
                    print(f"移动 {img_file} 失败: {e}")
            
            # 移除空的images目录
            remaining = list(images_dir.iterdir())
            if not remaining:
                try:
                    images_dir.rmdir()
                    processed_count += 1
                except Exception as e:
                    print(f"删除 {images_dir} 失败: {e}")
    
    print(f"处理了 {processed_count}/{len(class_dirs)} 个类别的 images 子目录")
        
    print("\n数据集处理完成！可以使用以下命令进行训练:")
    print("python scripts/train.py --dataset tinyimagenet --data_dir /content/drive/MyDrive/Quanta-RT/data --img_size 64 --patch_size 8")
    
    return True

if __name__ == "__main__":
    # 默认使用Google Drive中的路径
    data_dir = '/content/drive/MyDrive/Quanta-RT/data'
    
    # 允许通过命令行参数指定不同的数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    # 执行数据集准备
    prepare_tinyimagenet(data_dir)