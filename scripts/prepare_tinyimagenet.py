#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import zipfile
import shutil
from pathlib import Path

def prepare_tinyimagenet(data_dir='./data'):
    """
    解压和处理 TinyImageNet 数据集。
    
    Args:
        data_dir: 数据目录路径，包含tiny-imagenet-200.zip文件
    """
    # 确保data_dir目录存在
    data_dir = Path(data_dir)
    
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
        (val_dir / class_id).mkdir(exist_ok=True)
    
    # 移动图像到对应的类别目录
    for img_name, class_id in img_to_class.items():
        src = val_img_dir / img_name
        dst = val_dir / class_id / img_name
        if src.exists():
            shutil.move(str(src), str(dst))
    
    # 移动完成后删除images目录
    if val_img_dir.exists() and len(list(val_img_dir.iterdir())) == 0:
        val_img_dir.rmdir()
        print("验证集处理完成")
    else:
        print("警告: 验证集处理可能不完整")
    
    # 检查训练集目录结构 (验证 train/n**/images/ 目录是否包含图片文件)
    train_dir = extract_path / 'tiny-imagenet-200' / 'train'
    class_dirs = list(train_dir.glob('n*'))
    
    if not class_dirs:
        print("警告: 训练集目录结构不符合预期")
        return False
        
    print("数据集处理完成！可以使用以下命令进行训练:")
    print("python scripts/train.py --dataset tinyimagenet --data_dir ./data --img_size 64 --patch_size 8")
    
    return True

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data'
    prepare_tinyimagenet(data_dir)