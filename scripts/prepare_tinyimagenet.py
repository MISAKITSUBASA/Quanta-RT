#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path

def process_tinyimagenet_validation_set(val_dir):
    """
    处理 TinyImageNet 验证集目录结构，将图像按类别组织到子文件夹中。

    Args:
        val_dir (str): 验证集目录路径。
    """
    val_dir = Path(val_dir)
    val_img_dir = val_dir / 'images'
    annotations_file = val_dir / 'val_annotations.txt'

    if not val_img_dir.exists() or not annotations_file.exists():
        print("错误: 验证集目录结构不完整，缺少 images 文件夹或 val_annotations.txt 文件。")
        return False

    print("正在处理验证集目录结构...")

    # 读取 val_annotations.txt 文件，获取图像与类别的对应关系
    img_to_class = {}
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                class_id = parts[1]
                img_to_class[img_name] = class_id

    # 为每个类别创建子文件夹
    for class_id in set(img_to_class.values()):
        class_dir = val_dir / class_id
        class_dir.mkdir(exist_ok=True)

    # 移动图像到对应的类别子文件夹
    for img_name, class_id in img_to_class.items():
        src = val_img_dir / img_name
        dst = val_dir / class_id / img_name
        if src.exists():
            shutil.move(str(src), str(dst))

    # 删除原始 images 文件夹
    if val_img_dir.exists():
        val_img_dir.rmdir()

    print("验证集目录结构处理完成！")
    return True

if __name__ == "__main__":
    # 仅处理验证集
    val_dir = '/Users/qianqian/Desktop/Quanta-RT/data/tiny-imagenet-200/val'
    process_tinyimagenet_validation_set(val_dir)