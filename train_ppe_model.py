import os
import shutil
import yaml
import random
from ultralytics import YOLO

# 设置项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 只需要指向 Dataset 总文件夹
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets_formatted')  # 整理后的数据存放处
CONFIG_PATH = os.path.join(BASE_DIR, 'sh17_config.yaml')


def check_dataset_structure():
    print("正在检查数据集结构...")

    # 1. 强制清理旧数据
    if os.path.exists(OUTPUT_DIR):
        print(f"检测到旧的数据集目录: {OUTPUT_DIR}，正在清理并重新生成...")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except Exception as e:
            print(f"警告: 无法删除旧目录，请手动删除 {OUTPUT_DIR} 后重试。错误: {e}")
            return

    # 创建目标目录结构
    target_train_dir = os.path.join(OUTPUT_DIR, 'images', 'train')
    target_val_dir = os.path.join(OUTPUT_DIR, 'images', 'val')
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_val_dir, exist_ok=True)

    # --- 模式 A: 检查是否已经是分好 train/val 的结构 ---
    # 假设结构是 Dataset/images/train 和 Dataset/images/val
    src_img_train = os.path.join(DATASET_DIR, 'images', 'train')
    src_img_val = os.path.join(DATASET_DIR, 'images', 'val')

    if os.path.exists(src_img_train) and os.path.exists(src_img_val):
        print("√ 检测到已预先划分好的数据集结构 (images/train, images/val)")

        # 处理训练集
        train_files = [f for f in os.listdir(src_img_train) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"正在处理预划分的训练集 ({len(train_files)} 张)...")
        for f in train_files:
            _copy_file_and_label(src_img_train, f, target_train_dir, 'train')

        # 处理验证集
        val_files = [f for f in os.listdir(src_img_val) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"正在处理预划分的验证集 ({len(val_files)} 张)...")
        for f in val_files:
            _copy_file_and_label(src_img_val, f, target_val_dir, 'val')

    else:
        # --- 模式 B: 没分好，尝试作为一整坨处理 ---
        print("未检测到 train/val 子文件夹，尝试寻找所有图片并自动划分...")

        # 智能查找图片源目录
        possible_source_dirs = [
            DATASET_DIR,
            os.path.join(DATASET_DIR, 'images'),
            os.path.join(DATASET_DIR, 'JPEGImages'),
        ]

        source_dir = None
        all_files = []

        for d in possible_source_dirs:
            if os.path.exists(d):
                files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if len(files) > 0:
                    source_dir = d
                    all_files = files
                    print(f"√ 在以下路径找到 {len(files)} 张未划分图片: {source_dir}")
                    break

        if source_dir is None:
            print(f"\n错误：在 {DATASET_DIR} 及其子文件夹中均未找到任何图片文件！")
            print("请确认图片位于 Dataset/images/train 和 val 中，或者直接在 Dataset/images 下。")
            raise FileNotFoundError("未找到图片数据")

        # 自动划分数据集 (80% 训练, 20% 验证)
        random.shuffle(all_files)
        split_idx = int(len(all_files) * 0.8)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        print(f"正在自动划分并处理训练集文件 ({len(train_files)} 个)...")
        for f in train_files:
            _copy_file_and_label(source_dir, f, target_train_dir, 'train')

        print(f"正在自动划分并处理验证集文件 ({len(val_files)} 个)...")
        for f in val_files:
            _copy_file_and_label(source_dir, f, target_val_dir, 'val')

    # 最终自检
    if len(os.listdir(target_train_dir)) == 0:
        raise RuntimeError("严重错误：文件复制失败，目标训练集文件夹为空！请检查源文件夹路径是否正确。")

    print(f"数据准备完成。")


def _copy_file_and_label(source_dir, filename, target_img_dir, split_type):
    """
    辅助函数：复制图片并尝试寻找对应的标签文件
    支持多种标签存放习惯
    """
    # 1. 复制图片
    shutil.copy(os.path.join(source_dir, filename), os.path.join(target_img_dir, filename))

    # 2. 寻找并复制标签 (txt)
    txt_name = os.path.splitext(filename)[0] + '.txt'

    # 这里的 source_dir 可能是 .../images/train

    # 策略A：标签和图片在同一个文件夹 (最简单)
    # path: .../images/train/abc.txt
    path_a = os.path.join(source_dir, txt_name)

    # 策略B：标签在平级的 labels 文件夹对应的子目录里 (YOLO标准)
    # 如果图片在 .../images/train，标签就在 .../labels/train
    # 我们把路径里的 'images' 替换为 'labels' 试试
    if 'images' in source_dir:
        path_b = source_dir.replace('images', 'labels')
        path_b = os.path.join(path_b, txt_name)
    else:
        path_b = "NON_EXISTENT_PATH"  # 占位

    # 策略C：标签在上一级目录的 labels 文件夹里 (简单结构)
    # 图片: .../images/abc.jpg -> 标签: .../labels/abc.txt
    parent_dir = os.path.dirname(source_dir)
    path_c = os.path.join(parent_dir, 'labels', txt_name)

    final_src_txt = None
    if os.path.exists(path_a):
        final_src_txt = path_a
    elif os.path.exists(path_b):
        final_src_txt = path_b
    elif os.path.exists(path_c):
        final_src_txt = path_c

    if final_src_txt:
        label_dir = os.path.join(OUTPUT_DIR, 'labels', split_type)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(final_src_txt, os.path.join(label_dir, txt_name))


def create_yaml_config():
    """
    创建 YOLO 训练所需的配置文件
    """
    # 定义类别 (根据你的实际项目修改这里)
    class_names = {
        0: 'person',
        1: 'vest',
        2: 'helmet',
        3: 'head',
        4: 'face',
        5: 'glasses',
        6: 'face-mask-medical',
        7: 'ear',
        8: 'earmuffs',
        9: 'hands',
        10: 'gloves',
        11: 'foot',
        12: 'shoes',
        13: 'safety-vest',
        14: 'tools',
        15: 'medical-suit',
        16: 'safety-suit'
    }

    config = {
        'path': OUTPUT_DIR,  # 数据集根目录
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names
    }

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    print(f"配置文件已创建于: {CONFIG_PATH}")


def check_local_model(model_name='yolov8n.pt'):
    """
    检查本地是否存在权重文件，若不存在则尝试从网络下载
    """
    model_path = os.path.join(BASE_DIR, model_name)
    if os.path.exists(model_path):
        print(f"正在加载本地 YOLO 模型: {model_name} ...")
        return model_path
    else:
        print(f"本地未找到 {model_name}，正在尝试从网络自动下载...")
        return model_name


def main():
    try:
        # 1. 检查并准备数据
        check_dataset_structure()

        # 2. 创建配置
        create_yaml_config()

        # 3. 加载模型
        model_path = check_local_model('yolov8n.pt')
        model = YOLO(model_path)

        print("开始训练模型...")
        # 4. 开始训练
        results = model.train(
            data=CONFIG_PATH,
            epochs=100,
            imgsz=640,
            batch=8,
            workers=0,
            name='ppe_training_result',
            verbose=False
        )

        print("训练完成！")
        print(f"模型保存路径: {results.save_dir}")

    except Exception as e:
        print(f"\n发生错误: {e}")
        print("请检查上方错误信息进行排查。")


if __name__ == '__main__':
    main()