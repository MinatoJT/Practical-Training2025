# 软件设计说明

标识：PPE-SDD。版本 0.1，待评审。当前项目采用单个 Python 脚本组织流程，无独立服务或数据库。
源代码基线：`4bb13b0391a2f2581aec488c305a6ddaabde6ac8`；本版修订日期：2026-10-03。

## 控制与数据流

`main()` → `check_dataset_structure()` → `create_yaml_config()` → `check_local_model('yolov8n.pt')` → `YOLO(model_path)` → `model.train(...)` → 打印保存路径。

路径以脚本所在目录 `BASE_DIR` 为基准。数据整理先删除整个 `datasets_formatted/`，再建立图像目录。输入 Dataset 不被主动删除，但输出目录不应存放唯一数据。

| 函数 | 职责 | 关键行为与边界 |
|---|---|---|
| `check_dataset_structure()` | 识别布局、分组、复制、自检 | 优先预划分布局；否则按 80/20 随机划分；最后只检查训练图像目录是否非空 |
| `_copy_file_and_label()` | 复制图像并尝试标签匹配 | 图像先复制；缺标签时不会主动报错，也不验证标签内容 |
| `create_yaml_config()` | 写 YAML | 重写根配置，输出绝对数据根路径和 17 类名 |
| `check_local_model()` | 确定权重参数 | 本地存在则返回路径；否则返回模型名，由库尝试下载 |
| `main()` | 串联训练 | 捕获所有普通异常并打印，未重新抛出或显式设置失败退出码 |

## 图片与标签匹配

预划分模式读取 `Dataset/images/train` 和 `Dataset/images/val` 的直接子文件。自动模式依次扫描 `Dataset/`、`Dataset/images/`、`Dataset/JPEGImages/`，使用第一个找到图片的目录；先 `random.shuffle()` 再按 floor(N×0.8) 切分。

每个图像通过文件主名寻找同名 `.txt`，优先顺序如下：

1. 图像所在目录中的 `.txt`。
2. 将源目录字符串内 `images` 替换为 `labels` 后的路径。
3. 源目录上一级目录中的 `labels/<同名>.txt`。

查到第一个存在路径后复制，不解析内容。策略 2 使用字符串替换而非路径组件替换，目录名中含 `images` 的其他部分也可能受到影响。同名但不同图像扩展名可能对应相同标签主名，需在输入检查时排除冲突。

## 类别定义

脚本 `create_yaml_config()` 的当前定义为：

| 编号 | 名称 | 编号 | 名称 |
|---|---|---|---|
| 0 | person | 9 | hands |
| 1 | vest | 10 | gloves |
| 2 | helmet | 11 | foot |
| 3 | head | 12 | shoes |
| 4 | face | 13 | safety-vest |
| 5 | glasses | 14 | tools |
| 6 | face-mask-medical | 15 | medical-suit |
| 7 | ear | 16 | safety-suit |
| 8 | earmuffs | — | — |

已提交 YAML 仅记录编号 0、1、2。不能据此断言历史权重训练了全部 17 类，也不能仅按模型文件名确认权重内部类别。本次未反序列化权重。

## 训练调用参数

当前入口选 `yolov8n.pt`，显式传入：`data=CONFIG_PATH`、`epochs=100`、`imgsz=640`、`batch=8`、`workers=0`、`name='ppe_training_result'`、`verbose=False`。其余参数由实际安装的 Ultralytics 版本处理，本项目没有锁定这些默认值，也没有命令行参数解析器。

历史 result4 的 `args.yaml` 包含 `verbose: true`、实验名带 `4` 及历史路径，与当前入口并非完全一致。历史参数只能证明文件所记录的配置，不能证明当前源码重跑产生相同产物。

## 异常与安全风险

- 删除旧输出失败时，数据准备函数打印警告后直接返回，主流程仍可能继续写配置并训练。
- 找不到图片或训练目录为空时抛出异常，但会被 `main()` 捕获；自动化调用仅看退出码可能误判。
- 不显式检查验证集非空、漏标签、坐标范围、类别越界或训练/验证数据泄漏。
- 缺本地模型时可能触发网络下载；应预先核实下载来源、权重授权和网络条件。

以上均是静态检查发现，未在本次修复或通过运行验证。改动时应另行评审并同步更新需求、用例和配置基线。
