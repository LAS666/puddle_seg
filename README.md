# YOLO 分割代码

从最新 yolo_seg 整理，包含训练、评估、五折交叉验证和数据处理工具。
不含 datasets、runs、weights、虚拟环境和缓存；原项目保持不变。

## 环境

原环境：Python 3.10.12，Linux x86_64 / WSL2，PyTorch 2.13.0+cu126，
torchvision 0.28.0+cu126，Ultralytics 8.4.144。
`requirements.txt` 固定项目直接依赖版本；`environment-snapshot.txt` 是完整环境参考，
含 ROS、TensorRT 以及其他历史依赖，不要直接用它安装。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

版本来自本机已安装包，不代表远程软件源一定仍提供对应版本；尚未验证全新环境安装。
如果提示找不到版本，需要准备原环境对应 wheel 或选择经测试兼容的版本，不能认为已完全复现。
此配置针对原 Linux x86_64 CUDA 环境，不直接适用于 Jetson/aarch64。

## 训练和评估

1. 将 `configs/mud_mingxian.yaml` 的 path 改为本机数据集绝对路径。
2. 数据目录包含 images/train、images/val 和对应 labels/train、labels/val。
   当前模板为 0=puddle、1=mud，不自动修改标签。
3. 自行准备 weights/yolo11n-seg.pt；权重不提交 Git。

```bash
bash train.sh
EVAL_WEIGHTS=runs/mud_mingxian_2cls/weights/best.pt bash eval.sh
python eval.py --help
python eval_fcv.py --help
python train_5_fcv.py --help
python train_aug_5_fcv.py --help
```

train.sh 保留原参数（100 epochs、batch=16、imgsz=640），其余见 configs/train_cfg.yaml。
可在命令末尾传入 `data=/自己的/data.yaml model=/自己的/model.pt`。
Shell 入口使用已激活环境，可设置 PYTHON 或 YOLO_BIN 指定解释器或训练命令。
重复训练的保存目录可能带后缀，请给评估传入实际权重路径。

## 历史工具

所有工具从仓库根目录运行。原本的个人绝对路径替换为仓库相对路径，
使用前检查各脚本顶部的数据和输出路径。configs/data.yaml 是单类别混合数据模板，
需自行填写数据根目录；五折代码保留原水坑实验默认配置，不直接作为两类别配置。
train.py 是历史入口，其默认配置与 train.sh 不同。

tools 下包含删除、替换及转换数据的脚本，部分会在导入时执行。
不要批量运行或 import，使用前阅读源码并备份数据。
未启动训练、未初始化或上传 Git 仓库；发布前自行核对许可证与代码发布权限。
