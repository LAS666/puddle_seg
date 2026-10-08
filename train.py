from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

def train_normal(dataset_yaml, model_config, custom_cfg, pretrained=True):


    model = YOLO(model_config)

    # 训练参数
    train_args = {
        'data': dataset_yaml,
        'cfg': custom_cfg,
        'pretrained': pretrained,
        'epochs': 150,
        'patience': 150,
        'batch': 16,
        'imgsz': 640,
        'rect': True,
        'workers': 8,
        'close_mosaic': 0,
        'project': str(PROJECT_DIR / 'runs'),
        'name': 'mud_jingxiu',
    }

    # 开始训练
    try:
        model.train(**train_args)
    except Exception as e:
        print(f"Error training the model: {e}")
        raise

if __name__ == "__main__":
    train_normal(
        dataset_yaml=str(PROJECT_DIR / "datasets/mud_jingxiu/data.yaml"),
        model_config=str(PROJECT_DIR / "weights/yolo11n-seg.pt"),
        custom_cfg=str(PROJECT_DIR / "configs/train_cfg.yaml"),
        pretrained=True,
    )
