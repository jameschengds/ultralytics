from ultralytics import YOLO
from ultralytics import settings

# 加载一个模型
model = YOLO('yolov8x.yaml')  # 从YAML建立一个新模型
#
# settings.update({'runs_dir': '/mnt/dgx/ultralytics/runs', "datasets_dir": "/mnt/yolo_data",
#                  "weights_dir": "/mnt/dgx/ultralytics/weights"})
# 训练模型
# results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
results = model.train(data='xray.yaml', epochs=100, imgsz=640, device=[0, 1])
metrics = model.val()
