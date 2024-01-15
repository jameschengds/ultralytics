from ultralytics import YOLO

# 加载一个模型
model = YOLO('yolov8x.yaml')  # 从YAML建立一个新模型

# 训练模型
# results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
results = model.train(data='xray.yaml', epochs=100, imgsz=640, device=[1, 2])
metrics = model.val()