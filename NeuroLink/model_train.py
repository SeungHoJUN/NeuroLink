import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

# 전처리할 데이터 경로
image_folder = '/cv/data/images'
label_folder = '/cv/data/labels'

# 데이터 전처리
images, labels = preprocess_data(image_folder, label_folder)

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

model = YOLO('yolov8m.pt')

# 하이퍼파라미터 조정
param_grid = [
    {'learning_rate': 0.001, 'batch_size': 16},
    {'learning_rate': 0.001, 'batch_size': 32},
    {'learning_rate': 0.01, 'batch_size': 16},
    {'learning_rate': 0.01, 'batch_size': 32}
]

data_config_path = '/content/drive/MyDrive/cv/data.yaml'

def tune_hyperparameters(model, data_config_path, param_grid, test_image_path):
    best_params = None
    best_score = -np.inf
    
    for params in param_grid:
        results = model.train(data=data_config_path, patience=10, epochs=1, imgsz=640,
                              lr0=params['learning_rate'], batch=params['batch_size'])
        
        result = model.predict(test_image_path, save=False, conf=0.05)
        
        detections = result[0].boxes.data.cpu().numpy()
        score = np.sum(detections[:, 4])
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params

best_params = tune_hyperparameters(model, data_config_path, param_grid, test_image_path)
print(f"Best Hyperparameters: {best_params}")

# 최적학습
results = model.train(data=data_config_path, patience=40, epochs=1, imgsz=640,
                      lr0=best_params['learning_rate'], batch=best_params['batch_size'])

print("Training complete. Results:", results)
