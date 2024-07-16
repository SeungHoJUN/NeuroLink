import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

trained_model_path = '/cv/best.pt'
model = YOLO(trained_model_path)

test_image_path = "/test/images/SET1-1.JPG"
result = model.predict(test_image_path, save=False, conf=0.05)

new_width, new_height = 800, 600
plots = result[0].plot()
plots_resized = cv2.resize(plots, (new_width, new_height))

plots_rgb = cv2.cvtColor(plots_resized, cv2.COLOR_BGR2RGB)

detections = result[0].boxes.data.cpu().numpy()
print("Detected objects:", detections)

output_image_path = "output_image.jpg"
cv2.imwrite(output_image_path, plots_resized)
print(f"이미지는 다음과 같은 경로에 저장되었습니다. {output_image_path}")

fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(plots_rgb)
plt.axis('off')
plt.title('Detection Results')
plt.show()

# 탐지 결과
df_detections = pd.DataFrame(detections, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'])
df_detections.to_csv('detection_results.csv', index=False)
print("Detection results saved to detection_results.csv")

mean_confidence = df_detections['confidence'].mean()
print(f"Mean confidence score of detections: {mean_confidence:.2f}")

class_counts = df_detections['class'].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
class_counts.plot(kind='bar', ax=ax)

ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Detected Object Counts by Class')
plt.show()
