import os
import numpy as np
import cv2

def preprocess_data(image_folder, label_folder):
    images = []
    labels = []

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        img = img / 255.0
        images.append(img)

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_folder, label_name)
        with open(label_path, 'r') as f:
            label_data = f.readlines()
        labels.append(label_data)

    return np.array(images), labels