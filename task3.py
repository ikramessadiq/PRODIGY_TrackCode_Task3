import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


DATASET_PATH = "train" 
print("Files in folder :", os.listdir(DATASET_PATH))
IMG_SIZE = (32,32)  

def load_images_from_folder(folder):
    images = []
    labels = []
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        
        if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)

                if "cat" in img_file:
                    labels.append(0)
                elif "dog" in img_file:
                    labels.append(1)
                else:
                    print(f" Ignored: {img_file}")

    print(f"Number of images loaded: {len(images)}")  
    return np.array(images), np.array(labels)



images, labels = load_images_from_folder(DATASET_PATH)


images = images / 255.0 


images_flat = images.reshape(images.shape[0], -1)


X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', random_state=42)


svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)


print("Précision :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# Visualisation des prédictions
def visualize_predictions(X_test, y_test, y_pred, img_size):
   
    sample_idx = np.random.choice(range(len(X_test)), size=5, replace=False)
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(sample_idx):
        img = X_test[idx].reshape(img_size[0], img_size[1], 3)  
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"True: {int(y_test[idx])}\nPredicted: {int(y_pred[idx])}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions(X_test, y_test, y_pred, IMG_SIZE)
