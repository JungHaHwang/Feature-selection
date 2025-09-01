import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder, label):
    """
    Load images from a folder and assign a label.
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            with Image.open(filepath) as img:
                img_array = np.array(img)  # Keep original size
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images, labels

# Paths to the folders
folder_A = "./dataset_baseline/train/full"
folder_B = "./dataset_baseline/train/empty"
folder_C = "./dataset_baseline/test/full"
folder_D = "./dataset_baseline/test/empty"

# Load images and labels without resizing
images_A, labels_A = load_images_from_folder(folder_A, label=1)
images_B, labels_B = load_images_from_folder(folder_B, label=0)
images_C, labels_C = load_images_from_folder(folder_C, label=1)
images_D, labels_D = load_images_from_folder(folder_D, label=0)

# Combine data from both folders
images_train = np.array(images_A + images_B)
labels_train = np.array(labels_A + labels_B)
images_test = np.array(images_C + images_D)
labels_test = np.array(labels_C + labels_D)

# Save to .npy files
np.save("./dataset_baseline/train_data.npy", images_train)
np.save("./dataset_baseline/train_labels.npy", labels_train)
np.save("./dataset_baseline/test_data.npy", images_test)
np.save("./dataset_baseline/test_labels.npy", labels_test)

print("training and test dataset prepared successfully.")
