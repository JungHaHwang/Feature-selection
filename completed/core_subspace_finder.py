import os
import numpy as np
from itertools import product
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to load features and labels from .npy files
def load_data_npy(features_file, labels_file):
    features = np.load(features_file)
    print("Features shape:", features.shape)
    labels = np.load(labels_file)
    print("Labels shape:", labels.shape)
    return features, labels

# Function to get all valid rectangular submatrices for 5x5 matrices
def get_rectangular_submatrices():
    submatrices = []
    for start_row in range(5):
        for start_col in range(5):
            for end_row in range(start_row, 5):
                for end_col in range(start_col, 5):
                    submatrices.append((start_row, start_col, end_row, end_col))
    return submatrices

# Function to extract features based on selected submatrix across 1 matrices
def extract_submatrix_features(data, submatrix):
    start_row, start_col, end_row, end_col = submatrix
    selected_features = []
    for feature_tensor in data:
        submatrix_values = np.concatenate([
            feature_tensor[i, start_row:end_row+1, start_col:end_col+1].flatten() for i in range(1)
        ])
        selected_features.append(submatrix_values)
    return np.array(selected_features)

# Function to check if features are linearly separable using SVM
def is_linearly_separable(features, labels):
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(features, labels)
    predictions = classifier.predict(features)
    return accuracy_score(labels, predictions) == 1.0

# Modified function to find all smallest bounding rectangles
def find_all_smallest_bounding_rectangles(selected_combinations):
    smallest_area = float('inf')
    smallest_rectangles = []

    for combination in selected_combinations:
        min_row, min_col = 5, 5
        max_row, max_col = 0, 0

        # Combine all submatrices to find the bounding rectangle
        for submatrix in combination:
            start_row, start_col, end_row, end_col = submatrix
            min_row = min(min_row, start_row)
            min_col = min(min_col, start_col)
            max_row = max(max_row, end_row)
            max_col = max(max_col, end_col)

        # Calculate the area of the bounding rectangle
        area = (max_row - min_row + 1) * (max_col - min_col + 1)

        if area < smallest_area:
            smallest_area = area
            smallest_rectangles = [(min_col, min_row, max_col, max_row)]
        elif area == smallest_area:
            smallest_rectangles.append((min_col, min_row, max_col, max_row))
    
    # Remove duplicates by converting to a set of tuples and back to list
    smallest_rectangles = list(set(smallest_rectangles))
    return smallest_rectangles

# Main function for one-vs-all classification
def process_files_in_directory_npy(directory, labels_file):
    label_path = os.path.join(directory, labels_file)
    
    for file_name in os.listdir(directory):
        if file_name.startswith("outputs") and file_name.endswith(".npy"):
            features_path = os.path.join(directory, file_name)
            
            # Load and preprocess data
            features = np.load(features_path).squeeze(axis=1)
            labels = np.load(label_path).squeeze(axis=1)
            
            if features.shape[1:] != (1, 5, 5):
                raise ValueError(f"Unexpected features shape: {features.shape}. Expected (N, 1, 5, 5).")
            
            submatrices = get_rectangular_submatrices()
            unique_classes = np.unique(labels)
            
            # Store all valid submatrices for each class
            class_results = {selected_class: [] for selected_class in unique_classes}
            
            for selected_class in unique_classes:
                binary_labels = np.where(labels == selected_class, 1, 0)
                for submatrix in submatrices:
                    selected_features = extract_submatrix_features(features, submatrix)
                    if is_linearly_separable(selected_features, binary_labels):
                        class_results[selected_class].append(submatrix)
            
            # Find all the absolute smallest bounding rectangles
            smallest_rectangles = find_all_smallest_bounding_rectangles(
                product(*(class_results.values()))
            )
            
            # Format the result for the current file
            print(f"File: {file_name} | Core subspaces: {smallest_rectangles}")
            
            with open("./core_subspaces.txt", "a", encoding="utf-8") as f:
                f.write(f"File: {file_name} | Core subspaces: {smallest_rectangles}\n")

# Example usage
directory = './backbone_outputs'  # Replace with the actual directory path
labels_file = 'labels.npy'  # Replace with the actual labels file name
process_files_in_directory_npy(directory, labels_file)
