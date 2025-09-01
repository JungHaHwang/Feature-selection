# Parking Slot Dataset and Spatial Feature Selection  

This repository provides the **parking slot dataset** and Python code related to our **spatial feature selection** experiments.  

If you would like to try different datasets, you can download them from the links below:  
- [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)  
- [Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)  
- [Coffee Bean](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224)  
- [Concrete Crack](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification)  

---

## Before You Start  

This repository contains two main folders:  

- **`prepared`**: for users who want to reproduce our experiments.  
- **`completed`**: contains all results generated from our implementation.  

We recommend starting with the **`prepared`** folder if you want to replicate our experiments.  
Note that results may differ slightly from ours due to random weight initialization.  

---

## How to Run  

You must execute the code files **sequentially** in the following order:  

⚠️ **Important**: Before running each script, make sure that the file paths are correctly set.  

1. **`image_resize`**  
   - Resizes raw images to 222×222 and creates the `dataset_baseline` folder.  

2. **`preparing_baseline_dataset`**  
   - Loads images from `dataset_baseline` and generates NumPy files for data and labels.  

3. **`training_baseline`**  
   - Trains baseline models using the prepared data and labels.  
   - Saves models that achieve the desired training and test accuracy.  
   - Once trained, backbone outputs are automatically saved in the `backbone_outputs` folder.  

4. **`core_subspace_finder`**  
   - Identifies core subspaces from `backbone_outputs`.  
   - Saves the identified subspaces to `core_subspaces.txt`.  

5. **`core_subspace_sorting`**  
   - Sorts the core subspaces in `core_subspaces.txt` **by frequency of identification**.  
   - Saves the sorted array to the `core_subspace_sorting` folder.  

6. **`image_cropping`**  
   - Before running, select the desired core subspace and update the variable:  
     ```python
     core_subspace = [0, 2, 3, 3]
     ```  
   - Adjust this line to match your chosen subspace.  
   - After execution, cropped image folders will be generated under each class folder in `dataset_baseline`.  

7. **`preparing_cropped_dataset`**  
   - Loads cropped images and generates NumPy files for cropped data and labels.  

8. **`training_cropped`**  
   - Before running, modify the model architecture to match the cropped image dimensions. For example:  
     ```python
     self.fc1 = nn.Linear(1 * 4 * 2, 2)
     x = x.view(-1, 1 * 4 * 2)
     ```  
   - Update these lines according to your selected core subspace.  
   - Training results are saved in `Result_Training_cropped_XXX.txt`.  

---

✅ With this setup, you can easily reproduce our experiments or adapt the pipeline to your own datasets.  
