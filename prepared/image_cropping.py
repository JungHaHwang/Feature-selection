import os
from PIL import Image

def crop_images_in_folder(input_folder, output_folder, crop_box):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path)
            cropped_img = img.crop(crop_box)
            output_path = os.path.join(output_folder, filename)
            cropped_img.save(output_path)

input_folder_list = ["./dataset_baseline/train/full", "./dataset_baseline/train/empty", "./dataset_baseline/test/full", "./dataset_baseline/test/empty"]

# select one of core subspaces
core_subspace = [0, 2, 3, 3] # (min_col, min_row, max_col, max_row)

def corresponding_regions(core_subspace):
    crop_box = []
    min_col = core_subspace[0]
    min_row = core_subspace[1]
    max_col = core_subspace[2]
    max_row = core_subspace[3]
    corresponding_min_col = 32*min_col
    corresponding_min_row = 32*min_row
    corresponding_max_col = 32*max_col+93+1
    corresponding_max_row = 32*max_row+93+1
    crop_box.append(corresponding_min_col)
    crop_box.append(corresponding_min_row)
    crop_box.append(corresponding_max_col)
    crop_box.append(corresponding_max_row)
    return crop_box

for i in input_folder_list:
    output_folder = str(i)+"_cropped_"+str(core_subspace)
    
    # crop_box indicates pixel range corresponding to the core subspace
    crop_box = tuple(corresponding_regions(core_subspace))  # (min_col, min_row, max_col, max_row)

    crop_images_in_folder(i, output_folder, crop_box)