import os
from PIL import Image

def resize_and_convert_to_grayscale(input_folder, output_folder, target_size=(222, 222)):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        

        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            try:
                with Image.open(input_path) as img:
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    output_path = os.path.join(output_folder, filename)
                    resized_img.save(output_path)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 예제 실행
input_folder_list = ['./dataset_original/train/full',
                     './dataset_original/train/empty',
                     './dataset_original/test/full',
                     './dataset_original/test/empty']  
output_folder_list = ['./dataset_baseline/train/full',
                      './dataset_baseline/train/empty',
                      './dataset_baseline/test/full',
                      './dataset_baseline/test/empty']

for in_folder, out_folder in zip(input_folder_list, output_folder_list):
    resize_and_convert_to_grayscale(in_folder, out_folder)