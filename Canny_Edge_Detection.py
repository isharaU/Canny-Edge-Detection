import os
import cv2
import numpy as np
from PIL import Image

# see jpg,png or bmp files in the directory
valid_extensions = [".jpg", ".png", ".bmp"]

# list all files in the current directory
current_directory = os.getcwd()
files = [f for f in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, f))]

# filter out the files with valid extensions
valid_files = [f for f in files if any(f.endswith(ext) for ext in valid_extensions)]

number = 1
# choose a file to process

print("Available images:")
for image_name in valid_files:
    print(f"{number}. {image_name}")
    number += 1

while True:
    try:
        input_number = int(input("Enter the number of the image to process: "))
        chosen_image = valid_files[input_number - 1]
        break
    except ValueError:
        print("Invalid input. Please enter a number.")
        continue
    except IndexError:
        print("Invalid input. Please enter a number within the range.")
        continue

print(f"You chose {chosen_image}")

# convert to grayscale 
def convert_to_grayscale(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)

    # handeling aplha channel
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    grayscale_img = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    grayscale_img = grayscale_img.astype(np.uint8)
    return Image.fromarray(grayscale_img)

grayscale_image = convert_to_grayscale(chosen_image)
grayscale_image.show()  # Display the grayscale image

