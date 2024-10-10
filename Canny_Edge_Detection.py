import os

# see jpg,png or bmp files in the directory
valid_extensions = [".jpg", ".png", ".bmp"]

# get the current working directory
current_directory = os.getcwd()

# list all files in the current directory
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




