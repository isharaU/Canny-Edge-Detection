import os

# see jpg,png or bmp files in the directory
valid_extensions = [".jpg", ".png", ".bmp"]

# get the current working directory
current_directory = os.getcwd()

# list all files in the current directory
files = [f for f in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, f))]

# filter out the files with valid extensions
valid_files = [f for f in files if any(f.endswith(ext) for ext in valid_extensions)]

# print the valid files
print(valid_files)

