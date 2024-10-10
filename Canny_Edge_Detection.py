import os
import cv2
import sys
import numpy as np
from PIL import Image

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

# Helper function for Gaussian kernel generation
def gaussian_kernel(size, sigma=1.0):
    k = int(size / 2)
    x, y = np.mgrid[-k:k+1, -k:k+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return kernel

# Helper function for gradient calculation using Sobel operators
def sobel_filters(img):
    k_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    k_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    i_x = cv2.filter2D(img, -1, k_x)
    i_y = cv2.filter2D(img, -1, k_y)
    
    G = np.hypot(i_x, i_y)
    G = G / G.max() * 255  # Normalize gradient magnitude
    
    theta = np.arctan2(i_y, i_x)
    return G, theta

# Helper function for non-maximum suppression
def non_max_suppression(gradient, theta):
    M, N = gradient.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            q, r = 255, 255
            
            # Horizontal edge (0 degrees)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient[i, j+1]
                r = gradient[i, j-1]
            # Diagonal (45 degrees)
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient[i+1, j-1]
                r = gradient[i-1, j+1]
            # Vertical (90 degrees)
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient[i+1, j]
                r = gradient[i-1, j]
            # Diagonal (135 degrees)
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient[i-1, j-1]
                r = gradient[i+1, j+1]
            
            if (gradient[i, j] >= q) and (gradient[i, j] >= r):
                Z[i, j] = gradient[i, j]
            else:
                Z[i, j] = 0
    
    return Z

# Helper function for double thresholding
def threshold(img, low_threshold, high_threshold):
    weak = 75
    strong = 255
    
    strong_i, strong_j = np.nonzero(img >= high_threshold)
    weak_i, weak_j = np.nonzero((img >= low_threshold) & (img < high_threshold))
    
    result = np.zeros(img.shape, dtype=np.uint8)
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    return result, weak, strong

# Helper function for edge tracking by hysteresis
def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i, j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

# Main function to apply Canny Edge Detection
def canny_edge_detection(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Apply Gaussian Blur to smooth the image
    gaussian_kernel_size = 5
    blurred_img = cv2.filter2D(img, -1, gaussian_kernel(gaussian_kernel_size, sigma=1.4))
    
    # Step 2: Apply Sobel filters to find intensity gradients
    gradient_magnitude, gradient_direction = sobel_filters(blurred_img)
    print("Sobel filter applied...")
    
    # Step 3: Perform Non-Maximum Suppression to thin the edges
    non_max_img = non_max_suppression(gradient_magnitude, gradient_direction)
    print("Non-maximum suppression applied...")
    
    # Step 4: Apply double threshold to determine potential edges
    low_threshold = 50
    high_threshold = 150
    threshold_img, weak, strong = threshold(non_max_img, low_threshold, high_threshold)
    print("Double thresholding applied...")
    
    # Step 5: Perform edge tracking by hysteresis
    final_edges = hysteresis(threshold_img, weak, strong)
    print("Edge tracking by hysteresis applied...")
    # Save the output image with "_edge.png" suffix
    
    output_file_name = os.path.splitext(image_path)[0] + "_edge.png"
    cv2.imwrite(output_file_name, final_edges)
    print("Output image saved!")
    
    # Display the original and the edge-detected image
    cv2.imshow('Original Image', img)
    cv2.imshow('Canny Edge Detection', final_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # list all files in the current directory
    valid_extensions = [".jpg", ".png", ".bmp"]
    current_directory = os.getcwd()
    files = [f for f in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, f))]

    # filter out the files with valid extensions
    valid_files = [f for f in files if any(f.endswith(ext) for ext in valid_extensions)]
   
    # choose a file to process
    print("Available images:")
    number = 1
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
    image_grayscale = convert_to_grayscale(chosen_image)
    image_grayscale.save("grayscale_" + chosen_image)
    canny_edge_detection("grayscale_" + chosen_image)



