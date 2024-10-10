# Canny Edge Detection Script

This Python script implements the Canny Edge Detection algorithm to detect edges in images. It provides a user-friendly interface for processing image files in the current directory.

## Features

- Converts images to grayscale
- Applies Gaussian blur for noise reduction
- Uses Sobel filters for gradient calculation
- Performs non-maximum suppression for edge thinning
- Applies double thresholding and hysteresis for final edge detection
- Supports multiple image formats (JPG, PNG, BMP)
- Automatically resizes large images for faster processing
- Displays both original and edge-detected images

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Pillow (PIL)

## Installation

1. Clone this repository or download the script.
2. Install the required packages:

```bash
pip install opencv-python numpy pillow
```

## Usage

1. Place the script in the same directory as your image files.
2. Run the script:

```bash
python canny_edge_detection.py
```

3. The script will display a list of available images in the current directory.
4. Enter the number corresponding to the image you want to process.
5. The script will generate two output files:
   - `grayscale_<original_filename>`: The grayscale version of the input image
   - `<original_filename>_edge.png`: The edge-detected version of the input image
6. The script will also display the original and edge-detected images in separate windows.

## How It Works

The Canny Edge Detection algorithm consists of several steps:

1. Grayscale Conversion: Converts the input image to grayscale.
2. Gaussian Blur: Applies a Gaussian filter to reduce noise.
3. Gradient Calculation: Uses Sobel filters to calculate intensity gradients.
4. Non-Maximum Suppression: Thins out the edges by suppressing non-maximum pixels.
5. Double Thresholding: Identifies strong, weak, and non-relevant pixels.
6. Edge Tracking by Hysteresis: Finalizes the edges by including weak pixels that are connected to strong edges.

## Customization

You can adjust the following parameters in the script to fine-tune the edge detection:

- `gaussian_kernel_size`: Size of the Gaussian kernel for blurring (default: 5)
- `low_threshold` and `high_threshold`: Thresholds for the double thresholding step (default: 50 and 150)
- `max_width` and `max_height`: Maximum dimensions for image resizing (default: 800x600)

## Contributing

Feel free to fork this repository and submit pull requests with any improvements or bug fixes.

## License

This project is open-source and available under the MIT License.
