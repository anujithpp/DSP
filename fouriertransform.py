import numpy as np
import cv2
import matplotlib.pyplot as plt

# Specify the file path of your image
image_path = '/home/cusat/Desktop/ramanan/image.jpg'  # Change this to the correct path

# Read the image (ensure you handle the file path correctly)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Fourier Transform and Display
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    # Save the images to files
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Magnitude spectrum
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.axis('off')

    # Save the plot to a file
    plt.savefig('/home/cusat/Desktop/ramanan/output_plot.png')  # Specify your desired output path

    print("Plot saved as output_plot.png")
