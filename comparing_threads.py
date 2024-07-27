import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor


# Function to process a single image: convert it to grayscale, resize, blur, and detect edges
def process_image(image_path, output_dirs):
    try:
        # Read the image from the specified path
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize the image to half its original dimensions
        resized_image = cv2.resize(gray_image, (gray_image.shape[1] // 2, gray_image.shape[0] // 2))
        # Apply Gaussian blur to the image
        blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
        # Perform Canny edge detection
        edges = cv2.Canny(blurred_image, 50, 150)

        # Create output paths for each transformation
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        gray_output_path = os.path.join(output_dirs['gray'], f"{base_name}_gray.jpg")
        resized_output_path = os.path.join(output_dirs['resized'], f"{base_name}_resized.jpg")
        blurred_output_path = os.path.join(output_dirs['blurred'], f"{base_name}_blurred.jpg")
        edges_output_path = os.path.join(output_dirs['edges'], f"{base_name}_edges.jpg")

        # Save the processed images
        cv2.imwrite(gray_output_path, gray_image)
        cv2.imwrite(resized_output_path, resized_image)
        cv2.imwrite(blurred_output_path, blurred_image)
        cv2.imwrite(edges_output_path, edges)

        print(f"Processed and saved images for: {image_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


# Function to create output directories
def create_output_dirs(base_output_dir):
    dirs = {
        'gray': os.path.join(base_output_dir, 'gray'),
        'resized': os.path.join(base_output_dir, 'resized'),
        'blurred': os.path.join(base_output_dir, 'blurred'),
        'edges': os.path.join(base_output_dir, 'edges'),
    }
    for dir_path in dirs.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return dirs


# Function to process images using monothreading
def monothread_process(image_paths):
    # Define the base output directory for monothreaded processing
    output_dirs = create_output_dirs("output_monothread")

    start_time = time.time()
    # Process each image one by one
    for image_path in image_paths:
        process_image(image_path, output_dirs)
    end_time = time.time()

    print(f"Monothreading: Processed {len(image_paths)} images in {end_time - start_time:.2f} seconds")


# Function to process images using multithreading
def multithread_process(image_paths):
    # Define the base output directory for multithreaded processing
    output_dirs = create_output_dirs("output_multithread")

    start_time = time.time()
    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(lambda img_path: process_image(img_path, output_dirs), image_paths)
    end_time = time.time()

    print(f"Multithreading: Processed {len(image_paths)} images in {end_time - start_time:.2f} seconds")


def main():
    # Define the directory containing images to be processed
    image_dir = "images"

    # Get a list of image paths in the image directory
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if
                   img.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print("No images found in the 'images' directory.")
        return

    # Process images using monothreading and measure the time taken
    print("Starting monothread processing...")
    monothread_process(image_paths)

    # Process images using multithreading and measure the time taken
    print("Starting multithread processing...")
    multithread_process(image_paths)


if __name__ == "__main__":
    main()
