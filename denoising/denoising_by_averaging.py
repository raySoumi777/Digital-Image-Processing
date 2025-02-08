import cv2
import numpy as np
import os

# Noise Functions
def add_salt_and_pepper_noise(image, amount=0.01):
    noisy_image = np.copy(image)
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    # Add salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # Add pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

def add_gaussian_noise(image, mean=0, var=10):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    return noisy_image

def add_poisson_noise(image):
    noisy_image = np.random.poisson(image).astype(np.uint8)
    return noisy_image

def add_speckle_noise(image):
    gaussian = np.random.randn(*image.shape)
    noisy_image = np.clip(image + image * gaussian, 0, 255).astype(np.uint8)
    return noisy_image

def add_periodic_noise(image, frequency=10):
    rows, cols, _ = image.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    x_grid, y_grid = np.meshgrid(x, y)
    sinusoid = (np.sin(2 * np.pi * frequency * x_grid / cols) + 1) * 127.5
    sinusoid = sinusoid.astype(np.uint8)
    noisy_image = cv2.add(image, cv2.merge([sinusoid, sinusoid, sinusoid]))
    return noisy_image

# Generate Noisy Images
def generate_noisy_images(input_image_path, output_dir="output_noises"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {input_image_path}")

    noises = {
        "salt_and_pepper": add_salt_and_pepper_noise(image),
        "gaussian": add_gaussian_noise(image),
        "poisson": add_poisson_noise(image),
        "speckle": add_speckle_noise(image),
        "periodic": add_periodic_noise(image),
        "s&p_and_gaussian": add_salt_and_pepper_noise(add_gaussian_noise(image)),
        "s&p_and_poisson": add_salt_and_pepper_noise(add_poisson_noise(image)),
        "s&p_and_periodic": add_salt_and_pepper_noise(add_periodic_noise(image)),
        "s&p_and_speckle": add_salt_and_pepper_noise(add_speckle_noise(image)),
        "gaussian_and_poisson": add_gaussian_noise(add_poisson_noise(image)),
        "gaussian_and_speckle": add_gaussian_noise(add_speckle_noise(image)),
        "poisson_and_periodic": add_poisson_noise(add_periodic_noise(image)),
        "poisson_and_speckle": add_poisson_noise(add_speckle_noise(image)),
        "speckle_and_periodic": add_speckle_noise(add_periodic_noise(image)),
        "s&p_and_gaussian_and_poisson": add_salt_and_pepper_noise(add_gaussian_noise(add_poisson_noise(image))),
        "s&p_and_gaussian_and_speckle": add_salt_and_pepper_noise(add_gaussian_noise(add_speckle_noise(image))),
        "s&p_and_poisson_and_speckle": add_salt_and_pepper_noise(add_poisson_noise(add_speckle_noise(image))),
        "gaussian_and_poisson_and_periodic": add_gaussian_noise(add_poisson_noise(add_periodic_noise(image))),
        "gaussian_and_poisson_and_speckle": add_gaussian_noise(add_poisson_noise(add_speckle_noise(image))),
        "gaussian_and_speckle_and_periodic": add_gaussian_noise(add_speckle_noise(add_periodic_noise(image))),
        "poisson_and_speckle_and_periodic": add_poisson_noise(add_speckle_noise(add_periodic_noise(image))),
        "s&p_and_gaussian_and_poisson_and_periodic": add_salt_and_pepper_noise(add_gaussian_noise(add_poisson_noise(add_periodic_noise(image)))),
        "s&p_and_gaussian_and_poisson_and_speckle": add_salt_and_pepper_noise(add_gaussian_noise(add_poisson_noise(add_speckle_noise(image)))),
        "s&p_and_gaussian_and_speckle_and_periodic": add_salt_and_pepper_noise(add_gaussian_noise(add_speckle_noise(add_periodic_noise(image)))),
        "s&p_and_poisson_and_speckle_and_periodic": add_salt_and_pepper_noise(add_poisson_noise(add_speckle_noise(add_periodic_noise(image)))),
        "gaussian_and_poisson_and_speckle_and_periodic": add_gaussian_noise(add_poisson_noise(add_speckle_noise(add_periodic_noise(image)))),
        "s&p_and_gaussian_and_poisson_and_speckle_and_periodic": add_salt_and_pepper_noise(add_gaussian_noise(add_poisson_noise(add_speckle_noise(add_periodic_noise(image))))),

    }

    for noise_name, noisy_image in noises.items():
        output_path = os.path.join(output_dir, f"{noise_name}.png")
        cv2.imwrite(output_path, noisy_image)
        print(f"Saved: {output_path}")

# Denoise Images
def denoise_images(image_dir, output_path):
    """
    Denoise an image by averaging over multiple noisy versions.

    Args:
        image_dir (str): Path to the directory containing noisy images.
        output_path (str): Path to save the denoised image.
    """
    # List all images in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    if len(image_files) < 10:
        raise ValueError("At least 10 noisy images are required for denoising.")

    # Initialize an accumulator for summing images
    accumulator = None

    # Read and accumulate all images
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(image_dir, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)

        if accumulator is None:
            accumulator = np.zeros_like(img, dtype=np.float32)

        accumulator += img

    # Compute the average
    averaged_image = (accumulator / len(image_files)).astype(np.uint8)

    # Save the denoised image
    cv2.imwrite(output_path, averaged_image)
    print(f"Denoised image saved at: {output_path}")

# Example Usage
base_image_path = 'panda.jpg'  # Replace with your base image path
output_noisy_dir = 'output_noises'
output_denoised_image = 'denoised_image.png'

# Generate noisy images
generate_noisy_images(base_image_path, output_noisy_dir)

# Perform denoising
denoise_images(output_noisy_dir, output_denoised_image)
