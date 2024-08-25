import numpy as np
import os
import sys
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


# Reading images in folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img.flatten())
    return np.array(images), img.shape

# Find the eigenfaces
def compute_eigenfaces(images, variance_threshold):
    
    mean_face = np.mean(images, axis=0)
    centered_images = images - mean_face

    # Compute SVD
    U, S, Vt = np.linalg.svd(centered_images, full_matrices=False)

    # Compute the cumulative variance
    variance_explained = np.cumsum(S**2) / np.sum(S**2)

    num_components = np.searchsorted(variance_explained, variance_threshold) + 1

    return Vt[:num_components], mean_face, num_components


# Visualize Eigenfaces
def plot_eigenfaces(eigenfaces, image_shape, num_eigenfaces=10):
    plt.figure(figsize=(15, 5))
    for i in range(num_eigenfaces):
        plt.subplot(2, 5, i + 1)
        plt.imshow(eigenfaces[i].reshape(image_shape), cmap="gray")
        plt.title(f"Eigenface {i+1}")
        plt.axis("off")
    plt.show()


# Reconstruct images using eigenfaces
def reconstruct_images(images, mean_face, eigenfaces):
    projections = np.dot(images - mean_face, eigenfaces.T)
    reconstructed_images = np.dot(projections, eigenfaces) + mean_face
    return reconstructed_images


def save_reconstructed_images(reconstructed_images, folder, image_shape):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, img in enumerate(reconstructed_images):
        img = img.reshape(image_shape)  # Resize based on original image dimensions
        img = (
            (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        )  # Normalize to [0, 255]
        imsave(os.path.join(folder, f"face{i:02d}.pgm"), img.astype(np.uint8))


# Visualize original and reconstructed images
def plot_reconstructed_images(
    original_images, reconstructed_images, image_shape, num_images=5
):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(image_shape), cmap="gray")
        plt.title(f"Original {i+1}")
        plt.axis("off")

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstructed_images[i].reshape(image_shape), cmap="gray")
        plt.title(f"Reconstructed {i+1}")
        plt.axis("off")
    plt.show()


# Recognize faces in the test dataset
def recognize_faces(test_images, reconstructed_train_images, train_labels):
    recognized_labels = []
    for test_img in test_images:
        distances = np.linalg.norm(reconstructed_train_images - test_img, axis=1)
        recognized_label = train_labels[np.argmin(distances)]
        recognized_labels.append(recognized_label)
    return recognized_labels


if len(sys.argv) != 2:
    print("Usage: python face_recognition.py <variance_threshold>")
    sys.exit(1)


variance_threshold = float(sys.argv[1])

# Load training images
train_images, image_shape = load_images_from_folder("faces_training")
eigenfaces, mean_face, num_components = compute_eigenfaces(
    train_images, variance_threshold
)

# Save the number of principal components
with open("output.txt", "w") as f:
    f.write(f"########## Step 1 ##########\n")
    f.write(f"Input Percentage: {variance_threshold}\n")
    f.write(f"Selected Dimension: {num_components}\n")
    f.write("\n")


plot_eigenfaces(eigenfaces, image_shape)
reconstructed_train_images = reconstruct_images(train_images, mean_face, eigenfaces)
save_reconstructed_images(reconstructed_train_images, "results", image_shape)

# Compute reconstruction error
reconstruction_errors = np.mean(
    (train_images - reconstructed_train_images) ** 2, axis=1
)
# average reconstruction error
average_reconstruction_error = np.mean(reconstruction_errors)

# Save reconstruction errors
with open("output.txt", "a") as f:
    f.write(f"########## Step 2 ##########\n")
    f.write(f"Reconstruction error\n")
    f.write(f"average: {average_reconstruction_error} \n")
    for i, error in enumerate(reconstruction_errors):
        f.write(f"{i+1:02d}: {error}\n")
    f.write("\n")


# Draw the reconstructed images
plot_reconstructed_images(train_images, reconstructed_train_images, image_shape)


# Load test images
test_images, _ = load_images_from_folder("faces_test")

# Recognize faces
train_labels = [
    int(f.split("face")[-1].split(".pgm")[0]) for f in os.listdir("faces_training")
]
recognized_labels = recognize_faces(
    test_images, reconstructed_train_images, train_labels
)

# Save recognized labels
with open("output.txt", "a") as f:
    f.write(f"########## Step 3 ##########\n")
    for i, label in enumerate(recognized_labels):
        f.write(f"test{i+1:02d}.pgm ==> face{label}.pgm\n")
