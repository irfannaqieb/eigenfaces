# Simple Facial Recognition using Eigenfaces

This project demonstrates a basic facial recognition system using the concept of eigenfaces. Eigenfaces are the principal components of the face space and can be used for facial recognition by projecting new images into the face space and finding the closest match.

## Dependencies

To run this project, you will need the following Python libraries:

- `numpy`
- `scikit-image`
- `matplotlib`

You can install them using pip:

```bash
pip install numpy scikit-image matplotlib

```
##  How to run
1) Preparing Dataset:

    - Place your training images (e.g., face01.pgm, face02.pgm, ...) in the faces_training/ directory.
    - Place your test images (e.g., test01.pgm, test02.pgm, ...) in the faces_test/ directory.

2) Running the Script:

    - Run the script by providing a variance threshold. This threshold determines how many principal components (eigenfaces) will be used to represent the images.
    
### Example:
```bash
python face_recognition.py 0.95
```

## Principal Component Analysis (PCA)
A technique that is used to reduce the dimensionality of a
dataset while preserving variability as much as possible. We turn the data
into a new set of orthogonal coordinates called the principal components. The PCA algorithm
using Singular Value Decomposition (SVD) is done by centering the data, computing the
covariance matrix, and finally performing SVD.

When centering the data, we subtract the Mean of each feature from the dataset to center the data
around the origin. And then we calculate the covariance matrix of the centered data to
comprehend how the variables vary together and finally we decompose the covariance matrix
into three matrices using SVD.

### Face Reconstruction
The face reconstruction part uses the selected principal components (Eigenfaces) to reconstruct
the original images. This consists of the following steps.
1) Compute the projections of the original images onto the selected principal components.
2) Use the projections and the principal components to reconstruct the images.

With 0.95 as the amount of variance, the eigenfaces are as below.

### Error Calculation

The reconstruction error is calculated as the mean squared error between the original and
reconstructed images.

### Nearest Neighbor Algorithm (Face Recognition)
The nearest neighbor algorithm is a method for face recognition which is to find the most similar
face in the training set. This is done with the following steps.
1) Calculate the L2 distance between the test image and all reconstructed training images.
2) Identify the training image with the smallest distance to the test image


