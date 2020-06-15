# This code performs the Canny Edge Detection
# Its based on the following tutorial: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

import cv2
import numpy as np

from utils import normalize



# Applies convolution on a grayscaled image with given kernel with FFT
def __convFFT__(image, kernel):
    kernel_t = np.fft.fft2(kernel, s=(image.shape[:2]), axes=(0, 1))
    image_t = np.fft.fft2(image, axes=(0, 1))

    image_t = image_t * kernel_t[:, :]
    image_t = np.fft.ifft2(image_t, axes=(0, 1)).real

    return image_t

def __gaussianFilter__(image):
    t = np.linspace(-10, 10, 30)
    bump = np.exp(-0.1*t**2)
    bump /= np.trapz(bump)
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

    image_t = __convFFT__(image, kernel)
    image_t = normalize(image_t)

    return image_t

def __sobelFilter__(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = __convFFT__(image, Kx)
    Iy = __convFFT__(image, Ky)

    G = np.hypot(Ix, Iy)
    G = normalize(G)
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def __non_max_suppression__(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180


    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255
            r = 255

            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0

    return Z

def __double_threshold__(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def __hysteresis__(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def cannyEdgeDetection(image):
    image_f = __gaussianFilter__(image)
    image_f, theta = __sobelFilter__(image_f)
    image_f = __non_max_suppression__(image_f, theta)
    image_f, weak, strong = __double_threshold__(image_f)
    image_f = __hysteresis__(image_f, weak, strong)

    return image_f