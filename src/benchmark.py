import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime

import sys
sys.path.append("../data")

# Loading necessary files for each solution
face_cascade = cv2.CascadeClassifier('../data/lbpcascade_frontalface.xml')
model = load_model("../data/benchmark_model.h5")


"""
Returns all filenames from a given directory
"""
def __getFilenamesFromDir__(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))
    return files

"""
Returns the upper-left and lower-right coordinates of the face
using OpenCV solution. Assumes the image has at most one face
"""
def __getBoundingBoxCV__(image):
    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # In case the algorithm didn't find any face
    if (len(faces) == 0):
        return ((-1, -1), (-1, -1))

    x, y, w, h = faces[0]
    return ((x, y), (x+w, y+h))

"""
Returns the upper-left and lower-right coordinates of the face
using our CNN solution.
"""
def __getBoundingBoxCNN__(image):
    aux_image = cv2.cvtColor(cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    aux_image = (aux_image/255.0)[np.newaxis, :, :, :]
    x1, y1, width, height = model.predict(aux_image)[0]

    upper_left = (int(image.shape[1]*x1), int(image.shape[0]*y1))
    lower_right = (int(image.shape[1]*(x1+width)), int(image.shape[0]*(y1+height)))

    return (upper_left, lower_right)


"""
Returns the running time to get all bounding boxes of images
on image_paths using one of the methods, that are passed as parameter.
"""
def getRunningTimeForImages(get_bbox_method, images_paths):
    begin_time = datetime.datetime.now()

    for ip in images_paths:
        image = cv2.imread(ip)
        bbox = get_bbox_method(image)

    end_time = datetime.datetime.now()

    return (end_time - begin_time)


def main():
    # Getting all images from directory in path
    path = '../sample_imgs/raw'
    images_paths = __getFilenamesFromDir__(path)

    # Getting running time of both solutions
    cv_time = getRunningTimeForImages(__getBoundingBoxCV__, images_paths)
    cnn_time = getRunningTimeForImages(__getBoundingBoxCNN__, images_paths)

    print("Running time using OpenCV = " + str(cv_time))
    print("Running time using our CNN = " + str(cnn_time))


main()
