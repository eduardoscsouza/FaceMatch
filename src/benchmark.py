import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime
import sys
sys.path.append("../data")

import utils

# Making TensorFlow use just one process
cpus = tf.config.list_physical_devices(device_type='CPU')
tf.config.set_visible_devices(cpus)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Loading necessary files for each solution
face_cascade = cv2.CascadeClassifier('../data/lbpcascade_frontalface.xml')
model = load_model("../data/benchmark_model.h5")


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
        return (-1, -1, -1, -1)

    x, y, w, h = faces[0]
    return (x, y, x+w, y+h)

"""
Returns the upper-left and lower-right coordinates of the face
using our CNN solution.
"""
def __getBoundingBoxCNN__(image):
    aux_image = cv2.cvtColor(cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    aux_image = (aux_image/255.0)[np.newaxis, :, :, :]
    x1, y1, width, height = model.predict(aux_image)[0]

    x_up_left, y_up_left = int(image.shape[1]*x1), int(image.shape[0]*y1)
    x_down_right, y_down_right = int(image.shape[1]*(x1+width)), int(image.shape[0]*(y1+height))

    return (x_up_left, y_up_left, x_down_right, y_down_right)

"""
Calculates intersection over union between boxA and boxB
Based on https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
"""
def __bb_intersection_over_union__(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


"""
Returns all bounding boxes of given images on image_paths using
one of the methods, that are passed as parameter.
"""
def getBoundingBoxesOfImages(get_bbox_method, images_paths):
    bounding_boxes = []

    for ip in images_paths:
        image = cv2.imread(ip)
        bbox = get_bbox_method(image)
        bounding_boxes.append(bbox)

    return bounding_boxes

"""
Returns the running time to get all bounding boxes of images
on image_paths using one of the methods, that are passed as parameter.
"""
def getRunningTimeForImages(get_bbox_method, images_paths):
    begin_time = datetime.datetime.now()
    getBoundingBoxesOfImages(get_bbox_method, images_paths)
    end_time = datetime.datetime.now()

    return (end_time - begin_time)


"""
Calculates mean between of all bounding boxes Intersection over Union
Ignores cases where one of the bounding boxes is null ((-1, -1, -1, -1))
"""
def getIouMean(bboxesA, bboxesB):
    mean = 0
    cnt = 0

    for i in range(len(bboxesA)):
        if (bboxesA[i] != (-1, -1, -1, -1) and bboxesB[i] != (-1, -1, -1, -1)):
            iou = __bb_intersection_over_union__(bboxesA[i], bboxesB[i])
            mean += iou
            cnt += 1

    return mean/cnt


def main():
    # Getting all images from directory in path
    path = '../sample_imgs/raw'
    images_paths = utils.getFilenamesFromDir(path)


    # Getting running time of both solutions
    cv_time = getRunningTimeForImages(__getBoundingBoxCV__, images_paths)
    cnn_time = getRunningTimeForImages(__getBoundingBoxCNN__, images_paths)

    print("Running time using OpenCV = " + str(cv_time))
    print("Running time using our CNN = " + str(cnn_time))


    # Getting bounding boxes of both solutions
    cv_bboxes = getBoundingBoxesOfImages(__getBoundingBoxCV__, images_paths)
    cnn_bboxes = getBoundingBoxesOfImages(__getBoundingBoxCNN__, images_paths)

    # Calculating mean of Intersection over Union to compare the different results
    mean_iou = getIouMean(cv_bboxes, cnn_bboxes)
    print("Mean of Intersection over Union = " + str(mean_iou))


main()
