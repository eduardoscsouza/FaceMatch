import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

import sys
sys.path.append("../sample_imgs")

"""
Translates images randomly without taking the bounding box out of
the image. On the parts that weren't on the original images, pads
with edge.
"""
def translate_face_randomly(img, bbox):
    rows, cols = img.shape[:2]

    tx = random.uniform(-bbox[0]*cols, cols-bbox[2]*cols)
    ty = random.uniform(-bbox[1]*rows, rows-bbox[3]*rows)

    tx = int(tx)
    ty = int(ty)

    M = np.float32([[1, 0, tx], [0, 1, ty]])

    img = cv2.warpAffine(img, M, (cols, rows))

    if (ty >= 0):
        if (tx >= 0):
            img = img[ty:, tx:]
            img = np.stack([np.pad(img[:,:,c], ((ty, 0), (tx, 0)), mode='edge') for c in range(3)], axis=2)
        else:
            img = img[ty:, :cols+tx]
            img = np.stack([np.pad(img[:,:,c], ((ty, 0), (0, np.abs(tx))), mode='edge') for c in range(3)], axis=2)
    else:
        if (tx >= 0):
            img = img[:rows+ty, tx:]
            img = np.stack([np.pad(img[:,:,c], ((0, np.abs(ty)), (tx, 0)), mode='edge') for c in range(3)], axis=2)
        else:
            img = img[:rows+ty, :cols+tx]
            img = np.stack([np.pad(img[:,:,c], ((0, np.abs(ty)), (0, np.abs(tx))), mode='edge') for c in range(3)], axis=2)
    
    x1 = (bbox[0]*cols + tx)/cols
    y1 = (bbox[1]*rows + ty)/rows
    x2 = (bbox[2]*cols + tx)/cols
    y2 = (bbox[3]*rows + ty)/rows

    bbox = np.array([x1, y1, x2, y2])

    return img, bbox

"""
Rotates images randomly on the interval [-45 degrees, 45 degrees]. 
"""
def rotate_face_randomly(img, bbox):
    rows, cols = img.shape[:2]

    cx = (cols*bbox[0] + cols*bbox[2]) / 2
    cy = (rows*bbox[1] + rows*bbox[3]) / 2

    M = cv2.getRotationMatrix2D((cx, cy), random.uniform(-45, 45), 1) 

    img = cv2.warpAffine(img, M, (cols, rows))

    p1 = [[bbox[0]*cols], [bbox[1]*rows], [1]]
    p2 = [[bbox[2]*cols], [bbox[1]*rows], [1]]
    p3 = [[bbox[0]*cols], [bbox[3]*rows], [1]]
    p4 = [[bbox[2]*cols], [bbox[3]*rows], [1]]

    p1 = np.matmul(M, p1).flatten().astype(int)
    p2 = np.matmul(M, p2).flatten().astype(int)
    p3 = np.matmul(M, p3).flatten().astype(int)
    p4 = np.matmul(M, p4).flatten().astype(int)

    mn_x = min(p1[0], p2[0], p3[0], p4[0])/rows
    mn_y = min(p1[1], p2[1], p3[0], p4[0])/cols
    mx_x = max(p1[0], p2[0], p3[0], p4[0])/rows
    mx_y = max(p1[1], p2[1], p3[1], p4[1])/cols

    bbox = np.array([mn_x, mn_y, mx_x, mx_y])

    return img, bbox

"""
Apply some transformation on a set of images and bounding boxes.
- 'images' and 'bboxes' must have the same size
- All the images must have the same dimensions
- 'transformation' may get an image and a bounding box as parameters
    and also get an image and a bounding box as return
"""
def apply_transformation_to_images(images, bboxes, transformation):
    imgs_bboxes = np.array([transformation(img, bbox) for img, bbox in zip(images, bboxes)])
    images = imgs_bboxes[:, 0]
    bboxes = imgs_bboxes[:, 1]
    return images, bboxes

"""
Example of usage:
img1 bbox = [0.23227383863080683,0.10334788937409024,0.784841075794621,0.5589519650655022]
img2 bbox = [0.1702127659574468,0.15824915824915825,0.6926713947990544,0.6734006734006734]
img3 bbox = [0.432,0.2099644128113879,0.614,0.6583629893238434]

img = cv2.imread('../sample_imgs/raw/000003.jpg')
bbox = [0.432,0.2099644128113879,0.614,0.6583629893238434]

imgs = np.expand_dims(img, axis=0)
bboxes = np.expand_dims(bbox, axis=0)
imgs = np.concatenate((imgs, imgs))
bboxes = np.concatenate((bboxes, bboxes))

imgs, bboxes = apply_transformation_to_images(imgs, bboxes, translate_face_randomly)

rows, cols, _ = imgs[0].shape

imgs[0] = cv2.rectangle(imgs[0], (int(bboxes[0][0]*cols), int(bboxes[0][1]*rows)), (int(bboxes[0][2]*cols), int(bboxes[0][3]*rows)), (255, 0, 0), 2)

cv2.imshow('bla', imgs[0])
cv2.waitKey()
"""
