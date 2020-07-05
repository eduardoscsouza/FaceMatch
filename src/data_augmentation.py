import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

import sys
sys.path.append("../sample_imgs")

def translate_face_randomly(img, bbox):
    rows, cols = img.shape[:2]

    tx = random.uniform(-bbox[0]*cols, cols-bbox[2]*cols)
    ty = random.uniform(-bbox[1]*rows, rows-bbox[3]*rows)

    M = np.float32([[1, 0, tx], [0, 1, ty]])

    img = cv2.warpAffine(img, M, (cols, rows))
    print(type(img))

    x1 = int(bbox[0]*cols + tx)
    y1 = int(bbox[1]*rows + ty)
    x2 = int(bbox[2]*cols + tx)
    y2 = int(bbox[3]*rows + ty)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img

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

    mn_x = min(p1[0], p2[0], p3[0], p4[0])
    mn_y = min(p1[1], p2[1], p3[0], p4[0])
    mx_x = max(p1[0], p2[0], p3[0], p4[0])
    mx_y = max(p1[1], p2[1], p3[1], p4[1])

    cv2.rectangle(img, (int(bbox[0]*cols), int(bbox[1]*rows)), (int(bbox[2]*cols), int(bbox[3]*rows)), (0, 255, 0), 2)
    cv2.rectangle(img, (mn_x, mn_x), (mx_x, mx_y), (255, 0, 0), 2)

    return img

def apply_transformation_to_images(images, bboxes, transformation):
    images = [transformation(img, bbox) for img, bbox in zip(images, bboxs)]
    return images

img = cv2.imread('../sample_imgs/raw/000001.jpg')
bbox = [0.23227383863080683,0.10334788937409024,0.784841075794621,0.5589519650655022]

img = translate_face_randomly(img, bbox)
cv2.imshow('bla', img)
cv2.waitKey()
