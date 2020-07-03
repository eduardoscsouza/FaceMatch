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

    x1 = int(bbox[0]*cols + tx)
    y1 = int(bbox[1]*rows + ty)
    x2 = int(bbox[2]*cols + tx)
    y2 = int(bbox[3]*rows + ty)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img

img = cv2.imread('../sample_imgs/raw/000001.jpg')
bbox = [0.23227383863080683,0.10334788937409024,0.784841075794621,0.5589519650655022]

img = translate_face_randomly(img, bbox)
cv2.imshow('bla', img)
cv2.waitKey()
