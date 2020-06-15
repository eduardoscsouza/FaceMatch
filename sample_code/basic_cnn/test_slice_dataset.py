### Setup Tensorflow ###

import tensorflow as tf
tf.random.set_seed(42)

cpus = tf.config.list_physical_devices(device_type='CPU')
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=6000)])
    except RuntimeError as e:
        print(e)

#########################

from src.model_builders import *

import numpy as np
import pandas as pd
import cv2
import os



df = pd.read_csv("data/bboxs.csv").iloc[:100]
model = build_bbox_model()
model.load_weights("best_model.h5")

in_dir = "data/Img"
true_out_dir = "sample_cuts/true"
pred_out_dir = "sample_cuts/pred"
os.makedirs(true_out_dir, exist_ok=True)
os.makedirs(pred_out_dir, exist_ok=True)
for _, row in df.iterrows():
    img = cv2.imread(os.path.join(in_dir, row.iloc[0]), cv2.IMREAD_COLOR)

    x1, y1, width, height = row.iloc[1:]
    slice_i = slice(int(img.shape[0]*y1), int(img.shape[0]*(y1+height)))
    slice_j = slice(int(img.shape[1]*x1), int(img.shape[1]*(x1+width)))
    cut_img = img[slice_i, slice_j]
    cv2.imwrite(os.path.join(true_out_dir, row.iloc[0]), cut_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    aux_img = cv2.cvtColor(cv2.resize(img, (56, 56), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    aux_img = (aux_img/255.0)[np.newaxis, :, :, :]
    x1, y1, width, height = model.predict(aux_img)[0]
    slice_i = slice(int(img.shape[0]*y1), int(img.shape[0]*(y1+height)))
    slice_j = slice(int(img.shape[1]*x1), int(img.shape[1]*(x1+width)))
    cut_img = img[slice_i, slice_j]
    if (cut_img.shape[0] * cut_img.shape[1]) > 0:
        cv2.imwrite(os.path.join(pred_out_dir, row.iloc[0]), cut_img, [cv2.IMWRITE_JPEG_QUALITY, 95])