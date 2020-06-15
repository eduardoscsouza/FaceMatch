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
import cv2



model = build_bbox_model()
model.load_weights("best_model.h5")
cam = cv2.VideoCapture(0)
while(True):
    ret, frame = cam.read()

    aux_frame = cv2.cvtColor(cv2.resize(frame, (56, 56), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    aux_frame = (aux_frame/255.0)[np.newaxis, :, :, :]
    x1, y1, width, height = model.predict(aux_frame)[0]
    slice_i = slice(int(frame.shape[0]*y1), int(frame.shape[0]*(y1+height)))
    slice_j = slice(int(frame.shape[1]*x1), int(frame.shape[1]*(x1+width)))
    cut_frame = frame[slice_i, slice_j]

    cv2.imshow('Camera', frame)
    cv2.imshow('Face', cut_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()