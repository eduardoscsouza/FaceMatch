import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import MeanMetricWrapper



def __stateless_x2y2_bbox_iou__(y_true, y_pred):
    inter_left = tf.maximum(y_true[:, 0], y_pred[:, 0])
    inter_right = tf.minimum(y_true[:, 2], y_pred[:, 2])
    inter_top = tf.maximum(y_true[:, 1], y_pred[:, 1])
    inter_bottom = tf.minimum(y_true[:, 3], y_pred[:, 3])

    zeros = tf.zeros(shape=(1,))
    inter_width = tf.maximum(inter_right - inter_left, zeros)
    inter_height = tf.maximum(inter_bottom - inter_top, zeros)
    inter_area = inter_width * inter_height

    true_area = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    pred_area = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union_area = true_area + pred_area - inter_area

    return tf.math.divide_no_nan(inter_area, union_area)

def __stateless_bbox_iou__(y_true, y_pred):
    true_x2, true_y2 = y_true[:, 0]+y_true[:, 2], y_true[:, 1]+y_true[:, 3]
    pred_x2, pred_y2 = y_pred[:, 0]+y_pred[:, 2], y_pred[:, 1]+y_pred[:, 3]

    y_true = tf.stack([y_true[:, 0], y_true[:, 1], true_x2, true_y2], axis=1)
    y_pred = tf.stack([y_pred[:, 0], y_pred[:, 1], pred_x2, pred_y2], axis=1)

    return __stateless_x2y2_bbox_iou__(y_true, y_pred)

class MeanBBoxIoU(MeanMetricWrapper):
    def __init__(self, name='mean_bbox_iou', dtype=None, x2y2=False, **kwargs):
        bbox_iou_func = __stateless_x2y2_bbox_iou__ if x2y2 else __stateless_bbox_iou__
        super(MeanBBoxIoU, self).__init__(bbox_iou_func, name=name, dtype=dtype, **kwargs)



def build_bbox_model(input_size=(56, 56, 3),
                    n_conv_blocks=3, base_conv_n_filters=16,
                    n_dense_layers=2, dense_size=256, dropout_rate=0.25,
                    loss=MeanSquaredError(), optimizer=Adam(),
                    metrics=[MeanAbsoluteError()]):
    model_in = Input(shape=input_size)

    model = model_in
    for i in range(n_conv_blocks):
        model = Conv2D(base_conv_n_filters*(2**i), (3, 3), padding='same', activation='relu', name="block-{}_conv_0".format(i))(model)
        model = Conv2D(base_conv_n_filters*(2**i), (3, 3), padding='same', activation='relu', name="block-{}_conv_1".format(i))(model)
        model = MaxPooling2D((2, 2), strides=(2, 2), name="block-{}_pool".format(i))(model)

    model = Flatten()(model)
    for i in range(n_dense_layers):
        model = Dense(dense_size, activation='relu', name="dense-{}".format(i))(model)
        model = Dropout(dropout_rate)(model)

    model_out = Dense(4, activation='sigmoid', name="output")(model)
    model = Model(model_in, model_out)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


import numpy as np
from tensorflow.keras.layers import Lambda

def np_comp(y_true, y_pred):
    left = np.maximum(y_true[0], y_pred[0])
    top = np.maximum(y_true[1], y_pred[1])
    right = np.minimum(y_true[2], y_pred[2])
    bottom = np.minimum(y_true[3], y_pred[3])

    if (left < right and bottom > top):
        true_area = (y_true[2] - y_true[0]) * (y_true[3] - y_true[1])
        pred_area = (y_pred[2] - y_pred[0]) * (y_pred[3] - y_pred[1])
        inter_area = (right - left) * (bottom - top)
        return inter_area / (true_area + pred_area - inter_area)
    else:
        return 0.0

n = 10000
true = np.random.random((n, 4))
true[:, [2, 3]] = true[:, [0, 1]] + np.random.random((n, 2))
true = true.astype(np.float32)

pred = np.random.random((n, 4))
pred[:, [2, 3]] = pred[:, [0, 1]] + np.random.random((n, 2))
pred = pred.astype(np.float32)

correct = np.empty(shape=(n,), dtype=np.float32)
for i in range(n):
    correct[i] = np_comp(true[i], pred[i])

aux_true = tf.convert_to_tensor(true)
aux_pred = tf.convert_to_tensor(pred)
out = __stateless_x2y2_bbox_iou__(aux_true, aux_pred).numpy()

assert all(out == correct)

aux_true = np.copy(true)
aux_pred = np.copy(pred)
aux_true[:, [2, 3]] -= aux_true[:, [0, 1]]
aux_pred[:, [2, 3]] -= aux_pred[:, [0, 1]]
aux_true = tf.convert_to_tensor(aux_true)
aux_pred = tf.convert_to_tensor(aux_pred)
new_out = __stateless_bbox_iou__(aux_true, aux_pred).numpy()

assert np.allclose(new_out, correct, atol=1e-07)

model_in = Input(shape=(4,))
model = Lambda(lambda x : x)(model_in)
model = Model(model_in, model)
model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanBBoxIoU(x2y2=True)])

correct_mean = np.sum(correct) / n

out_mean = model.evaluate(pred, true, batch_size=32)[1]
assert np.allclose(correct_mean, out_mean, atol=1e-07)

model_in = Input(shape=(4,))
model = Lambda(lambda x : x)(model_in)
model = Model(model_in, model)
model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[MeanBBoxIoU(x2y2=False)])

new_out_mean = model.evaluate(aux_pred, aux_true, batch_size=32)[1]
assert np.allclose(correct_mean, new_out_mean, atol=1e-07)