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
                    metrics=[MeanAbsoluteError(), MeanBBoxIoU(x2y2=False)]):
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