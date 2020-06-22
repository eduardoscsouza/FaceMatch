import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import Input, Layer, MaxPooling2D, SeparableConv2D
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

def build_bbox_separable_model(input_size=(56, 56, 3),
                    n_conv_blocks=3, base_conv_n_filters=16,
                    n_dense_layers=2, dense_size=256, dropout_rate=0.25,
                    loss=MeanSquaredError(), optimizer=Adam(),
                    metrics=[MeanAbsoluteError(), MeanBBoxIoU(x2y2=False)]):
    model_in = Input(shape=input_size)

    model = model_in
    for i in range(n_conv_blocks):
        model = SeparableConv2D(base_conv_n_filters*(2**i), (3, 3), padding='same', activation='relu', name="block-{}_conv_0".format(i))(model)
        model = SeparableConv2D(base_conv_n_filters*(2**i), (3, 3), padding='same', activation='relu', name="block-{}_conv_1".format(i))(model)
        model = MaxPooling2D((2, 2), strides=(2, 2), name="block-{}_pool".format(i))(model)

    model = Flatten()(model)
    for i in range(n_dense_layers):
        model = Dense(dense_size, activation='relu', name="dense-{}".format(i))(model)
        model = Dropout(dropout_rate)(model)

    model_out = Dense(4, activation='sigmoid', name="output")(model)
    model = Model(model_in, model_out)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model



def build_feature_extractor(vgg_weights_filepath="../data/vgg_face_weights.h5", extraction_layer_indx=1):
    model_in = Input(shape=(224, 224, 3))

    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model_in)
    model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(128, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(256, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(256, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(256, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(512, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2))(model)

    model_dense_0 = Conv2D(4096, (7, 7), padding='valid', activation='relu')(model)
    model_dense_0 = Dropout(0.5)(model_dense_0)

    model_dense_1 = Conv2D(4096, (1, 1), padding='valid', activation='relu')(model_dense_0)
    model_dense_1 = Dropout(0.5)(model_dense_1)

    model_dense_2 = Conv2D(2622, (1, 1), padding='valid')(model_dense_1)
    model_dense_2 = Flatten()(model_dense_2)

    model_out = Activation('softmax')(model_dense_2)

    vgg_model = Model(model_in, model_out)
    vgg_model.load_weights(vgg_weights_filepath)

    extraction_layers = []
    extraction_layers += [Flatten()(model_dense_0)]
    extraction_layers += [Flatten()(model_dense_1)]
    extraction_layers += [model_dense_2]
    extractor_model = Model(model_in, extraction_layers[extraction_layer_indx])

    return extractor_model



class CosineDistance(Layer):
    def __init__(self, name=None, **kwargs):
        super(CosineDistance, self).__init__(name=name, trainable=False, **kwargs)

    def call(self, anch, comp):
        mult = tf.reduce_sum(anch * comp, axis=1, keepdims=True)
        norm_mult = tf.norm(anch, axis=1, keepdims=True, ord='euclidean') * tf.norm(comp, axis=1, keepdims=True, ord='euclidean')
        dist = 1.0 - tf.math.divide_no_nan(mult, norm_mult)
        return dist

class EuclidianDistanceSquared(Layer):
    def __init__(self, name=None, **kwargs):
        super(EuclidianDistanceSquared, self).__init__(name=name, trainable=False, **kwargs)

    def call(self, anch, comp):
        dist = tf.square(anch - comp)
        dist = tf.reduce_sum(dist, axis=1, keepdims=True)
        return dist

class TripletLoss(Layer):
    def __init__(self, alpha=0.5, name=None, **kwargs):
        super(TripletLoss, self).__init__(name=name, trainable=False, **kwargs)
        self.alpha = alpha

    def call(self, pos_dist, neg_dist):
        tripl = tf.maximum(pos_dist - neg_dist + self.alpha, 0)
        self.add_loss(tripl)
        return tripl

def build_triplet_model(dist_type='eucl', alpha=0.5, vgg_weights_filepath="../data/vgg_face_weights.h5", extraction_layer_indx=1):
    extractor_model = build_feature_extractor(vgg_weights_filepath=vgg_weights_filepath, extraction_layer_indx=extraction_layer_indx)

    anchor_in = Input(shape=(224, 224, 3), name="anchor_in")
    anchor_out = extractor_model(anchor_in)

    pos_in = Input(shape=(224, 224, 3), name="pos_in")
    pos_out = extractor_model(pos_in)

    neg_in = Input(shape=(224, 224, 3), name="neg_in")
    neg_out = extractor_model(neg_in)

    if dist_type == 'cos':
        pos_dist = CosineDistance(name="pos_dist")(anchor_out, pos_out)
        neg_dist = CosineDistance(name="neg_dist")(anchor_out, neg_out)
    else:
        pos_dist = EuclidianDistanceSquared(name="pos_dist")(anchor_out, pos_out)
        neg_dist = EuclidianDistanceSquared(name="neg_dist")(anchor_out, neg_out)

    triplet = TripletLoss(alpha=alpha)(pos_dist, neg_dist)

    triplet_model = Model([anchor_in, pos_in, neg_in], triplet)
    return triplet_model