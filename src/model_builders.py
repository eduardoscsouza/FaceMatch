import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import Input, Lambda, Layer, MaxPooling2D, SeparableConv2D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.metrics import TrueNegatives, FalsePositives, FalseNegatives, TruePositives
from tensorflow.keras.optimizers import Adam, Adamax
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



def build_vgg16_feature_extractor(vgg_weights_filepath="../data/vgg_face_weights.h5", extraction_layer_indx=3, name="vgg16_face_extractor"):
    with tf.name_scope(name) as scope:
        model_in = Input(shape=(224, 224, 3), name="input")

        model = Conv2D(64, (3, 3), padding='same', activation='relu', name="block-0_conv_0")(model_in)
        model = Conv2D(64, (3, 3), padding='same', activation='relu', name="block-0_conv_1")(model)
        model = MaxPooling2D((2, 2), strides=(2, 2), name="block-0_pool")(model)

        model = Conv2D(128, (3, 3), padding='same', activation='relu', name="block-1_conv_0")(model)
        model = Conv2D(128, (3, 3), padding='same', activation='relu', name="block-1_conv_1")(model)
        model = MaxPooling2D((2, 2), strides=(2, 2), name="block-1_pool")(model)

        model = Conv2D(256, (3, 3), padding='same', activation='relu', name="block-2_conv_0")(model)
        model = Conv2D(256, (3, 3), padding='same', activation='relu', name="block-2_conv_1")(model)
        model = Conv2D(256, (3, 3), padding='same', activation='relu', name="block-2_conv_2")(model)
        model = MaxPooling2D((2, 2), strides=(2, 2), name="block-2_pool")(model)

        model = Conv2D(512, (3, 3), padding='same', activation='relu', name="block-3_conv_0")(model)
        model = Conv2D(512, (3, 3), padding='same', activation='relu', name="block-3_conv_1")(model)
        model = Conv2D(512, (3, 3), padding='same', activation='relu', name="block-3_conv_2")(model)
        model = MaxPooling2D((2, 2), strides=(2, 2), name="block-3_pool")(model)

        model = Conv2D(512, (3, 3), padding='same', activation='relu', name="block-4_conv_0")(model)
        model = Conv2D(512, (3, 3), padding='same', activation='relu', name="block-4_conv_1")(model)
        model = Conv2D(512, (3, 3), padding='same', activation='relu', name="block-4_conv_2")(model)
        model = MaxPooling2D((2, 2), strides=(2, 2), name="block-4_pool")(model)

        model_dense_0 = Conv2D(4096, (7, 7), padding='valid', name="dense-0")(model)
        model_actv_0 = Activation('relu', name="dense-0_actv")(model_dense_0)
        model_drop_0 = Dropout(0.5, name="dense-0_drop")(model_actv_0)

        model_dense_1 = Conv2D(4096, (1, 1), padding='valid', name="dense-1")(model_drop_0)
        model_actv_1 = Activation('relu', name="dense-1_actv")(model_dense_1)
        model_drop_1 = Dropout(0.5, name="dense-1_drop")(model_actv_1)

        model_dense_2 = Conv2D(2622, (1, 1), padding='valid', name="dense-2")(model_drop_1)
        model_actv_2 = Activation('softmax', name="dense-2_actv")(model_dense_2)

        vgg_model = Model(model_in, model_actv_2)
        vgg_model.load_weights(vgg_weights_filepath)

        extraction_layer = [model_dense_0, model_actv_0, model_dense_1, model_actv_1, model_dense_2, model_actv_2][extraction_layer_indx]
        extraction_layer = Flatten(name="extract_flat")(extraction_layer)
        extractor_model = Model(model_in, extraction_layer, name=scope)

        return extractor_model



class L2Normalization(Layer):
    def __init__(self, name=None, **kwargs):
        kwargs.update({"trainable":False})
        super(L2Normalization, self).__init__(name=name, **kwargs)

    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)

def add_l2_norm(extractor_model):
    with tf.name_scope(extractor_model.name) as scope:
        l2_norm = L2Normalization(name="l2_normalization")(extractor_model.output)
        return Model(extractor_model.input, l2_norm, name=scope)

def build_vgg16_triplet_extractor(vgg_weights_filepath="../data/vgg_face_weights.h5", extraction_layer_indx=3, name="vgg16_face_extractor"):
    extractor_model = build_vgg16_feature_extractor(vgg_weights_filepath=vgg_weights_filepath, extraction_layer_indx=extraction_layer_indx, name=name)
    return add_l2_norm(extractor_model)

def build_triplet_training_model(extractor_model, dist_type='eucl', alpha=1.0, optimizer=Adamax()):
    triplet_loss = tfa.losses.TripletSemiHardLoss(margin=alpha, distance_metric='angular' if (dist_type=='cos') else 'L2', name="triplet_loss")
    extractor_model.compile(optimizer=optimizer, loss=triplet_loss)
    return extractor_model



class CosineDistance(Layer):
    def __init__(self, name=None, dtype='float32', **kwargs):
        kwargs.update({"trainable":False})
        super(CosineDistance, self).__init__(name=name, dtype=dtype, **kwargs)
        self.aux_one = tf.constant(1.0, dtype=dtype)

    def call(self, inputs):
        anch, comp = inputs[0], inputs[1]
        mult = tf.reduce_sum(anch * comp, axis=1, keepdims=True)
        norm_mult = tf.norm(anch, axis=1, keepdims=True, ord='euclidean') * tf.norm(comp, axis=1, keepdims=True, ord='euclidean')
        dist = self.aux_one - tf.math.divide_no_nan(mult, norm_mult)
        return dist

class EuclidianDistanceSquared(Layer):
    def __init__(self, name=None, dtype='float32', **kwargs):
        kwargs.update({"trainable":False})
        super(EuclidianDistanceSquared, self).__init__(name=name, dtype=dtype, **kwargs)

    def call(self, inputs):
        anch, comp = inputs[0], inputs[1]
        dist = tf.reduce_sum(tf.square(anch - comp), axis=1, keepdims=True)
        return dist

class TripletLoss(Layer):
    def __init__(self, alpha=1.0, name=None, dtype='float32', **kwargs):
        kwargs.update({"trainable":False})
        super(TripletLoss, self).__init__(name=name, dtype=dtype, **kwargs)
        self.alpha = tf.constant(alpha, dtype=dtype)
        self.aux_zero = tf.constant(0.0, dtype=dtype)

    def call(self, inputs):
        pos_dist, neg_dist = inputs[0], inputs[1]
        tripl = tf.maximum(pos_dist - neg_dist + self.alpha, self.aux_zero)
        return tripl

    def get_config(self):
        return {"alpha":float(self.alpha), "name":self.name, "dtype":self.dtype}

def build_triplet_distances_model(extractor_model, dist_type='eucl', alpha=1.0, add_loss=False):
    anchor_in = Input(shape=(224, 224, 3), name="anchor_in")
    anchor_out = extractor_model(anchor_in)

    pos_in = Input(shape=(224, 224, 3), name="pos_in")
    pos_out = extractor_model(pos_in)

    neg_in = Input(shape=(224, 224, 3), name="neg_in")
    neg_out = extractor_model(neg_in)

    if dist_type == 'cos':
        pos_dist = CosineDistance(name="pos_dist")([anchor_out, pos_out])
        neg_dist = CosineDistance(name="neg_dist")([anchor_out, neg_out])
    else:
        pos_dist = EuclidianDistanceSquared(name="pos_dist")([anchor_out, pos_out])
        neg_dist = EuclidianDistanceSquared(name="neg_dist")([anchor_out, neg_out])

    triplet = TripletLoss(alpha=alpha)([pos_dist, neg_dist])
    triplet_model = Model([anchor_in, pos_in, neg_in], triplet)
    triplet_model.add_metric(pos_dist, aggregation='mean', name="pos_dist_mean")
    triplet_model.add_metric(neg_dist, aggregation='mean', name="neg_dist_mean")
    if add_loss:
        triplet_model.add_loss(triplet)
    else:
        triplet_model.add_metric(triplet, aggregation='mean', name="triplet_loss_mean")

    triplet_model.compile(optimizer=Adamax(), loss=None)
    return triplet_model

def build_triplet_classifier_model(extractor_model, dist_type='eucl', threshold=1.0):
    anchor_in = Input(shape=(224, 224, 3), name="anchor_in")
    anchor_out = extractor_model(anchor_in)

    compare_in = Input(shape=(224, 224, 3), name="compare_in")
    compare_out = extractor_model(compare_in)

    if dist_type == 'cos':
        dist = CosineDistance(name="dist")([anchor_out, compare_out])
    else:
        dist = EuclidianDistanceSquared(name="dist")([anchor_out, compare_out])

    model = Lambda(lambda x : tf.cast((x < threshold), tf.float32))(dist)
    model = Model([anchor_in, compare_in], model)
    model.compile(optimizer=Adamax(), loss=None, metrics=[BinaryAccuracy(), Precision(), Recall(),
                TrueNegatives(), FalsePositives(), FalseNegatives(), TruePositives()])

    return model



if __name__ == '__main__':
    m = build_triplet_training_model(build_vgg16_triplet_extractor())
    m.summary()

    import numpy as np

    n_samples, n_feats = 1000, 256
    a, b = np.random.random((n_samples, n_feats)).astype(np.float32), np.random.random((n_samples, n_feats)).astype(np.float32)

    correct = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        aux_a, aux_b = a[i], b[i]
        correct[i] = 1.0 - (np.sum(aux_a*aux_b) / (np.linalg.norm(aux_a) * np.linalg.norm(aux_b)))

    in_1 = Input(shape=(n_feats,))
    in_2 = Input(shape=(n_feats,))
    out = CosineDistance()([in_1, in_2])
    model = Model([in_1, in_2], out)
    assert np.allclose(model.predict([a, b])[:, 0], correct)

    correct = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        aux_a, aux_b = a[i], b[i]
        correct[i] = np.square(np.linalg.norm(aux_a-aux_b))

    in_1 = Input(shape=(n_feats,))
    in_2 = Input(shape=(n_feats,))
    out = EuclidianDistanceSquared()([in_1, in_2])
    model = Model([in_1, in_2], out)
    assert np.allclose(model.predict([a, b])[:, 0], correct)



    import cv2
    import gc
    from glob import glob
    from time import time
    from tensorflow.keras.applications import vgg16
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

    imgs = []
    for img in sorted(glob("test_imgs/*.jpg")):
        imgs += [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)]
    imgs = vgg16.preprocess_input(np.asarray(imgs)).astype(np.float32)

    n_samples = len(imgs)
    split_size = n_samples//3
    d_split = int(np.floor(split_size/2))
    alpha = np.random.random((1,)).astype(np.float32)[0]
    true_class = np.random.randint(2, size=split_size)

    a, b, c = imgs[:split_size], imgs[split_size:2*split_size], imgs[2*split_size:3*split_size]
    d = np.concatenate((b[:d_split], c[d_split:]), axis=0)

    for extraction_layer_indx in range(3):
        extractor_model = build_vgg16_triplet_extractor(extraction_layer_indx=extraction_layer_indx)
        p_a, p_b, p_c, p_d = extractor_model.predict(a), extractor_model.predict(b), extractor_model.predict(c), extractor_model.predict(d)

        correct = np.empty(split_size, dtype=np.float32)
        correct_dp = np.empty(split_size, dtype=np.float32)
        correct_dn = np.empty(split_size, dtype=np.float32)
        for i in range(split_size):
            aux_a, aux_b, aux_c = p_a[i], p_b[i], p_c[i]
            d_p = np.square(np.linalg.norm(aux_a-aux_b))
            d_n = np.square(np.linalg.norm(aux_a-aux_c))
            correct[i] = np.maximum(d_p - d_n + alpha, 0)
            correct_dp[i] = d_p
            correct_dn[i] = d_n
        correct_dp = np.sum(correct_dp) / len(correct_dp)
        correct_dn = np.sum(correct_dn) / len(correct_dn)
        correct_tr = np.sum(correct) / len(correct)

        model = build_triplet_distances_model(extractor_model, dist_type='eucl', alpha=alpha)
        pos_dist, neg_dist, triplet = model.evaluate([a, b, c])[1:]
        assert np.allclose(model.predict([a, b, c])[:, 0], correct, rtol=0.000316228, atol=1e-06)
        assert np.allclose(pos_dist, correct_dp)
        assert np.allclose(neg_dist, correct_dn)
        assert np.allclose(triplet, correct_tr)

        threshold = (pos_dist + neg_dist) / 2.0
        correct_out = np.empty(split_size, dtype=np.float32)
        for i in range(split_size):
            aux_a, aux_d = p_a[i], p_d[i]
            correct_out[i] = np.square(np.linalg.norm(aux_a-aux_d))
            correct_out[i] = correct_out[i] < threshold
        correct_acc = accuracy_score(true_class, correct_out).astype(np.float32)
        correct_prec = precision_score(true_class, correct_out).astype(np.float32)
        correct_rec = recall_score(true_class, correct_out).astype(np.float32)
        correct_tn, correct_fp, correct_fn, correct_tp = confusion_matrix(true_class, correct_out).astype(np.float32).ravel()

        model = build_triplet_classifier_model(extractor_model, dist_type='eucl', threshold=threshold)
        out = model.predict([a, d])[:, 0]
        acc, prec, rec, tn, fp, fn, tp = model.evaluate([a, d], true_class)[1:]
        assert np.all(out == correct_out)
        assert np.all(acc == correct_acc)
        assert np.all(prec == correct_prec)
        assert np.all(rec == correct_rec)
        assert np.all(tn == correct_tn)
        assert np.all(fp == correct_fp)
        assert np.all(fn == correct_fn)
        assert np.all(tp == correct_tp)

        correct = np.empty(split_size, dtype=np.float32)
        correct_dp = np.empty(split_size, dtype=np.float32)
        correct_dn = np.empty(split_size, dtype=np.float32)
        for i in range(split_size):
            aux_a, aux_b, aux_c = p_a[i], p_b[i], p_c[i]
            d_p = 1.0 - (np.sum(aux_a*aux_b) / (np.linalg.norm(aux_a) * np.linalg.norm(aux_b)))
            d_n = 1.0 - (np.sum(aux_a*aux_c) / (np.linalg.norm(aux_a) * np.linalg.norm(aux_c)))
            correct[i] = np.maximum(d_p - d_n + alpha, 0)
            correct_dp[i] = d_p
            correct_dn[i] = d_n
        correct_dp = np.sum(correct_dp) / len(correct_dp)
        correct_dn = np.sum(correct_dn) / len(correct_dn)
        correct_tr = np.sum(correct) / len(correct)

        model = build_triplet_distances_model(extractor_model, dist_type='cos', alpha=alpha)
        pos_dist, neg_dist, triplet = model.evaluate([a, b, c])[1:]
        assert np.allclose(model.predict([a, b, c])[:, 0], correct, rtol=0.000316228, atol=1e-06)
        assert np.allclose(pos_dist, correct_dp)
        assert np.allclose(neg_dist, correct_dn)
        assert np.allclose(triplet, correct_tr)

        threshold = (pos_dist + neg_dist) / 2.0
        correct_out = np.empty(split_size, dtype=np.float32)
        for i in range(split_size):
            aux_a, aux_d = p_a[i], p_d[i]
            correct_out[i] = 1.0 - (np.sum(aux_a*aux_d) / (np.linalg.norm(aux_a) * np.linalg.norm(aux_d)))
            correct_out[i] = correct_out[i] < threshold
        correct_acc = accuracy_score(true_class, correct_out).astype(np.float32)
        correct_prec = precision_score(true_class, correct_out).astype(np.float32)
        correct_rec = recall_score(true_class, correct_out).astype(np.float32)
        correct_tn, correct_fp, correct_fn, correct_tp = confusion_matrix(true_class, correct_out).astype(np.float32).ravel()

        model = build_triplet_classifier_model(extractor_model, dist_type='cos', threshold=threshold)
        out = model.predict([a, d])[:, 0]
        acc, prec, rec, tn, fp, fn, tp = model.evaluate([a, d], true_class)[1:]
        assert np.all(out == correct_out)
        assert np.all(acc == correct_acc)
        assert np.all(prec == correct_prec)
        assert np.all(rec == correct_rec)
        assert np.all(tn == correct_tn)
        assert np.all(fp == correct_fp)
        assert np.all(fn == correct_fn)
        assert np.all(tp == correct_tp)

        del extractor_model, p_a, p_b, p_c, p_d
    del correct, in_1, in_2, out, model
    tf.keras.backend.clear_session()
    gc.collect()

    from tempfile import NamedTemporaryFile
    from tensorflow.keras.models import load_model

    with NamedTemporaryFile(suffix=".h5") as temp_file:
        temp_filename = temp_file.name
        for extraction_layer_indx in range(3):
            for dist_type in ['eucl', 'cos']:
                alpha = np.random.random((1,)).astype(np.float32)[0]
                extractor_model = build_vgg16_triplet_extractor(extraction_layer_indx=extraction_layer_indx)
                for add_loss in [True, False]:
                    model = build_triplet_distances_model(extractor_model, dist_type=dist_type, alpha=alpha, add_loss=add_loss)
                    model.save_weights(temp_filename)
                    model.load_weights(temp_filename)
                    model.save(temp_filename)
                    model = load_model(temp_filename, custom_objects={'L2Normalization':L2Normalization,
                    'CosineDistance':CosineDistance, 'EuclidianDistanceSquared':EuclidianDistanceSquared, 'TripletLoss':TripletLoss})
                    tf.keras.backend.clear_session()

                model = build_triplet_training_model(extractor_model, dist_type=dist_type, alpha=alpha, optimizer=Adamax())
                model.save_weights(temp_filename)
                model.load_weights(temp_filename)
                model.save(temp_filename)
                model = load_model(temp_filename, custom_objects={'L2Normalization':L2Normalization,
                'CosineDistance':CosineDistance, 'EuclidianDistanceSquared':EuclidianDistanceSquared, 'TripletLoss':TripletLoss})
                tf.keras.backend.clear_session()

    '''
    tf.keras.backend.clear_session()
    model = build_triplet_model(dist_type='cos', alpha=alpha)
    model.compile()
    gc.collect()

    model.fit(x=[a, b, c], batch_size=8)
    t0 = time()
    model.fit(x=[a, b, c], batch_size=8)
    print(time() - t0)
    # 15.407991409301758 seconds for 20 batches of 8

    tf.keras.backend.clear_session()
    model = build_triplet_model(dist_type='eucl', alpha=alpha)
    model.compile()
    gc.collect()

    model.fit(x=[a, b, c], batch_size=8)
    t0 = time()
    model.fit(x=[a, b, c], batch_size=8)
    print(time() - t0)
    # 13.582106113433838 seconds for 20 batches of 8
    '''