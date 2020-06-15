import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, SGD



def build_bbox_model(input_size=(56, 56, 3),
                    n_conv_blocks=3, base_conv_n_filters=16,
                    n_dense_layers=2, dense_size=256, dropout_rate=0.25,
                    loss=MeanSquaredError(), optimizer=Adam()):
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
    model.compile(loss=loss, optimizer=optimizer,
                metrics=[MeanAbsoluteError()])

    return model