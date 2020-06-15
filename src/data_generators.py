from tensorflow.keras.preprocessing.image import ImageDataGenerator



def get_bboxs_generator(bboxs_df, imgs_dir="../data/Imgs_Resize",
                    out_image_size=(56, 56), color='rgb', preprocess_func=None,
                    batch_size=32):

    gen_args = dict(featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    zca_epsilon=1e-06,
                    rotation_range=0.0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    brightness_range=None,
                    shear_range=0.0,
                    zoom_range=0.0,
                    channel_shift_range=0.0,
                    fill_mode='nearest',
                    cval=0.0,
                    horizontal_flip=False,
                    vertical_flip=False,
                    rescale=1.0/255.0,
                    preprocessing_function=preprocess_func,
                    data_format='channels_last',
                    validation_split=0.0,
                    dtype=None)

    flow_args = dict(directory=imgs_dir,
                    x_col="image_id",
                    y_col=["x_1", "y_1", "width", "height"],
                    weight_col=None,
                    target_size=out_image_size,
                    color_mode=color,
                    classes=None,
                    class_mode='multi_output',
                    batch_size=batch_size,
                    shuffle=True,
                    seed=None,
                    save_to_dir=None,
                    save_prefix='',
                    save_format='jpg',
                    subset=None,
                    interpolation='bilinear',
                    validate_filenames=True)

    datagen = ImageDataGenerator(**gen_args).flow_from_dataframe(bboxs_df, **flow_args)
    return datagen