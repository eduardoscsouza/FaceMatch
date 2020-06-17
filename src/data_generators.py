from tensorflow.keras.preprocessing.image import ImageDataGenerator



def get_bboxs_generator(bboxs_df, imgs_dir="../data/Img_Resize", batch_size=32,
                    out_image_size=(56, 56), color='rgb', preprocess_func=None,
                    shuffle=True, seed=None,
                    gen_args=None, flow_args=None):

    if gen_args is None:
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

    if flow_args is None:
        flow_args = dict(directory=imgs_dir,
                        x_col="image_id",
                        y_col=list(bboxs_df.columns[1:]),
                        weight_col=None,
                        target_size=out_image_size,
                        color_mode=color,
                        classes=None,
                        class_mode='raw',
                        batch_size=batch_size,
                        shuffle=shuffle,
                        seed=seed,
                        save_to_dir=None,
                        save_prefix='',
                        save_format='jpg',
                        subset=None,
                        interpolation='bilinear',
                        validate_filenames=True)

    return ImageDataGenerator(**gen_args).flow_from_dataframe(bboxs_df, **flow_args)