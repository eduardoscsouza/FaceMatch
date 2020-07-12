### Setup Tensorflow ###

import tensorflow as tf
tf.random.set_seed(42)

cpus = tf.config.list_physical_devices(device_type='CPU')
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    try:
        #tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=6500)])
    except RuntimeError as e:
        print(e)

#tf.config.set_visible_devices(cpus)
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(16)
#tf.config.list_logical_devices()

#########################

### Setup Local Libs ###

import sys
sys.path.append("../../src/")

from utils import *
from training import *
from data_generators import *
from model_builders import *

#########################

from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam, Adamax
from skimage import feature
import cv2
import os
import gc
import shutil

data_dir = "../../data/"
bboxs_csv = os.path.join(data_dir, "bboxs_x2y2.csv")
splits_csv = os.path.join(data_dir, "splits.csv")
imgs_dir = os.path.join(data_dir, "Img_Resize/")

tops = get_best_exps("../results/grid_search", top_k=1)
results_dir = "../results/tops/regular/"
tensorboard_dir = "../tensorboard_logs/tops/regular/"
os.makedirs(results_dir, exist_ok=True)
for top in tops:
    top[1].to_csv(os.path.join(results_dir, "original_metrics.csv"))
    img_size, n_conv_blocks, base_conv_n_filters, dense_size = [int(val.split("-")[1]) for val in top[0].split(os.sep)[-2].split("_")]

    for color in ['hsv', 'gray']:
        def preproc(imgs):
            cv2_col = cv2.COLOR_RGB2HSV if (color=='hsv') else cv2.COLOR_RGB2GRAY
            imgs = np.stack([cv2.cvtColor(img, cv2_col) for img in imgs]) / 255.0
            return imgs

        exp_name = "imgsize-{}_convblocks-{}_basefilters-{}_densesize-{}_color-{}".format(img_size, n_conv_blocks, base_conv_n_filters, dense_size, color)

        train_df, val_df, _ = get_train_val_test_dfs(bboxs_csv, splits_csv)
        train_datagen = BBoxsGenerator(train_df, imgs_dir=imgs_dir, out_image_size=(img_size, img_size), resize=(img_size!=224), preprocess_func=preproc)
        val_datagen = BBoxsGenerator(val_df, imgs_dir=imgs_dir, out_image_size=(img_size, img_size), resize=(img_size!=224), preprocess_func=preproc)
        del train_df, val_df
        gc.collect()

        model = build_bbox_model(input_size=(img_size, img_size, 3 if (color=='hsv') else 1),
                                n_conv_blocks=n_conv_blocks, base_conv_n_filters=base_conv_n_filters,
                                n_dense_layers=2, dense_size=dense_size, dropout_rate=0.30,
                                loss=MeanSquaredError(), optimizer=Adam(),
                                metrics=[MeanAbsoluteError(), MeanBBoxIoU(x2y2=True)])

        run_experiment(model, exp_name, train_datagen, val_datagen,
                    results_dir=results_dir, tensorboard_logdir=tensorboard_dir,
                    generator_queue_size=50, generator_workers=8, use_multiprocessing=False)

        del train_datagen, val_datagen, model
        gc.collect()

        tf.keras.backend.clear_session()
        gc.collect()

    def preproc(imgs):
        imgs = np.stack([feature.canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)) for img in imgs]) / 255.0
        return imgs

    exp_name = "imgsize-{}_convblocks-{}_basefilters-{}_densesize-{}_canny".format(img_size, n_conv_blocks, base_conv_n_filters, dense_size)

    train_df, val_df, _ = get_train_val_test_dfs(bboxs_csv, splits_csv)
    train_datagen = BBoxsGenerator(train_df, imgs_dir=imgs_dir, out_image_size=(img_size, img_size), resize=(img_size!=224), preprocess_func=preproc)
    val_datagen = BBoxsGenerator(val_df, imgs_dir=imgs_dir, out_image_size=(img_size, img_size), resize=(img_size!=224), preprocess_func=preproc)
    del train_df, val_df
    gc.collect()

    model = build_bbox_model(input_size=(img_size, img_size, 1),
                            n_conv_blocks=n_conv_blocks, base_conv_n_filters=base_conv_n_filters,
                            n_dense_layers=2, dense_size=dense_size, dropout_rate=0.30,
                            loss=MeanSquaredError(), optimizer=Adam(),
                            metrics=[MeanAbsoluteError(), MeanBBoxIoU(x2y2=True)])

    run_experiment(model, exp_name, train_datagen, val_datagen,
                results_dir=results_dir, tensorboard_logdir=tensorboard_dir,
                generator_queue_size=50, generator_workers=8, use_multiprocessing=False)

    del train_datagen, val_datagen, model
    gc.collect()

    tf.keras.backend.clear_session()
    gc.collect()