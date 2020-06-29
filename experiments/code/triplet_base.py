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

from tensorflow.keras.applications import vgg16
import pandas as pd
import os
import gc
import shutil

data_dir = "../../data/"
results_dir = "../results/triplet_base/"
tensorboard_dir = "../tensorboard_logs/triplet_base/"

indvs_csv = os.path.join(data_dir, "indvs.csv")
splits_csv = os.path.join(data_dir, "splits.csv")
imgs_dir = os.path.join(data_dir, "Img_Crop_Resize/")
vgg_weights_filepath = os.path.join(data_dir, "vgg_face_weights.h5")

train_df, val_df, _ = get_train_val_test_dfs(indvs_csv, splits_csv)
gen_kwargs = dict(min_indv_imgs=5, imgs_dir=imgs_dir,
                batch_size=8, out_dtype=np.float32, out_color='rgb',
                resize=False, cv2_resize_inter=cv2.INTER_LINEAR, out_image_size=(224, 224),
                preprocess_func=vgg16.preprocess_input)
for dist in ['eucl', 'cos']:
    exp_name = "dist-{}".format(dist)
    aux_exp_dir = os.path.join(results_dir, exp_name)
    aux_tensorboard_dir = os.path.join(tensorboard_dir, exp_name)
    if not os.path.isfile(os.path.join(aux_exp_dir, "metrics.csv")):
        if os.path.isdir(aux_exp_dir):
            shutil.rmtree(aux_exp_dir)
        if os.path.isdir(aux_tensorboard_dir):
            shutil.rmtree(aux_tensorboard_dir)

        train_datagen = FaceTripleGenerator(train_df, **gen_kwargs)
        val_datagen = FaceTripleGenerator(val_df, **gen_kwargs)
        gc.collect()

        model = build_triplet_model(dist_type=dist, alpha=1.0,
                                    vgg_weights_filepath=vgg_weights_filepath, extraction_layer_indx=1,
                                    extra_out_layer=None, optimizer=Adamax())
        model.summary()

        run_experiment(model, exp_name, train_datagen, val_datagen,
                    epochs=200, steps_per_epoch=50, validation_steps=10,
                    results_dir=results_dir, tensorboard_logdir=tensorboard_dir,
                    best_model_metric="val_loss",earlystop_metric="loss",
                    earlystop_min_delta=0.01, early_stop_patience=20,
                    evaluation_steps=250)

        del train_datagen, val_datagen, model
        gc.collect()

        tf.keras.backend.clear_session()
        gc.collect()