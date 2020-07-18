### Setup Tensorflow ###

import tensorflow as tf
tf.random.set_seed(42)

cpus = tf.config.list_physical_devices(device_type='CPU')
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    try:
        #tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=6000)])
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
from tensorflow.keras.models import load_model
import pandas as pd
import os
import gc
import shutil

data_dir = "../../data/"
results_dir = "../results/triplet_no_train/"
tensorboard_dir = "../tensorboard_logs/triplet_no_train/"

indvs_csv = os.path.join(data_dir, "indvs.csv")
splits_csv = os.path.join(data_dir, "splits.csv")
imgs_dir = os.path.join(data_dir, "Img_Crop_Resize/")
vgg_weights_filepath = os.path.join(data_dir, "vgg_face_weights.h5")

train_df, val_df, _ = get_train_val_test_dfs(indvs_csv, splits_csv)
gen_args = dict(min_indv_imgs=5, imgs_dir=imgs_dir,
                batch_n_indvs=4, batch_indv_n_imgs=4,
                out_dtype=np.float32, out_color='rgb',
                resize=False, out_image_size=(224, 224), cv2_inter=cv2.INTER_LINEAR,
                preprocess_func=vgg16.preprocess_input)
eval_gen_args = dict(min_indv_imgs=5, imgs_dir=imgs_dir,
                    batch_size=16,
                    out_dtype=np.float32, out_color='rgb',
                    resize=False, out_image_size=(224, 224), cv2_inter=cv2.INTER_LINEAR,
                    preprocess_func=vgg16.preprocess_input)

for dist in ['eucl', 'cos']:
    exp_name = "dist-{}".format(dist)
    aux_exp_dir = os.path.join(results_dir, exp_name)
    if not os.path.isfile(os.path.join(aux_exp_dir, "classifier_metrics.csv")):
        vgg16_extractor = build_vgg16_triplet_extractor(vgg_weights_filepath=vgg_weights_filepath, extraction_layer_indx=3)

        dist_train_datagen = TripletDistancesGenerator(train_df, **eval_gen_args)
        dist_val_datagen = TripletDistancesGenerator(val_df, **eval_gen_args)
        class_train_datagen = TripletClassifierGenerator(train_df, **eval_gen_args)
        class_val_datagen = TripletClassifierGenerator(val_df, **eval_gen_args)
        gc.collect()

        evaluate_triplet(vgg16_extractor, exp_name=exp_name,
                        dist_train_datagen=dist_train_datagen, dist_val_datagen=dist_val_datagen,
                        class_train_datagen=class_train_datagen, class_val_datagen=class_val_datagen,
                        dist_type=dist, alpha=1.0,
                        results_dir=results_dir,
                        evaluation_steps=2000, generator_queue_size=20, generator_workers=4, use_multiprocessing=False,
                        distrib_batch_size=500, distrib_bins=50000)

        del dist_train_datagen, dist_val_datagen
        del class_train_datagen, class_val_datagen
        del vgg16_extractor
        tf.keras.backend.clear_session()
        gc.collect()