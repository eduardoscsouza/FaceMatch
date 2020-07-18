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
results_dir = "../results/triplet_separable_from_scratch/"
tensorboard_dir = "../tensorboard_logs/triplet_separable_from_scratch/"

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
    aux_tensorboard_dir = os.path.join(tensorboard_dir, exp_name)
    if not os.path.isfile(os.path.join(aux_exp_dir, "metrics.csv")):
        if os.path.isdir(aux_exp_dir):
            shutil.rmtree(aux_exp_dir)
        if os.path.isdir(aux_tensorboard_dir):
            shutil.rmtree(aux_tensorboard_dir)

        train_datagen = TripletTrainGenerator(train_df, **gen_args)
        val_datagen = TripletTrainGenerator(val_df, **gen_args)
        gc.collect()

        vgg16_extractor = build_vgg16_triplet_extractor(vgg_weights_filepath=None, extraction_layer_indx=3, separable=True)
        model = build_triplet_training_model(vgg16_extractor, dist_type=dist, alpha=1.0, optimizer=Adamax())
        model.summary()

        run_experiment(model, exp_name, train_datagen, val_datagen,
                    epochs=800, steps_per_epoch=200, validation_steps=20,
                    results_dir=results_dir, tensorboard_logdir=tensorboard_dir,
                    best_model_metric="val_loss", best_model_metric_mode='min',
                    earlystop_metric="loss", earlystop_metric_mode='min',
                    earlystop_min_delta=0.0005, early_stop_patience=50,
                    generator_queue_size=20, generator_workers=4, use_multiprocessing=False,
                    evaluation_steps=2000)

        del train_datagen, val_datagen, model
        gc.collect()

        tf.keras.backend.clear_session()
        gc.collect()

    if ( os.path.isfile(os.path.join(aux_exp_dir, "metrics.csv")) and
    (not os.path.isfile(os.path.join(aux_exp_dir, "classifier_metrics.csv"))) ):
        vgg16_extractor = load_model(os.path.join(aux_exp_dir, "best_model.h5"), compile=False,
                                    custom_objects={'L2Normalization':L2Normalization})

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