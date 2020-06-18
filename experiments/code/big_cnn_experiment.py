### Setup Tensorflow ###

import tensorflow as tf
tf.random.set_seed(42)

cpus = tf.config.list_physical_devices(device_type='CPU')
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
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

import os

exp_name = "big_cnn"

data_dir = "../../data/"
results_dir = "../results/"
tensorboard_dir = "../tensorboard_logs/"

bboxs_csv = os.path.join(data_dir, "bboxs.csv")
splits_csv = os.path.join(data_dir, "splits.csv")
imgs_dir = os.path.join(data_dir, "Img_Resize/")

train_df, val_df, _ = get_train_val_test_dfs(bboxs_csv, splits_csv)
train_datagen = get_bboxs_generator(train_df, imgs_dir=imgs_dir)
val_datagen = get_bboxs_generator(val_df, imgs_dir=imgs_dir)

model = build_bbox_model(input_size=(56, 56, 3),
                    n_conv_blocks=3, base_conv_n_filters=64,
                    n_dense_layers=2, dense_size=1024, dropout_rate=0.4)
run_experiment(model, exp_name, train_datagen, val_datagen,
            results_dir=results_dir, tensorboard_logdir=tensorboard_dir)