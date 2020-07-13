### Setup Tensorflow ###

import tensorflow as tf
tf.random.set_seed(42)

cpus = tf.config.list_physical_devices(device_type='CPU')
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    try:
        #tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=4500)])
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
from tensorflow.keras.optimizers import Adamax
import os
import gc
import shutil

data_dir = "../../data/"
results_dir = "../results/grid_search_adamax/"
tensorboard_dir = "../tensorboard_logs/grid_search_adamax/"

bboxs_csv = os.path.join(data_dir, "bboxs_x2y2.csv")
splits_csv = os.path.join(data_dir, "splits.csv")
imgs_dir = os.path.join(data_dir, "Img_Resize/")

for img_size in [56, 112, 224]:
    for n_conv_blocks in [3, 4, 5]:
        if ((img_size / (2**n_conv_blocks)) >= 7):
            for base_conv_n_filters in [16, 32, 64]:
                for dense_size in [64, 256, 1024]:
                    exp_name = "imgsize-{}_convblocks-{}_basefilters-{}_densesize-{}".format(img_size, n_conv_blocks, base_conv_n_filters, dense_size)
                    aux_exp_dir = os.path.join(results_dir, exp_name)
                    aux_tensorboard_dir = os.path.join(tensorboard_dir, exp_name)
                    if not os.path.isfile(os.path.join(aux_exp_dir, "metrics.csv")):
                        if os.path.isdir(aux_exp_dir):
                            shutil.rmtree(aux_exp_dir)
                        if os.path.isdir(aux_tensorboard_dir):
                            shutil.rmtree(aux_tensorboard_dir)

                        train_df, val_df, _ = get_train_val_test_dfs(bboxs_csv, splits_csv)
                        train_datagen = BBoxsGenerator(train_df, imgs_dir=imgs_dir, out_image_size=(img_size, img_size), resize=(img_size!=224), batch_size=16)
                        val_datagen = BBoxsGenerator(val_df, imgs_dir=imgs_dir, out_image_size=(img_size, img_size), resize=(img_size!=224), batch_size=16)
                        del train_df, val_df
                        gc.collect()

                        model = build_bbox_model(input_size=(img_size, img_size, 3),
                                                n_conv_blocks=n_conv_blocks, base_conv_n_filters=base_conv_n_filters,
                                                n_dense_layers=2, dense_size=dense_size, dropout_rate=0.30,
                                                loss=MeanSquaredError(), optimizer=Adamax(),
                                                metrics=[MeanAbsoluteError(), MeanBBoxIoU(x2y2=True)])

                        run_experiment(model, exp_name, train_datagen, val_datagen,
                                    results_dir=results_dir, tensorboard_logdir=tensorboard_dir,
                                    generator_queue_size=50, generator_workers=8, use_multiprocessing=False)

                        del train_datagen, val_datagen, model
                        gc.collect()

                        tf.keras.backend.clear_session()
                        gc.collect()

                        ### Temporary ###
                        sys.exit(0)
                        ###########
