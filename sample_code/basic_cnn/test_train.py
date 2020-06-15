### Setup Tensorflow ###

import tensorflow as tf
tf.random.set_seed(42)

cpus = tf.config.list_physical_devices(device_type='CPU')
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=6000)])
    except RuntimeError as e:
        print(e)

#tf.config.set_visible_devices(cpus)
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(16)
#tf.config.list_logical_devices()

#########################

import sys
sys.path.append("../../")

from src.utils import *
from src.training import *
from src.data_generators import *
from src.model_builders import *



train_df, val_df, _ = get_train_val_test_dfs("../../data/bboxs.csv", "../../data/splits.csv")
train_datagen, val_datagen = get_bboxs_generator(train_df, imgs_dir="../../data/Img_Resize/"), get_bboxs_generator(val_df, imgs_dir="../../data/Img_Resize/")

model = build_bbox_model()
train_model(model, train_datagen, val_datagen)