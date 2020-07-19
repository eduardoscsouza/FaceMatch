import numpy as np
import scipy.special

from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show



def make_plot(title, pos_hist, pos_edges, neg_hist, neg_edges):
    p = figure(title=title)
    p.quad(top=pos_hist, bottom=0, left=pos_edges[:-1], right=pos_edges[1:],
        fill_color="blue", line_color="black", alpha=0.5, legend_label="Positive")
    p.quad(top=neg_hist, bottom=0, left=neg_edges[:-1], right=neg_edges[1:],
        fill_color="red", line_color="black", alpha=0.5, legend_label="Negative")

    p.y_range.start = 0
    p.y_range.end = 1500
    p.x_range.start = 0
    p.x_range.end = 2
    p.legend.location = "top_center"
    p.xaxis.axis_label = 'Distance'
    p.yaxis.axis_label = 'Samples Count'
    p.grid.grid_line_color="black"
    return p

measured = np.load("regular_eucl_pos.npy")
pos_hist, pos_edges = np.histogram(measured, bins=100)

measured = np.load("regular_eucl_neg.npy")
neg_hist, neg_edges = np.histogram(measured, bins=100)

p1 = make_plot("Distances Distribution", pos_hist, pos_edges, neg_hist, neg_edges)
output_file('histogram_reg.html')
show(p1)

measured = np.load("separable_eucl_pos.npy")
pos_hist, pos_edges = np.histogram(measured, bins=100)

measured = np.load("separable_eucl_neg.npy")
neg_hist, neg_edges = np.histogram(measured, bins=100)

p2 = make_plot("Distances Distribution", pos_hist, pos_edges, neg_hist, neg_edges)
output_file('histogram_sep.html')
show(p2)

'''
import tensorflow as tf
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=6000)])
'''

'''
import tensorflow as tf
cpus = tf.config.list_physical_devices(device_type='CPU')
tf.config.set_visible_devices(cpus)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
'''

'''
import sys
sys.path.append("src/")

from utils import *
from training import *
from data_generators import *
from model_builders import *

from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import load_model
from time import time
import numpy as np
import os
import gc



n_samples = 500
batch_size = 16
#imgs_224 = np.random.rand(n_samples, 224, 224, 3).astype(np.float32)
#imgs_112 = np.random.rand(n_samples, 112, 112, 3).astype(np.float32)
imgs = np.random.rand(n_samples, 224, 224, 3).astype(np.float32)
gc.collect()

regular = "experiments/results/triplet_from_scratch/dist-eucl/best_model.h5"
separable = "experiments/results/triplet_separable_from_scratch/dist-eucl/best_model.h5"
for path in [regular, separable]:
    model = load_model(path, compile=False,
                    custom_objects={'L2Normalization':L2Normalization})
    model.summary()
    gc.collect()

    t = time()
    model.predict(imgs, batch_size=batch_size, verbose=False)
    t = time() - t
    print(t)

    del model, t
    gc.collect()

    tf.keras.backend.clear_session()
    gc.collect()
'''


'''
def __get_dists_distrib__(model, datagen,
                        evaluation_steps=2000, generator_queue_size=15, generator_workers=1, use_multiprocessing=True,
                        distrib_batch_size=500):

    total_samples = evaluation_steps * datagen.batch_size
    batches_starts = list(range(0, evaluation_steps, distrib_batch_size)) + [evaluation_steps]
    cut_batches_starts = np.asarray(batches_starts) * datagen.batch_size

    tripl, pos, neg = [np.zeros((total_samples,), dtype=np.float32) for _ in range(3)]
    for i in range(len(batches_starts) - 1):
        cur_cut = slice(cut_batches_starts[i], cut_batches_starts[i+1])
        cur_batches = batches_starts[i+1] - batches_starts[i]
        cur_out = model.predict(datagen, steps=cur_batches, callbacks=None,
                                max_queue_size=generator_queue_size, workers=generator_workers, use_multiprocessing=use_multiprocessing,
                                verbose=True)
        tripl[cur_cut], pos[cur_cut], neg[cur_cut] = cur_out[0][:, 0], cur_out[1][:, 0], cur_out[2][:, 0]

    df = pd.DataFrame.from_dict(data={"Triplet Loss Mean":[np.mean(tripl)], "Triplet Loss Std":[np.std(tripl)],
                                    "Pos Dist Mean":[np.mean(pos)], "Pos Dist Std":[np.std(pos)],
                                    "Neg Dist Mean":[np.mean(neg)], "Neg Dist Std":[np.std(neg)]})

    return df, tripl, pos, neg



data_dir = "./data/"
indvs_csv = os.path.join(data_dir, "indvs.csv")
splits_csv = os.path.join(data_dir, "splits.csv")
imgs_dir = os.path.join(data_dir, "Img_Crop_Resize/")
_, val_df, _ = get_train_val_test_dfs(indvs_csv, splits_csv)

eval_gen_args = dict(min_indv_imgs=5, imgs_dir=imgs_dir,
                    batch_size=16,
                    out_dtype=np.float32, out_color='rgb',
                    resize=False, out_image_size=(224, 224), cv2_inter=cv2.INTER_LINEAR,
                    preprocess_func=vgg16.preprocess_input)
path = "experiments/results/triplet_from_scratch/dist-{}/best_model.h5"
for dist in ["eucl", "cos"]:
    model = load_model(path.format(dist), compile=False,
                    custom_objects={'L2Normalization':L2Normalization})
    model = build_triplet_distances_model(model, dist_type=dist, alpha=1.0, add_loss=False)

    val_datagen = TripletDistancesGenerator(val_df, **eval_gen_args)
    df, tripl, pos, neg = __get_dists_distrib__(model, val_datagen, evaluation_steps=2000,
                        generator_queue_size=20, generator_workers=4, use_multiprocessing=False)

    df.to_csv("regular_{}.csv".format(dist), index=False)
    np.save("regular_{}_tripl.npy".format(dist), tripl)
    np.save("regular_{}_pos.npy".format(dist), pos)
    np.save("regular_{}_neg.npy".format(dist), neg)

eval_gen_args = dict(min_indv_imgs=5, imgs_dir=imgs_dir,
                    batch_size=16,
                    out_dtype=np.float32, out_color='rgb',
                    resize=False, out_image_size=(224, 224), cv2_inter=cv2.INTER_LINEAR,
                    preprocess_func=normalize_255)
path = "experiments/results/triplet_separable_from_scratch/dist-{}/best_model.h5"
for dist in ["eucl", "cos"]:
    model = load_model(path.format(dist), compile=False,
                    custom_objects={'L2Normalization':L2Normalization})
    model = build_triplet_distances_model(model, dist_type=dist, alpha=1.0, add_loss=False)

    val_datagen = TripletDistancesGenerator(val_df, **eval_gen_args)
    df, tripl, pos, neg = __get_dists_distrib__(model, val_datagen, evaluation_steps=2000,
                        generator_queue_size=20, generator_workers=4, use_multiprocessing=False)

    df.to_csv("separable_{}.csv".format(dist), index=False)
    np.save("separable_{}_tripl.npy".format(dist), tripl)
    np.save("separable_{}_pos.npy".format(dist), pos)
    np.save("separable_{}_neg.npy".format(dist), neg)
'''


'''
n_samples = 2500
batch_size = 32
#imgs_224 = np.random.rand(n_samples, 224, 224, 3).astype(np.float32)
#imgs_112 = np.random.rand(n_samples, 112, 112, 3).astype(np.float32)
imgs = np.random.rand(n_samples, 112, 112, 3).astype(np.float32)
gc.collect()

regular = "experiments/results/grid_search_adamax/imgsize-112_convblocks-4_basefilters-64_densesize-1024/best_model.h5"
separable = "experiments/results/separable_grid_search_adamax/imgsize-112_convblocks-3_basefilters-64_densesize-256/best_model.h5"
for path in [regular, separable]:
    model = load_model(path, compile=False,
                    custom_objects={'L2Normalization':L2Normalization})
    model.summary()
    gc.collect()

    t = time()
    model.predict(imgs, batch_size=batch_size, verbose=True)
    t = time() - t
    print(t)

    del model, t
    gc.collect()

    tf.keras.backend.clear_session()
    gc.collect()
'''