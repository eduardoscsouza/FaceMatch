from model_builders import build_triplet_distances_model, build_triplet_classifier_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import gc



# Trains a model, saving the best model during training
# and saving logs to tensorboard
def train_model(model, train_datagen, val_datagen,
                epochs=1000, steps_per_epoch=200, validation_steps=100,
                tensorboard_logdir="../experiments/tensorboard_logs",
                best_model_filepath="best_model.h5", best_model_metric="val_mean_bbox_iou", best_model_metric_mode='max',
                earlystop_metric="loss", earlystop_metric_mode='min', earlystop_min_delta=0.000225, early_stop_patience=80,
                generator_queue_size=15, generator_workers=1, use_multiprocessing=True):

    tensorboard = TensorBoard(log_dir=tensorboard_logdir,
                            histogram_freq=0,
                            write_graph=False,
                            write_images=False,
                            update_freq='epoch',
                            profile_batch=2,
                            embeddings_freq=0,
                            embeddings_metadata=None)

    early_stop = EarlyStopping(monitor=earlystop_metric,
                            min_delta=earlystop_min_delta,
                            patience=early_stop_patience,
                            verbose=True,
                            mode=earlystop_metric_mode,
                            baseline=None,
                            restore_best_weights=False)

    checkpoint = ModelCheckpoint(filepath=best_model_filepath,
                            monitor=best_model_metric,
                            verbose=True,
                            save_best_only=True,
                            save_weights_only=True,
                            mode=best_model_metric_mode,
                            save_freq='epoch')

    model.fit(train_datagen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=False,
            callbacks=[tensorboard, early_stop, checkpoint],
            validation_data=val_datagen,
            validation_steps=validation_steps,
            validation_freq=1,
            max_queue_size=generator_queue_size,
            workers=generator_workers,
            use_multiprocessing=use_multiprocessing,
            sample_weight=None,
            class_weight=None,
            initial_epoch=0)



# Evaluateas a model with its internal metrics
def evaluate_model(model, datagen, evaluation_steps=2000, generator_queue_size=15, generator_workers=1, use_multiprocessing=True):
    return model.evaluate(datagen, steps=evaluation_steps, callbacks=None,
                        max_queue_size=generator_queue_size, workers=generator_workers, use_multiprocessing=use_multiprocessing,
                        verbose=False)

# Generetas a dataframe from a list of metrics
def get_evaluation_df(model, eval_metrics, include_loss=True):
    metrics_cols = [' '.join([word.capitalize() for word in metric.split('_')]) for metric in model.metrics_names]
    if not include_loss:
        eval_metrics = eval_metrics[1:]
        metrics_cols = metrics_cols[1:]

    return pd.DataFrame([eval_metrics], columns=metrics_cols)



# Combines metrics dataframes
def combine_train_val_dfs(train_df, val_df):
    df = pd.concat([train_df, val_df], ignore_index=True)
    df.insert(0, "Set", ["Train", "Val"])
    return df

# Runs an experiment, by training and evaluating a model
def run_experiment(model, exp_name, train_datagen, val_datagen,
                results_dir="../experiments/results",
                epochs=1000, steps_per_epoch=200, validation_steps=50,
                tensorboard_logdir="../experiments/tensorboard_logs",
                best_model_metric="val_mean_bbox_iou", best_model_metric_mode='max',
                earlystop_metric="loss", earlystop_metric_mode='min', earlystop_min_delta=0.001, early_stop_patience=80,
                generator_queue_size=15, generator_workers=1, use_multiprocessing=True,
                evaluation_steps=2000):

    tensorboard_logdir = os.path.join(tensorboard_logdir, exp_name)
    outfiles_dir = os.path.join(results_dir, exp_name)
    best_model_filepath = os.path.join(outfiles_dir, "best_model.h5")
    metrics_csv_filepath = os.path.join(outfiles_dir, "metrics.csv")
    os.makedirs(outfiles_dir, exist_ok=True)

    train_model(model, train_datagen, val_datagen,
                epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                tensorboard_logdir=tensorboard_logdir,
                best_model_filepath=best_model_filepath, best_model_metric=best_model_metric, best_model_metric_mode=best_model_metric_mode,
                earlystop_metric=earlystop_metric, earlystop_metric_mode=earlystop_metric_mode,
                earlystop_min_delta=earlystop_min_delta, early_stop_patience=early_stop_patience,
                generator_queue_size=generator_queue_size, generator_workers=generator_workers, use_multiprocessing=use_multiprocessing)
    model.load_weights(best_model_filepath)
    model.save(best_model_filepath, overwrite=True, include_optimizer=False, save_format='h5')
    gc.collect()

    eval_args = dict(evaluation_steps=evaluation_steps, generator_queue_size=generator_queue_size,
                    generator_workers=generator_workers, use_multiprocessing=use_multiprocessing)
    train_df = get_evaluation_df(model, evaluate_model(model, train_datagen, **eval_args))
    val_df = get_evaluation_df(model, evaluate_model(model, val_datagen, **eval_args))
    combine_train_val_dfs(train_df, val_df).to_csv(metrics_csv_filepath, index=False)
    gc.collect()



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
                                verbose=False)
        tripl[cur_cut], pos[cur_cut], neg[cur_cut] = cur_out[0][:, 0], cur_out[1][:, 0], cur_out[2][:, 0]

    df = pd.DataFrame.from_dict(data={"Triplet Loss Mean":[np.mean(tripl)], "Triplet Loss Std":[np.std(tripl)],
                                    "Pos Dist Mean":[np.mean(pos)], "Pos Dist Std":[np.std(pos)],
                                    "Neg Dist Mean":[np.mean(neg)], "Neg Dist Std":[np.std(neg)]})

    return df, tripl, pos, neg

def __get_best_threshold__(pos, neg, distrib_bins=50000):
    pos, neg = np.sort(pos), np.sort(neg)

    stop = np.amin([np.amax(neg), np.mean(neg)+3*np.std(neg)])
    thresholds = list(np.linspace(0, stop, num=distrib_bins)) + [stop]

    aux_len = len(neg)
    corrects = np.asarray([[thre, np.searchsorted(pos, thre) + (aux_len - np.searchsorted(neg, thre))] for thre in thresholds])
    threshold = corrects[np.argmax(corrects, axis=0)[1], 0]

    return threshold

def __get_classifier_metrics__(pos, neg, threshold):
    pred = np.concatenate([(pos < threshold), (neg < threshold)], axis=0).astype(np.int32)
    true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0).astype(np.int32)

    acc = accuracy_score(true, pred)
    prec = precision_score(true, pred)
    rec = recall_score(true, pred)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    df = pd.DataFrame.from_dict(data={"Binary Accuracy":[acc], "Precision":[prec], "Recall":[rec],
                                    "True Negatives":[tn], "False Positives":[fp],
                                    "False Negatives":[fn], "True Positives":[tp]})

    return df

# Evaluatioan specific for the triplet model
# Calculates the mean distances, and generates a classifier
# using those distances to get a threshold for classification
def evaluate_triplet(extractor_model, exp_name,
                    dist_train_datagen, dist_val_datagen, class_train_datagen, class_val_datagen,
                    dist_type='eucl', alpha=1.0,
                    results_dir="../experiments/results",
                    evaluation_steps=2000, generator_queue_size=15, generator_workers=1, use_multiprocessing=True,
                    distrib_batch_size=500, distrib_bins=50000):

    outfiles_dir = os.path.join(results_dir, exp_name)
    distances_csv_filepath = os.path.join(outfiles_dir, "distances.csv")
    classifier_csv_filepath = os.path.join(outfiles_dir, "classifier_metrics.csv")
    os.makedirs(outfiles_dir, exist_ok=True)

    distrib_args = dict(evaluation_steps=evaluation_steps, generator_queue_size=generator_queue_size,
                    generator_workers=generator_workers, use_multiprocessing=use_multiprocessing,
                    distrib_batch_size=distrib_batch_size)

    model = build_triplet_distances_model(extractor_model, dist_type=dist_type, alpha=alpha, add_loss=False)
    train_df, _, train_pos, train_neg = __get_dists_distrib__(model, dist_train_datagen, **distrib_args)
    val_df, _, val_pos, val_neg = __get_dists_distrib__(model, dist_val_datagen, **distrib_args)
    combine_train_val_dfs(train_df, val_df).to_csv(distances_csv_filepath, index=False)
    del model, train_df, val_df
    gc.collect()

    threshold = __get_best_threshold__(train_pos, train_neg, distrib_bins=distrib_bins)
    train_df = __get_classifier_metrics__(train_pos, train_neg, threshold)
    val_df = __get_classifier_metrics__(val_pos, val_neg, threshold)
    combine_train_val_dfs(train_df, val_df).to_csv(classifier_csv_filepath, index=False)
    del train_df, val_df
    gc.collect()