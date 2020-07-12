from model_builders import build_triplet_distances_model, build_triplet_classifier_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pandas as pd
import os
import gc



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



def evaluate_model(model, datagen, evaluation_steps=2000, generator_queue_size=15, generator_workers=1, use_multiprocessing=True):
    return model.evaluate(datagen, steps=evaluation_steps, callbacks=None,
                        max_queue_size=generator_queue_size, workers=generator_workers, use_multiprocessing=use_multiprocessing,
                        verbose=False)

def get_evaluation_df(model, eval_metrics, include_loss=True):
    metrics_cols = [' '.join([word.capitalize() for word in metric.split('_')]) for metric in model.metrics_names]
    if not include_loss:
        eval_metrics = eval_metrics[1:]
        metrics_cols = metrics_cols[1:]

    return pd.DataFrame([eval_metrics], columns=metrics_cols)



def combine_train_val_dfs(train_df, val_df):
    df = pd.concat([train_df, val_df], ignore_index=True)
    df.insert(0, "Set", ["Train", "Val"])
    return df

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



def evaluate_triplet(extractor_model, exp_name,
                    dist_train_datagen, dist_val_datagen, class_train_datagen, class_val_datagen,
                    dist_type='eucl', alpha=1.0,
                    results_dir="../experiments/results",
                    evaluation_steps=2000, generator_queue_size=15, generator_workers=1, use_multiprocessing=True):

    outfiles_dir = os.path.join(results_dir, exp_name)
    distances_csv_filepath = os.path.join(outfiles_dir, "distances.csv")
    classifier_csv_filepath = os.path.join(outfiles_dir, "classifier_metrics.csv")
    os.makedirs(outfiles_dir, exist_ok=True)

    eval_args = dict(evaluation_steps=evaluation_steps, generator_queue_size=generator_queue_size,
                    generator_workers=generator_workers, use_multiprocessing=use_multiprocessing)

    model = build_triplet_distances_model(extractor_model, dist_type=dist_type, alpha=alpha, add_loss=False)
    train_df = get_evaluation_df(model, evaluate_model(model, dist_train_datagen, **eval_args), include_loss=False)
    val_df = get_evaluation_df(model, evaluate_model(model, dist_val_datagen, **eval_args), include_loss=False)
    threshold = (train_df["Pos Dist Mean"] + train_df["Neg Dist Mean"]) / 2.0
    combine_train_val_dfs(train_df, val_df).to_csv(distances_csv_filepath, index=False)
    del model, train_df, val_df
    gc.collect()

    model = build_triplet_classifier_model(extractor_model, dist_type=dist_type, threshold=threshold)
    train_df = get_evaluation_df(model, evaluate_model(model, class_train_datagen, **eval_args), include_loss=False)
    val_df = get_evaluation_df(model, evaluate_model(model, class_val_datagen, **eval_args), include_loss=False)
    combine_train_val_dfs(train_df, val_df).to_csv(classifier_csv_filepath, index=False)
    del model, train_df, val_df
    gc.collect()