from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



def train_model(model, train_datagen, val_datagen,
                epochs=1000, steps_per_epoch=200, validation_steps=100, early_stop_patience=80,
                tensorboard_logdir="logs", best_model_filepath="best_model.h5",
                earlystop_metric="loss", checkpoint_metric="val_mean_absolute_error"):
    tensorboard = TensorBoard(log_dir=tensorboard_logdir,
                            histogram_freq=0,
                            write_graph=False,
                            write_images=False,
                            update_freq='epoch',
                            profile_batch=2,
                            embeddings_freq=0,
                            embeddings_metadata=None)

    early_stop = EarlyStopping(monitor=earlystop_metric,
                            min_delta=0.001,
                            patience=early_stop_patience,
                            verbose=True,
                            mode='auto',
                            baseline=None,
                            restore_best_weights=False)

    checkpoint = ModelCheckpoint(best_model_filepath,
                            monitor=checkpoint_metric,
                            verbose=True,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='auto',
                            save_freq='epoch')

    model.fit(train_datagen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=False,
            callbacks=[tensorboard, early_stop, checkpoint],
            validation_data=val_datagen,
            validation_steps=validation_steps,
            validation_freq=1,
            max_queue_size=50,
            workers=4,
            use_multiprocessing=False,
            sample_weight=None,
            class_weight=None,
            initial_epoch=0)