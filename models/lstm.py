"""

"""
import tensorflow as tf
import tensorflow_probability as tfp


def median_error_fraction(y_true, y_pred):
    error = abs((y_true - y_pred) / y_true)
    return tfp.stats.percentile(error, 50.0, interpolation='midpoint')


class OximetryLSTM(tf.keras.models.Sequential):

    learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                   patience=10, min_lr=0.00001, verbose=1)
    early_stopping_criterion = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=22, verbose=1)
    additional_metrics = [tf.keras.metrics.MeanAbsolutePercentageError()]
    loss_function = tf.keras.losses.MeanAbsoluteError()
    number_of_epochs = 100
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def __init__(self, model_name: str = None):
        super(OximetryLSTM, self).__init__()
        if model_name is None:
            model_name = "OximetryLSTM"
        self.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(41, 1)))
        self.add(tf.keras.layers.LSTM(100, return_sequences=True, activation='relu'))
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Dense(1000))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Dense(1000))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.checkpoint_path = tf.keras.callbacks.ModelCheckpoint(f"{model_name}.h5",
                                                                  monitor='val_median_error_fraction',
                                                                  mode='min', verbose=1,
                                                                  save_best_only=True)

    def compile(self):
        super(OximetryLSTM, self).compile(optimizer=self.optimizer,
                                          loss=self.loss_function,
                                          metrics=[self.additional_metrics, median_error_fraction])

    def fit(self, train_data, validation_data):
        super(OximetryLSTM, self).fit(train_data,
                                      epochs=self.number_of_epochs,
                                      validation_data=validation_data,
                                      callbacks=[self.learning_rate_scheduler,
                                                 self.early_stopping_criterion,
                                                 self.checkpoint_path])

    @staticmethod
    def load(model_name):
        return tf.keras.models.load_model(f'{model_name}.h5', compile=False)