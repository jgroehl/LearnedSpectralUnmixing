"""

"""

import tensorflow as tf
import tensorflow_probability as tfp


def median_error_fraction(y_true, y_pred):
    error = abs((y_true - y_pred) / y_true)
    return tfp.stats.percentile(error, 50.0, interpolation='midpoint')


def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(41, 1)))
    model.add(tf.keras.layers.LSTM(100, return_sequences=True, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1000))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1000))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


class LSTMParams:

    learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                                   patience=5, min_lr=0.000001, verbose=1)
    early_stopping_criterion = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=11, verbose=1)
    additional_metrics = [tf.keras.metrics.MeanAbsolutePercentageError()]
    loss_function = tf.keras.losses.MeanAbsoluteError()
    number_of_epochs = 100
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    def __init__(self, name: str = None, wl: int = 0):
        super(LSTMParams, self).__init__()
        if name is None:
            name = "OximetryLSTM"

        self.checkpoint_path = tf.keras.callbacks.ModelCheckpoint(f"H:/learned spectral unmixing/models_LSTM/{name}_LSTM_{wl}.h5",
                                                                  monitor='val_median_error_fraction',
                                                                  mode='min', verbose=1,
                                                                  save_best_only=True,
                                                                  save_weights_only=True)

    def compile(self, model):
        model.compile(optimizer=self.optimizer,
                      loss=self.loss_function,
                      metrics=[self.additional_metrics, median_error_fraction],
                      run_eagerly=True)

    def fit(self, train_data, validation_data, model):
        model.fit(train_data,
                  epochs=self.number_of_epochs,
                  validation_data=validation_data,
                  callbacks=[self.learning_rate_scheduler,
                             self.early_stopping_criterion,
                             self.checkpoint_path])

    @staticmethod
    def load(model_name):
        ret_model = get_model()
        ret_model.load_weights(model_name)
        return ret_model
