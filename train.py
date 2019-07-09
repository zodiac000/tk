import tensorflow as tf
import pandas as pd
from model import create_model
from dataset import get_ds
import pdb

# tf.enable_eager_execution()

dict_loc = 'saved_dict.csv'
epochs = 5
batch_size = 4

model = create_model()
model.summary()

ds_train, ds_test = get_ds(dict_loc, epochs, batch_size)

tensorboard_cb = tf.keras.callbacks.TensorBoard(update_freq='batch')
callbacks = [
    tensorboard_cb,
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./check_points/mymodel_{epoch}.h5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1)
]
model.fit(ds_train, epochs=epochs, validation_data=ds_test, callbacks=callbacks)

