import os
import tensorflow as tf
from load_model import load_model

NUM_CLASSES = 10

def ensure_gpu_or_fail():
  gpus = tf.config.list_physical_devices("GPU")
  if not gpus:
    raise RuntimeError("No GPUs visible to TensorFlow. Aborting.")
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  tf.config.set_soft_device_placement(False)

def train(config, save_file):
  ensure_gpu_or_fail()
  if not os.path.exists("./models"):
    os.makedirs("./models")

  # load dataset
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  x_train = x_train / 255.
  x_test = x_test / 255.

  y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

  # load model
  model = load_model(config)

  model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
  )
  callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(save_file, monitor='val_loss', save_best_only=True),
  ]

  model.fit(
    x_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    callbacks=callbacks,
  )
 
  # eval
  loss, acc = model.evaluate(x_test, y_test)

  return model, loss, acc