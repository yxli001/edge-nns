import os
import yaml
import argparse

import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import models


def load_data(base_dir="./data/ds8_only", local_id=0):
    train_data = "{}/dec6_ds8_quant/QuantizedInputTrainSetLocal{}.csv".format(
        base_dir, local_id
    )
    train_label = "{}/dec6_ds8_quant/TrainSetLabelLocal{}.csv".format(
        base_dir, local_id
    )
    test_data = "{}/dec6_ds8_quant/QuantizedInputTestSetLocal{}.csv".format(
        base_dir, local_id
    )
    test_label = "{}/dec6_ds8_quant/TestSetLabelLocal{}.csv".format(base_dir, local_id)


    df1 = pd.read_csv(train_data)
    df2 = pd.read_csv(train_label)
    df3 = pd.read_csv(test_data)
    df4 = pd.read_csv(test_label)

    X_train = df1.values
    X_test = df3.values
    y_train = df2.values
    y_test = df4.values

    print("Training set shape         :", X_train.shape)
    print("Training set shape (labels):", y_train.shape)
    print("Test set shape             :", X_test.shape)
    print("Test set shape (labels)    :", y_test.shape)

    return X_train, y_train, X_test, y_test


def load_model(config, pretrained_model=None):
    build_model = getattr(models, config["model"]["name"])

    model = build_model(
      config["model"]["input_shape"], 
      dense_widths=config["model"]["dense_widths"],
      logit_total_bits=config["model"]["logit_total_bits"],
      logit_int_bits=config["model"]["logit_int_bits"],
      activation_total_bits=config["model"]["activation_total_bits"],
      activation_int_bits=config["model"]["activation_int_bits"],
    )

    # Load pretrained model
    if pretrained_model:
        model.load_weights(pretrained_model)
    return model


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # Make log dir
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Build model
    model = load_model(config, pretrained_model=args.pretrained_model)

    model.compile(
        optimizer=Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # default from_logits=False
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    print(model.summary())

    # Train model
    model_name = f"{config['model']['name']}_{'_'.join(map(str, config['model']['dense_widths']))}"
    model_file = os.path.join(save_dir, f"{model_name}.h5")
    model_log_file = os.path.join(save_dir, f"{model_name}_eval.txt")

    callbacks = [
        # EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True),
    ]
    model.fit(
        X_train,
        y_train,
        callbacks=callbacks,
        epochs=150, 
        batch_size=1024,
        validation_split=0.2,
        shuffle=True,
    )

    model.save(model_file)

    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {acc}")
    print(f"Test loss: {loss}", file=open(model_log_file, "w"))
    print(f"Test accuracy: {acc}", file=open(model_log_file, "w"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./model_ckpts/resnet_v1_eembc_quantized_baseline_fkeras/baseline_fkeras.yml",
        help="specify yaml config",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="specify pretrained model file path",
    )
    args = parser.parse_args()

    main(args)
