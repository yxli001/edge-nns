import copy
import os
import itertools
import yaml
from load_model import load_model
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

NUM_CLASSES = 5

def gen_configs():
    
    width_options = [
        [32],
        [64],
        [128],
        [256],
        [32, 32],
        [64, 64],
        [128, 128],
        [32, 64],
        [64, 128],
        [128, 256],
        [32, 64, 128],
        [64, 128, 256],
    ]
    
    activation_total_bits_options = [2, 4, 6, 8]
    logits_total_bits_options = [2, 4, 6, 8]
    
    base_config = {
        'data': {
            'name': 'jsc',
            'input_shape': [16],
            'num_classes': 5
        }
    }
    
    configs = []
    
    for widths, activation_total_bits, logits_total_bits in itertools.product(
        width_options,
        activation_total_bits_options,
        logits_total_bits_options,
    ):
        config = copy.deepcopy(base_config)
        
        config['model'] = {
            "dense_widths": widths,
        }
        
        config['quantization'] = {
            "logit_total_bits": logits_total_bits,
            "logit_int_bits": 2 if logits_total_bits > 2 else 1,
            "logit_quantizer": "quantized_bits",
            "activation_total_bits": activation_total_bits,
            "activation_int_bits": 2 if activation_total_bits > 2 else 1,
            "activation_quantizer": "quantized_relu",
            "alpha": 1,
        }
        
        configs.append(config)
    
    return configs


def train_single(config, save_dir):
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # load dataset
    print("Fetching JSC dataset from OpenML...")
    data = fetch_openml('hls4ml_lhc_jets_hlf')
    X = data['data'].values
    y = data['target'].values
    
    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # load model
    model = load_model(config)
    
    # train
    model_file = os.path.join(save_dir, "model.h5")
    model_log_file = os.path.join(save_dir, "eval.txt")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True),
    ]
    
    model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=128,
        validation_split=0.2,
        shuffle=True,
        callbacks=callbacks,
    )
    
    model.save(model_file)
    
    # eval
    loss, acc = model.evaluate(X_test, y_test)
    
    # save
    with open(model_log_file, "w") as f:
        f.write(f"Test loss: {loss}\n")
        f.write(f"Test accuracy: {acc}\n")


def train_all():
    configs = gen_configs()
    
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    for i, config in enumerate(configs):
        model_dir = f"./models/model_{i}"
        config_file = f"{model_dir}/config.yml"
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            with open(config_file, "w") as f:
                yaml.dump(config, f)
        
        print(f"\n[{i+1}/{len(configs)}] Training model {i}")
        train_single(config, model_dir)


if __name__ == "__main__":
    train_all()
