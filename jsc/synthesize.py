import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ensemble'))

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import yaml
import numpy as np
import h5py
import tensorflow as tf
import hls4ml
import qkeras

keras = tf.keras

# --- Config ---
ENSEMBLE_DIR  = os.path.join(os.path.dirname(__file__), "ensemble/results/ensemble_36")
WEIGHTS_FILE  = os.path.join(ENSEMBLE_DIR, "ensemble_best.weights.h5")
CONFIG_FILE   = os.path.join(ENSEMBLE_DIR, "config.yml")
ENSEMBLE_SIZE = 2
NUM_CLASSES   = 5
# ---------------


def build_member_no_names(config):
    """
    Build a JSC member model WITHOUT explicit layer names so QKeras
    auto-names them as q_dense, q_dense_1, q_dense_2 — matching what
    was saved in ensemble_best.weights.h5.
    """
    q = config["quantization"]
    lq = qkeras.quantizers.quantized_bits(q["logit_total_bits"], q["logit_int_bits"])
    aq = qkeras.quantizers.quantized_relu(q["activation_total_bits"], q["activation_int_bits"])

    x = x_in = keras.layers.Input(tuple(config["data"]["input_shape"]))
    for w in config["model"]["dense_widths"]:
        x = qkeras.qlayers.QDense(w, kernel_quantizer=lq, bias_quantizer=lq)(x)
        x = keras.layers.BatchNormalization()(x)
        x = qkeras.qlayers.QActivation(activation=aq)(x)
    x = qkeras.qlayers.QDense(NUM_CLASSES, kernel_quantizer=lq, bias_quantizer=lq)(x)
    return keras.Model(inputs=x_in, outputs=x)


def load_member_weights_h5py(member_model, weights_file, member_idx):
    """
    Load weights for one ensemble member directly from the h5 file using h5py.
    Matches layers by name and assigns vars by index.
    Skips any layers not found in the file (e.g. a_dense from newer QKeras).
    """
    member_key = "functional" if member_idx == 0 else f"functional_{member_idx}"
    loaded, skipped = 0, 0

    with h5py.File(weights_file, "r") as f:
        if f"layers/{member_key}/layers" not in f:
            raise ValueError(f"Could not find '{member_key}' in {weights_file}. "
                             f"Top-level keys: {list(f['layers'].keys())}")
        member_layers = f[f"layers/{member_key}/layers"]

        for layer in member_model.layers:
            name = layer.name
            if name not in member_layers or "vars" not in member_layers[name]:
                skipped += 1
                continue
            vars_grp = member_layers[name]["vars"]
            for i, var in enumerate(layer.variables):
                if str(i) in vars_grp:
                    var.assign(np.array(vars_grp[str(i)]))
                    loaded += 1

    print(f"  Member {member_idx}: loaded {loaded} weight tensors, skipped {skipped} layers")


def extract_members(config_file, weights_file, size):
    """
    Build standalone member models and load their weights directly from h5.
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)
    input_shape = tuple(config["data"]["input_shape"])

    # Build members and call with shared input so all sublayer variables are created
    shared_input = keras.layers.Input(shape=input_shape, name="ensemble_input")
    members, outputs = [], []
    print("Building member models...")
    for i in range(size):
        m = build_member_no_names(config)
        members.append(m)
        outputs.append(m(shared_input))
    # Create ensemble model just to trigger variable creation in all layers
    keras.Model(inputs=shared_input, outputs=keras.layers.Average()(outputs))

    # Load weights for each member directly via h5py
    print(f"Loading weights from: {weights_file}")
    for i, m in enumerate(members):
        load_member_weights_h5py(m, weights_file, i)
    print("Weights loaded.")

    # Wrap each member as a standalone model for hls4ml
    standalone = []
    for i, m in enumerate(members):
        inp = keras.layers.Input(shape=input_shape, name="input1")
        out = m(inp)
        standalone_model = keras.Model(inputs=inp, outputs=out, name=f"jsc_member_{i+1}")
        standalone.append(standalone_model)

    return standalone, config


def synthesize_member(member_model, member_idx, config):
    output_dir = os.path.join(
        os.path.dirname(__file__),
        f"hls4ml_jsc_member{member_idx}"
    )

    print(f"\n{'='*60}")
    print(f"Synthesizing Member {member_idx}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    member_model.summary()

    hls_config = hls4ml.utils.config_from_keras_model(member_model, granularity="name")

    log_bits = config["quantization"]["logit_total_bits"]
    log_int  = config["quantization"]["logit_int_bits"]
    hls_config["Model"]["Precision"] = f"ap_fixed<{log_bits},{log_int}>"
    hls_config["Model"]["ReuseFactor"] = 1

    import pprint
    print("hls4ml config:")
    pprint.pprint(hls_config)

    hls_model = hls4ml.converters.convert_from_keras_model(
        member_model,
        hls_config=hls_config,
        output_dir=output_dir,
        backend="VivadoAccelerator",
        board="pynq-z2",
        interface="axi_stream",
        driver="python",
    )

    print(f"\nBuilding Member {member_idx} (C synthesis + IP export)...")
    hls_model.build(csim=False, synth=True, export=True)

    ip_path = os.path.join(output_dir, "myproject_prj", "solution1", "impl", "ip")
    print(f"\nMember {member_idx} done.")
    print(f"  IP core: {ip_path}")
    return ip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--member",
        choices=["1", "2", "all"],
        default="all",
        help="Which member to synthesize: 1, 2, or all (default: all)"
    )
    args = parser.parse_args()

    members, config = extract_members(CONFIG_FILE, WEIGHTS_FILE, ENSEMBLE_SIZE)

    to_run = []
    if args.member == "1":
        to_run = [(1, members[0])]
    elif args.member == "2":
        to_run = [(2, members[1])]
    else:
        to_run = [(1, members[0]), (2, members[1])]

    ip_paths = {}
    for idx, model in to_run:
        ip_paths[idx] = synthesize_member(model, idx, config)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for idx, path in ip_paths.items():
        print(f"  Member {idx} IP core: {path}")
    print("\nNext: import these IP folders into Vivado IP catalog.")


if __name__ == "__main__":
    main()
