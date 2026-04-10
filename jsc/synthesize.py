import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ensemble'))

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import yaml
import tensorflow as tf
import hls4ml
import load_model as jsc_load_model

keras = tf.keras

# --- Config ---
ENSEMBLE_DIR = os.path.join(os.path.dirname(__file__), "ensemble/results/ensemble_36")
WEIGHTS_FILE = os.path.join(ENSEMBLE_DIR, "ensemble_best.weights.h5")
CONFIG_FILE  = os.path.join(ENSEMBLE_DIR, "config.yml")
FPGA_PART    = "xc7z020clg400-1"   # Zynq on PYNQ-Z2
ENSEMBLE_SIZE = 2
# ---------------


def rebuild_ensemble_and_extract_members(config_file, weights_file, size):
    """
    Reconstructs the QEnsemble Keras graph, loads saved weights,
    then returns the two individual member Keras models.
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)

    input_shape = tuple(config["data"]["input_shape"])
    shared_input = keras.layers.Input(shape=input_shape, name="ensemble_input")

    members = []
    outputs = []
    for i in range(size):
        m = jsc_load_model.load_model(config_file)
        members.append(m)
        # mathc the naming used during training (see qensemble.py line 57)
        for layer in m.layers:
            layer._name = f"m{i}_{layer.name}"
        out = m(shared_input)
        outputs.append(out)

    avg_output = keras.layers.Average()(outputs)
    ensemble_model = keras.Model(inputs=shared_input, outputs=avg_output)

    print(f"Loading weights from: {weights_file}")
    ensemble_model.load_weights(weights_file)
    print("Weights loaded.")

    # wrap each member as a standalone model so hls4ml can convert it
    standalone = []
    for i, m in enumerate(members):
        inp = keras.layers.Input(shape=input_shape, name="input1")
        out = m(inp)
        standalone_model = keras.Model(inputs=inp, outputs=out, name=f"jsc_member_{i+1}")
        standalone.append(standalone_model)

    return standalone


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

    # generate hls4ml config from the Keras model
    hls_config = hls4ml.utils.config_from_keras_model(member_model, granularity="name")

    # match the bit widths used during training
    act_bits = config["quantization"]["activation_total_bits"]
    act_int  = config["quantization"]["activation_int_bits"]
    log_bits = config["quantization"]["logit_total_bits"]
    log_int  = config["quantization"]["logit_int_bits"]

    hls_config["Model"]["Precision"] = f"ap_fixed<{log_bits},{log_int}>"
    hls_config["Model"]["ReuseFactor"] = 1

    print("hls4ml config:")
    import pprint
    pprint.pprint(hls_config)

    # convert to HLS project
    hls_model = hls4ml.converters.convert_from_keras_model(
        member_model,
        hls_config=hls_config,
        output_dir=output_dir,
        backend="VivadoAccelerator",
        board="pynq-z2",
        interface="axi_stream",
        driver="python",
    )

    # C synthesis + export IP core
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

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    print("Reconstructing ensemble and extracting individual members...")
    members = rebuild_ensemble_and_extract_members(CONFIG_FILE, WEIGHTS_FILE, ENSEMBLE_SIZE)

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
