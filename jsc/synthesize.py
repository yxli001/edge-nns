import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ensemble'))

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import yaml
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


def extract_members(config_file, weights_file, size):
    """
    Rebuild the ensemble graph using auto-named layers (matching the saved
    weight names: q_dense, q_dense_1, ...) then load weights and return
    each member as a standalone model.
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)
    input_shape = tuple(config["data"]["input_shape"])

    shared_input = keras.layers.Input(shape=input_shape, name="ensemble_input")

    members = []
    outputs = []
    print("Reconstructing ensemble graph (auto-named layers)...")
    for i in range(size):
        m = build_member_no_names(config)
        members.append(m)
        out = m(shared_input)
        outputs.append(out)

    avg_output = keras.layers.Average()(outputs)
    ensemble_model = keras.Model(inputs=shared_input, outputs=avg_output)

    print(f"Loading weights from: {weights_file}")
    ensemble_model.load_weights(weights_file)
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
