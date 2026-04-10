import sys
import os
import argparse
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ensemble'))

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import yaml
import tensorflow as tf
import hls4ml

import load_model as jsc_load_model
from qensemble import QEnsemble

keras = tf.keras

# --- Config ---
ENSEMBLE_DIR  = os.path.join(os.path.dirname(__file__), "ensemble/results/ensemble_36")
WEIGHTS_FILE  = os.path.join(ENSEMBLE_DIR, "ensemble_best.weights.h5")
CONFIG_FILE   = os.path.join(ENSEMBLE_DIR, "config.yml")
ENSEMBLE_SIZE = 2
# ---------------


def extract_members(config_file, weights_file, size):
    """
    Reconstruct the ensemble using the exact same QEnsemble code used during
    training, load the saved weights, then return standalone member models.
    """
    # QEnsemble.__init__ needs a model_dir to copy config into — use a temp dir
    tmp_dir = tempfile.mkdtemp(prefix="qensemble_tmp_")

    # dummy load_data — we only need the graph, not training data
    def dummy_load_data():
        return (None, None), (None, None)

    print("Reconstructing ensemble graph (using QEnsemble)...")
    qe = QEnsemble(
        config_file=config_file,
        load_model_fn=jsc_load_model.load_model,
        load_data_fn=dummy_load_data,
        compile_kwargs={},
        size=size,
        seed=42,
        model_dir=tmp_dir,
    )

    print(f"Loading weights from: {weights_file}")
    qe.ensemble_model.load_weights(weights_file)
    print("Weights loaded.")

    with open(config_file) as f:
        config = yaml.safe_load(f)
    input_shape = tuple(config["data"]["input_shape"])

    # Wrap each member as a standalone model for hls4ml
    standalone = []
    for i, m in enumerate(qe.models):
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
