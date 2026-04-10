"""
Inspect ensemble_best.weights.h5 to see how weights are actually named,
then compare against the rebuilt model's layer names.

Run: python inspect_weights.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ensemble'))
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import h5py
import yaml
import tensorflow as tf
import load_model as jsc_load_model
from qensemble import QEnsemble
import tempfile

keras = tf.keras

ENSEMBLE_DIR = os.path.join(os.path.dirname(__file__), "ensemble/results/ensemble_36")
WEIGHTS_FILE = os.path.join(ENSEMBLE_DIR, "ensemble_best.weights.h5")
CONFIG_FILE  = os.path.join(ENSEMBLE_DIR, "config.yml")

# --- 1. Print what's actually stored in the weights file ---
print("=" * 60)
print("Keys inside ensemble_best.weights.h5:")
print("=" * 60)
with h5py.File(WEIGHTS_FILE, "r") as f:
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}  shape={obj.shape}")
    f.visititems(walk)

# --- 2. Rebuild ensemble and print layer names ---
print("\n" + "=" * 60)
print("Rebuilt ensemble model layer names:")
print("=" * 60)
tmp_dir = tempfile.mkdtemp()

def dummy_load_data():
    return (None, None), (None, None)

qe = QEnsemble(
    config_file=CONFIG_FILE,
    load_model_fn=jsc_load_model.load_model,
    load_data_fn=dummy_load_data,
    compile_kwargs={},
    size=2,
    seed=42,
    model_dir=tmp_dir,
)
for layer in qe.ensemble_model.layers:
    print(f"  {layer.name}  vars={[v.name for v in layer.variables]}")
