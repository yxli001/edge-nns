#!/usr/bin/env python3
print("=== DIAGNOSTIC SCRIPT ===")

# 1. Check basic imports
try:
    import pandas as pd
    print("✅ pandas imported")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")
    exit(1)

# 2. Check CSV reading
try:
    df = pd.read_csv('results.csv')
    print(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} cols")
    print(f"   Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ CSV read failed: {e}")
    exit(1)

# 3. Check matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported with Agg backend")
except ImportError as e:
    print(f"❌ matplotlib import failed: {e}")
    exit(1)

# 4. Try to create a simple plot
try:
    plt.figure(figsize=(8, 5))
    plt.scatter(df['num_params'], df['test_accuracy'], alpha=0.5)
    plt.title('Test Plot')
    plt.xlabel('Params')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('test_diagnostic.png')
    print("✅ Test plot created: test_diagnostic.png")
except Exception as e:
    print(f"❌ Plot creation failed: {e}")
    import traceback
    traceback.print_exc()

print("=== DIAGNOSTIC COMPLETE ===")
