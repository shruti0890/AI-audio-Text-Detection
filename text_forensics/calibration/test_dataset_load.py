"""Quick test: load HC3 dataset and verify samples."""
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

from text_forensics.calibration.run_calibration_hc3 import load_datasets

h, a, src = load_datasets()
print(f"Loaded: {len(h)} human, {len(a)} AI")
print(f"Source: {src[:80]}")
if h:
    print(f"Human[0] ({len(h[0].split())} words): {h[0][:120]}...")
if a:
    print(f"AI[0] ({len(a[0].split())} words): {a[0][:120]}...")
