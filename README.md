# gpu-test-aws

Minimal script to verify GPU provisioning on an AWS instance for the NDAC
AWS hosting pilot.

## What it does

- Reports whether CUDA/a GPU is visible to PyTorch, and basic device info
  (name, memory, compute capability)
- Runs a matrix-multiplication benchmark on CPU and (if available) GPU
- Prints a CPU-vs-GPU speedup number as concrete evidence the GPU is
  actually being used, not just present

## Usage

```bash
pip install torch
python gpu_test.py
```

On a non-GPU instance, it still runs (CPU-only) and reports that clearly,
rather than failing — useful for confirming the script itself works before
testing on the real GPU instance.

