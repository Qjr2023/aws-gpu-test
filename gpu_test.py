"""
gpu_test.py

Minimal GPU provisioning test for AWS.

Purpose: confirm that a GPU instance actually has a working, visible,
CUDA-capable GPU for PyTorch, and show a basic CPU-vs-GPU speed
comparison so the result is meaningful (not just "it ran").

Usage:
    python gpu_test.py

Requires:
    pip install torch   (CUDA build matching the instance's driver)
"""

import time
import torch


def report_environment():
    print("=" * 50)
    print("ENVIRONMENT")
    print("=" * 50)
    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPU count         : {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name}  "
                  f"({props.total_memory / 1e9:.1f} GB, "
                  f"compute capability {props.major}.{props.minor})")
    else:
        print("No GPU detected — running CPU-only.")
    print()


def benchmark(device_name, size=4096, n_iters=10):
    """Time n_iters square matrix multiplications on the given device."""
    device = torch.device(device_name)
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up (important for GPU: excludes one-time CUDA init/context cost)
    for _ in range(3):
        _ = a @ b
    if device_name == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = a @ b
    if device_name == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_iters) * 1000
    print(f"{device_name.upper():<6} avg matmul time ({size}x{size}, "
          f"{n_iters} iters): {avg_ms:.2f} ms")
    return avg_ms


def main():
    report_environment()

    print("=" * 50)
    print("BENCHMARK: matrix multiplication")
    print("=" * 50)

    cpu_time = benchmark("cpu")

    if torch.cuda.is_available():
        gpu_time = benchmark("cuda")
        speedup = cpu_time / gpu_time
        print()
        print(f"Speedup (CPU time / GPU time): {speedup:.1f}x")
        print()
        print("RESULT: GPU is available and functioning correctly.")
    else:
        print()
        print("RESULT: No GPU detected on this instance.")
        print("If this is supposed to be a GPU instance, check:")
        print("  - instance type (must be a GPU instance, e.g. g4dn/g5/p3)")
        print("  - NVIDIA driver installed (run: nvidia-smi)")
        print("  - PyTorch installed with matching CUDA build")


if __name__ == "__main__":
    main()
