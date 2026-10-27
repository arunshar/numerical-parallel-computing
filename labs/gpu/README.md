# GPU teaching experiments

The numbered Python programs execute CPU references or explicit performance models. They require NumPy and use small deterministic inputs. They do not execute CUDA or establish GPU performance.

You can run the complete CPU collection from the repository root.

```bash
for script in labs/gpu/[0-9][0-9]_*.py; do python "$script"; done
```

The original `native_suite.cu` provides CUDA kernels for vector addition, rectangular tiled matrix multiplication, zero-padded cross-correlation, an atomic histogram, and a block reduction with a host final stage. The host program checks results against simple references, rejects nonfinite comparison values, and reports CUDA errors. The matrix test multiplies a 37-by-29 matrix by a 29-by-35 matrix, which exercises multiple blocks and partial tiles. Its block reduction requires the displayed 256-thread launch. The suite is a teaching implementation with fixed small test sizes.

Native CUDA compilation and execution are **UNVERIFIED** in this edition. The authoring machine did not provide `nvcc` or an NVIDIA GPU. A source review and a passing NumPy model are not substitutes for that validation.

You can compile and run the native suite on a supported CUDA system. These commands describe work to perform and are not a record of completed GPU execution.

```bash
mkdir -p build
nvcc -O2 -std=c++17 labs/gpu/native_suite.cu -o build/gpu_native_suite
./build/gpu_native_suite
compute-sanitizer --tool memcheck ./build/gpu_native_suite
compute-sanitizer --tool racecheck ./build/gpu_native_suite
```

The suite deliberately reports correctness rather than performance. A benchmark should add warmup, repeated measurements, a stated timing scope, device metadata, and appropriate numerical tolerances. A passing fixed-size suite should be expanded to randomized and adversarial shapes before it supports broader correctness claims.
