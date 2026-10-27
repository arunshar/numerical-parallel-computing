"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
lanes = list(range(32))
for stride in [1, 2, 8]:
    addresses = [4 * stride * lane for lane in lanes]
    segments = {a // 32 for a in addresses}
    useful, charged = 128, 32 * len(segments)
    print(f"word_stride={stride}, segments={len(segments)}, useful_fraction={useful/charged:.3f}")
peak_gflops, bandwidth_gbs = 1000.0, 100.0
for intensity in [0.25, 2.0, 10.0, 20.0]:
    bound = min(peak_gflops, bandwidth_gbs * intensity)
    print(f"intensity={intensity:.2f}, roofline_bound={bound:.1f} GFLOP/s")
assert min(peak_gflops, bandwidth_gbs * 2) == 200
