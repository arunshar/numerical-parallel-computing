"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
upload, compute, download = 2.0, 5.0, 2.0
for chunks in [1,2,4,8]:
    serial=chunks*(upload+compute+download)
    ideal=upload+compute+download+(chunks-1)*max(upload,compute,download)
    print(f"chunks={chunks}, sequential={serial:.1f} ms, ideal_pipeline={ideal:.1f} ms, speedup_bound={serial/ideal:.3f}")
interior, halo_exchange, boundary = 6.0,4.0,2.0
serial=interior+halo_exchange+boundary
overlap=max(interior,halo_exchange)+boundary
assert overlap==8.0
print(f"halo_sequential={serial:.1f} ms, halo_ideal_overlap={overlap:.1f} ms")
