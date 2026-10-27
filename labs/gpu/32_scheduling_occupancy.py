"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import math
max_threads, registers, shared, max_blocks = 2048, 65536, 65536, 16
threads = 256
for regs_per_thread, shared_per_block in [(32, 8192), (64, 8192), (32, 24576)]:
    resident = min(max_blocks, max_threads // threads,
                   registers // (regs_per_thread * threads),
                   shared // shared_per_block)
    warps = resident * math.ceil(threads / 32)
    occupancy = resident * threads / max_threads
    print(f"registers/thread={regs_per_thread}, shared/block={shared_per_block}: "
          f"blocks={resident}, warps={warps}, occupancy={occupancy:.3f}")
assert 65536 // (64 * 256) == 4
