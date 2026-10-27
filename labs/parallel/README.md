# Parallel-track experiments

The numbered Python files are deterministic models or small algorithmic correctness checks. You can run one with `python labs/parallel/17-architecture-cache.py`. You can run every Python example with `python labs/parallel/run_all.py`. These files require Python and NumPy. The saved evidence records outputs from the local validation run.

The C examples are original native implementations. They use at most four threads or ranks. Their checks establish correctness on small local inputs, and they do not establish distributed-cluster speedup.

## You can build the pthread example with a C compiler.

```bash
cc -std=c11 -O2 -Wall -Wextra -pthread labs/parallel/pthread_sum.c -o /tmp/pthread_sum
/tmp/pthread_sum
```

## You can build the OpenMP example with an OpenMP-capable compiler.

A GCC installation with OpenMP support can use this command.

```bash
gcc -std=c11 -O2 -Wall -Wextra -fopenmp labs/parallel/openmp_sum.c -o /tmp/openmp_sum
/tmp/openmp_sum
```

The local Apple Clang validation used Homebrew libomp with these flags. Other platforms need their own compiler and library configuration.

```bash
clang -std=c11 -O2 -Wall -Wextra -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -Wl,-rpath,/opt/homebrew/opt/libomp/lib -lomp labs/parallel/openmp_sum.c -o /tmp/openmp_sum
/tmp/openmp_sum
```

## You can build the MPI examples with the MPI compiler wrapper.

```bash
mpicc -std=c11 -O2 -Wall -Wextra labs/parallel/mpi_exchange.c -o /tmp/mpi_exchange
mpiexec -n 4 /tmp/mpi_exchange
mpicc -std=c11 -O2 -Wall -Wextra labs/parallel/mpi_collectives.c -o /tmp/mpi_collectives
mpiexec -n 4 /tmp/mpi_collectives
```

The local Open MPI run used `--oversubscribe` because it launched four small ranks on one host. That launcher option is implementation-specific and is not part of the MPI language API. The collective example requires exactly four ranks. The ring example supports any positive rank count supplied by the launcher.

The build products belong outside the repository. No executable binaries are committed. The local evidence file records compilation success, execution return codes, output, and machine scope. A later run on another system should save a new record rather than overwrite the meaning of the existing evidence.

## You can rerun all native checks with one command.

```bash
python labs/parallel/check_native.py --output evidence/parallel/native-new-run.json
```

The helper records compiler and MPI launcher versions, source-file SHA-256 hashes, commands, return codes, and outputs. It creates all binaries in a temporary directory and removes that directory afterward. You can set `CC`, `MPICC`, and `MPIEXEC` when your toolchain uses different executable names. A missing compiler or runtime is an environment failure, not a successful verification.
