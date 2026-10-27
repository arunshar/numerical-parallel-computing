# Run the course locally

You need Python 3.12 or newer and Git. The basic experiments use NumPy, and the build environment includes the scientific Python packages used by the examples.

## 1. Get the repository.

```bash
git clone https://github.com/arunshar/numerical-parallel-computing.git
cd numerical-parallel-computing
```

## 2. Create an environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-build.txt
```

## 3. Run a lesson.

Each lesson gives the command for its lab. You should run those commands from the repository root. You can also run the complete portable verification suite.

```bash
python scripts/verify_course.py
```

The suite executes the Python teaching examples and checks the lecture structure. It does not compile every native program or claim CUDA execution. The machine-readable result appears in `evidence/validation.json`.

## 4. Build the site and combined document.

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
python scripts/build_book.py
python scripts/build_reader.py
```

You can open `docs/_build/html/index.html` in a browser. The combined Markdown document appears in `dist/STUDY_GUIDE.md`. The repository also provides a generated HTML edition when the release includes one.

## You should keep generated results separate from sources.

| Path | Contents | Whether you can regenerate it |
|---|---|---|
| `docs/lectures/` contains the authored lesson pages. | The lesson prose and quiz definitions live here. | You should preserve edits to these sources. |
| `labs/` contains the original teaching programs. | Each track has its own directory. | You should preserve edits to these sources. |
| `evidence/` contains validation records. | The records describe actual runs and their machine scope. | You can regenerate them by running the stated commands. |
| `docs/_build/` contains the rendered website. | Sphinx creates these files. | You can rebuild this directory. |
| `dist/` contains the combined reading editions. | The build script collects the authored lessons. | You can rebuild these documents. |

## Common failures have specific causes.

`ModuleNotFoundError` appears when the selected Python lacks a dependency.
: You should activate `.venv` and install `requirements-build.txt` with that environment's Python.

`mpicc` is unavailable when MPI is not installed.
: You can still run the CPU models. You need a separate MPI installation to compile the native examples.

`nvcc` is unavailable when the CUDA toolkit is not installed.
: You can run the CPU reference examples. Native CUDA execution requires compatible NVIDIA hardware and a toolkit.

A speed measurement changes across runs.
: You should repeat measurements, warm up the workload, record uncertainty, and compare correctness before interpreting performance.
