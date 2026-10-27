# Claude continuation handoff

Arun requested a detailed study document for CSCI 5302 and CSCI 5451, with simple explanations, worked examples, tutorials, and coverage of the associated textbook topics. He approved the public repository name `arunshar/numerical-parallel-computing`.

## You should start with the sources and evidence.

Read `README.md`, `docs/reference/coverage.md`, `docs/reference/sources.md`, and `docs/reference/validation.md`. Then inspect `coverage/book-chapters.json`, the track-specific coverage JSON files, and `evidence/validation.json`. These files distinguish original explanations, selected experiments, compressed surveys, and hardware validation.

The previous local archive contained the instructors' original PDFs and homework bundles. Those third-party files are excluded from the public repository. `coverage/public-source-inventory.json` retains their source URLs, retrieval times, and hashes. You can obtain the files from the original sources when access permits.

The complete textbooks were not available during this build. The author and publisher outlines establish chapter scope, and public lecture slides establish course topics. Do not claim that every textbook proof, exercise, or subsection has been checked against the full edition.

## The authored sources have a stable layout.

Lessons 1–16 cover numerical methods. Lessons 17–30 cover CPU and distributed computing. Lessons 31–46 cover GPU computing. Lesson 0 prepares the reader, and lesson 47 connects the tracks with a manufactured diffusion problem and a communication model.

The Markdown lectures live in `docs/lectures/`. The capstone notebook is generated from `labs/src/47-integrated-capstone.py`. The portable teaching scripts live in `labs/numerical/`, `labs/parallel/`, and `labs/gpu/`. The native C and CUDA examples are stored within their relevant tracks.

`dist/STUDY_GUIDE.md` is the combined reading edition. It is generated from the lesson sources. You should edit the sources and rebuild the reading edition instead of editing the generated book alone.

## You should preserve the evidence boundaries.

The Python experiments use small synthetic teaching inputs. Their successful execution validates their stated assertions on those inputs. It does not validate all possible inputs, numerical stability in every regime, a distributed cluster, or CUDA hardware performance.

Native pthread, OpenMP, and MPI checks have their own evidence when the required tools were available. Native CUDA execution is UNVERIFIED unless a later record names the hardware, compiler, command, input, and result. A CPU model of a GPU algorithm is not a GPU benchmark.

The study material includes a useful algebraic identity for the pairwise batched matrix-product example. Real-arithmetic equivalence does not guarantee bitwise floating-point equivalence or permission under an instructor's assignment constraints. Keep those distinctions explicit.

## The next refinement pass should deepen the compressed areas.

1. You should compare each lesson with the actual textbook edition when Arun provides authorized access. Expand the chapter ledger into a subsection ledger before calling the coverage exhaustive.
2. You should strengthen the advanced numerical surveys with complete derivations and independent examples. Priority topics include specialized eigensolvers, best approximation, finite elements, spectral discretizations, and integral equations.
3. You should extend the distributed examples to multiple machines only when Arun authorizes that runtime. Record the network and process placement before interpreting scaling.
4. You should compile and test the CUDA examples on a compatible NVIDIA device. Add sanitizer checks, boundary cases, and numerical comparisons before collecting performance data.
5. You should deepen the advanced GPU case studies, including MRI reconstruction, electrostatic cutoff binning, dynamic parallelism, and cluster overlap.
6. You should read the prose as a student. Replace unexplained jargon with definitions and small examples while preserving equations and claims.
7. You should check every quiz answer against its lesson and executable evidence. A plausible-looking distractor is not a substitute for a checked explanation.

## You should run the repository checks before proposing changes.

```bash
python -m pip install -r requirements-build.txt
python scripts/verify_course.py
python labs/build_notebooks.py 47
python scripts/build_inventory.py
python scripts/build_book.py
python scripts/build_reader.py
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The notebook build uses a local helper inside the repository. A browser notebook can download the helper from the public GitHub source. The helper URL and repository settings must stay aligned if the repository is renamed.

You should work on a branch and open a draft pull request after publication. Arun should review substantive changes before they merge. The public source files contain no credentials and should remain that way.

## The publishing states are separate.

The GitHub repository stores the source and generated reading editions. A GitHub Actions workflow verifies the portable labs, builds the Sphinx site with warnings treated as errors, and deploys GitHub Pages. A successful remote workflow and live page checks establish website publication.

The repository also contains a Read the Docs configuration. A local preflight can validate that configuration, but a live Read the Docs project still requires Arun to import the repository through his account. Do not describe that service as live until its build and URL are verified.
