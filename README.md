# Numerical and Parallel Computing

This study course brings together UMN CSCI 5302 and CSCI 5451 through original explanations, worked examples, exercises, and runnable experiments. The 48 lessons connect numerical accuracy with CPU, distributed, and GPU algorithm design.

Arun Sharma requested this edition. Codex drafted the material and recorded the checks. The repository includes a concrete handoff for Claude to review and refine it.

## You can start with the website or the complete document.

- You can read the [course website](https://arunshar.github.io/numerical-parallel-computing/).
- You can download the [complete interactive HTML study guide](dist/STUDY_GUIDE.html) and open it locally. It includes offline mathematics, collapsible navigation, keyboard navigation, search, and glossary definitions.
- You can read the [complete Markdown study guide](dist/STUDY_GUIDE.md) in GitHub or another editor.
- You can inspect the [coverage map](docs/reference/coverage.md), [source list](docs/reference/sources.md), and [validation record](docs/reference/validation.md).
- Claude should begin with the [continuation handoff](CLAUDE_HANDOFF.md).

## The course has three connected tracks.

| Lessons | The track teaches these topics. |
|---|---|
| 0 | The preparation lesson explains prerequisites, notation, execution, and evidence. |
| 1–16 | The numerical track covers floating-point errors, linear systems, least squares, eigenvalues, nonlinear equations, optimization, interpolation, quadrature, ODEs, boundary problems, PDEs, Fourier methods, simulation, and iterative solvers. |
| 17–30 | The CPU and distributed track covers architecture, decomposition, pthreads, OpenMP, communication, MPI, scaling models, dense matrices, sorting, graphs, search, dynamic programming, parallel FFTs, hybrid programming, and quantum basics. |
| 31–46 | The GPU track covers CUDA execution, scheduling, memory, performance, convolution, stencils, histograms, reductions, scans, sorting, sparse graphs, deep learning, scientific applications, clusters, and advanced CUDA topics. |
| 47 | The capstone connects a manufactured diffusion problem with decomposition and a declared communication-cost model. |

The coverage ledger maps 67 textbook chapters and appendices from Heath, Ascher and Greif, Grama and colleagues, and Hwu, Kirk, and El Hajj. Public course materials establish the class-specific topics, including the CSCI 5302 supplements, Mooney's five homework topics, and Saad's older lecture series.

The complete textbooks were not available for this build. The chapter map uses public author and publisher outlines, public lecture materials, and primary documentation. Some advanced subjects receive conceptual surveys. The ledger and handoff identify those areas for expansion. This edition does not claim to reproduce every proof, exercise, or subsection in the books.

## You can run the portable experiments locally.

```bash
git clone https://github.com/arunshar/numerical-parallel-computing.git
cd numerical-parallel-computing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-build.txt
python scripts/verify_course.py
python labs/build_notebooks.py 47
```

The 46 standalone Python labs use small teaching inputs and assertions. Their outputs are recorded in `evidence/`. The executed capstone notebook contains its outputs and interactive quiz. Every lesson includes at least five review questions.

The native pthread, OpenMP, and MPI examples have separate local correctness records. The CUDA examples require an NVIDIA environment and remain UNVERIFIED for native compilation and execution. CPU models and invented cost parameters establish no GPU or cluster performance measurements.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arunshar/numerical-parallel-computing/blob/main/docs/lectures/47-integrated-capstone.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arunshar/numerical-parallel-computing/main?labpath=docs/lectures/47-integrated-capstone.ipynb)

The browser buttons open the capstone notebook. The other lessons provide standalone Python scripts and local execution instructions.

## You can rebuild the reading editions and the website.

```bash
python scripts/build_inventory.py
python scripts/build_book.py
python scripts/build_reader.py
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The HTML reading edition requires Pandoc when it is rebuilt. The finished HTML file has no external JavaScript or font dependency. The Sphinx website uses MathJax for mathematical typesetting.

GitHub Actions verifies the portable labs, builds the website with warnings treated as errors, checks that every page fits a 375 px phone screen, and deploys GitHub Pages. You can run the phone-screen check locally with `python scripts/check_mobile.py docs/_build/html --include-static`. It needs Chrome. The repository also includes a Read the Docs configuration. That service needs a separate account import before it hosts a live copy.

## The repository keeps source ownership clear.

The public repository contains original notes, original teaching code, and a source inventory with links and hashes. It excludes third-party textbook files, lecture PDFs, and homework bundles. `scripts/fetch_public_sources.py` can fetch linked public resources into an ignored local directory for personal study when their source servers permit access.

The [MIT license](LICENSE) applies to this repository's original code and notes. The [third-party notice](THIRD_PARTY_NOTICES.md) explains the source boundaries. It grants no rights to external course or textbook materials.
