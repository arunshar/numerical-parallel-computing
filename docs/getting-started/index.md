# How this course works

This course connects two questions. Numerical analysis asks whether a computation gives a meaningful answer. Parallel computing asks how several processors can produce that answer efficiently. You need both questions when a faster implementation changes rounding order, communication cost, or memory use.

The numerical track supports CSCI 5302. The CPU, distributed, and GPU tracks support CSCI 5451. The chapters also cover major topics from Heath, Ascher and Greif, Grama and colleagues, and Hwu and colleagues. The {doc}`../reference/coverage` page separates course topics, textbook extensions, and survey material.

## You can choose a reading path.

| Your goal | Your starting path |
|---|---|
| You want to prepare for CSCI 5302. | You should study lessons 1–16 and work through each numerical experiment. |
| You want to prepare for CSCI 5451. | You should study lessons 17–46 and revisit numerical lessons 1–4 as needed. |
| You want to connect numerical methods with hardware. | You should pair linear systems with distributed matrix multiplication, PDEs with stencils, and summation error with reductions. |
| You want to refine the material with Claude. | You should start with the repository's CLAUDE_HANDOFF.md and the coverage ledger. |

## Each lesson gives you an experiment.

A lesson starts with the problem and defines the terms it needs. A worked example uses small numbers so you can check the reasoning by hand. A lab then tests one claim using an independent implementation. The exercises change that setup, and the quiz tests the important distinctions.

The Python experiments run on a CPU. A model of a network or GPU is explicitly a model. Its output does not measure a real cluster or GPU. The native C, MPI, and CUDA examples have separate requirements and validation records.

## You should read the output before interpreting it.

You should identify the input, algorithm, expected result, and comparison rule before running a program. A printed speedup is meaningful only when the implementations solve the same problem with comparable accuracy. A residual is meaningful only after you connect it to the scale and conditioning of the problem.

The example data are synthetic teaching inputs. The experiments do not reproduce the instructors' homework results or any research benchmark. Timing varies across machines, so deterministic operation counts and numerical invariants carry most of the lessons.

## You can run locally or in a notebook environment.

| Environment | What it supports | What you must check |
|---|---|---|
| A local Python environment supports the NumPy experiments. | You can run the original lab scripts and rebuild the documentation. | You must install the declared dependencies. |
| Google Colab supports portable Python notebooks. | You can run small CPU experiments without a local Python installation. | An assigned runtime does not guarantee a particular GPU. |
| A local MPI installation supports the native distributed examples. | You can test small rank counts on one machine. | One-machine MPI does not measure a cluster network. |
| A CUDA machine supports the native GPU examples. | You can compile kernels and measure device behavior. | You must record the GPU, compiler, flags, inputs, and numerical checks. |

The small CPU experiments are designed to finish quickly and use less than one gigabyte of memory. You should inspect a script before increasing a problem size.
