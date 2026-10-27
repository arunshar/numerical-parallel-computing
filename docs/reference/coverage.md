# Course and textbook coverage

This ledger maps the public course topics and all chapter outlines from the four associated books to the study lessons. A mapped chapter means that the notes explain its subject and include selected examples. It does not mean that every proof, exercise, historical note, or software variant in the full text has been reproduced.

The full textbooks were not available during the initial build. Public slides, author and publisher chapter outlines, and primary API documentation supplied the verifiable scope. The track-specific JSON ledgers in `coverage/` distinguish explanations from compressed surveys.

## The textbook map covers the complete chapter outlines.

| Book | Chapter | Topic | Lessons |
|---|---|---|---|
| Heath | 1 | Scientific Computing | {doc}`1 <../lectures/01-errors-floating-point>` |
| Heath | 2 | Systems of Linear Equations | {doc}`2 <../lectures/02-linear-systems>`, {doc}`15 <../lectures/15-iterative-solvers>` |
| Heath | 3 | Linear Least Squares | {doc}`3 <../lectures/03-least-squares-svd>` |
| Heath | 4 | Eigenvalue Problems | {doc}`4 <../lectures/04-eigenvalues>` |
| Heath | 5 | Nonlinear Equations | {doc}`5 <../lectures/05-nonlinear-equations>` |
| Heath | 6 | Optimization | {doc}`6 <../lectures/06-unconstrained-optimization>`, {doc}`7 <../lectures/07-constrained-optimization-svm>` |
| Heath | 7 | Interpolation | {doc}`8 <../lectures/08-interpolation-splines>` |
| Heath | 8 | Numerical Integration and Differentiation | {doc}`9 <../lectures/09-quadrature-differentiation>` |
| Heath | 9 | Initial Value Problems for Ordinary Differential Equations | {doc}`10 <../lectures/10-ode-initial-value>` |
| Heath | 10 | Boundary Value Problems for Ordinary Differential Equations | {doc}`11 <../lectures/11-boundary-value>` |
| Heath | 11 | Partial Differential Equations | {doc}`12 <../lectures/12-pde>` |
| Heath | 12 | Fast Fourier Transform | {doc}`13 <../lectures/13-fourier-fft>` |
| Heath | 13 | Random Numbers and Simulation | {doc}`14 <../lectures/14-random-simulation>` |
| Ascher and Greif | 1 | Numerical Algorithms | {doc}`1 <../lectures/01-errors-floating-point>`, {doc}`16 <../lectures/16-numerical-workflow>` |
| Ascher and Greif | 2 | Roundoff Errors | {doc}`1 <../lectures/01-errors-floating-point>` |
| Ascher and Greif | 3 | Nonlinear Equations in One Variable | {doc}`5 <../lectures/05-nonlinear-equations>` |
| Ascher and Greif | 4 | Linear Algebra Background | {doc}`2 <../lectures/02-linear-systems>`, {doc}`3 <../lectures/03-least-squares-svd>`, {doc}`4 <../lectures/04-eigenvalues>` |
| Ascher and Greif | 5 | Linear Systems: Direct Methods | {doc}`2 <../lectures/02-linear-systems>` |
| Ascher and Greif | 6 | Least Squares | {doc}`3 <../lectures/03-least-squares-svd>` |
| Ascher and Greif | 7 | Linear Systems: Iterative Methods | {doc}`15 <../lectures/15-iterative-solvers>` |
| Ascher and Greif | 8 | Eigenvalues and Singular Values | {doc}`4 <../lectures/04-eigenvalues>` |
| Ascher and Greif | 9 | Nonlinear Systems and Optimization | {doc}`5 <../lectures/05-nonlinear-equations>`, {doc}`6 <../lectures/06-unconstrained-optimization>`, {doc}`7 <../lectures/07-constrained-optimization-svm>` |
| Ascher and Greif | 10 | Polynomial Interpolation | {doc}`8 <../lectures/08-interpolation-splines>` |
| Ascher and Greif | 11 | Piecewise Polynomial Interpolation | {doc}`8 <../lectures/08-interpolation-splines>` |
| Ascher and Greif | 12 | Best Approximation | {doc}`8 <../lectures/08-interpolation-splines>`, {doc}`16 <../lectures/16-numerical-workflow>` |
| Ascher and Greif | 13 | Fourier Transform | {doc}`13 <../lectures/13-fourier-fft>` |
| Ascher and Greif | 14 | Numerical Differentiation | {doc}`9 <../lectures/09-quadrature-differentiation>` |
| Ascher and Greif | 15 | Numerical Integration | {doc}`9 <../lectures/09-quadrature-differentiation>` |
| Ascher and Greif | 16 | Differential Equations | {doc}`10 <../lectures/10-ode-initial-value>`, {doc}`11 <../lectures/11-boundary-value>`, {doc}`12 <../lectures/12-pde>` |
| Grama and colleagues | 1 | Introduction | {doc}`17 <../lectures/17-architecture-cache>` |
| Grama and colleagues | 2 | Parallel Programming Platforms | {doc}`17 <../lectures/17-architecture-cache>` |
| Grama and colleagues | 3 | Principles of Parallel Algorithm Design | {doc}`18 <../lectures/18-decomposition-task-graphs>` |
| Grama and colleagues | 4 | Basic Communication Operations | {doc}`20 <../lectures/20-communication-networks>`, {doc}`22 <../lectures/22-mpi-collectives-topology>` |
| Grama and colleagues | 5 | Analytical Modeling of Parallel Programs | {doc}`23 <../lectures/23-performance-models>` |
| Grama and colleagues | 6 | Programming Using the Message-Passing Paradigm | {doc}`21 <../lectures/21-mpi-point-to-point>`, {doc}`22 <../lectures/22-mpi-collectives-topology>` |
| Grama and colleagues | 7 | Programming Shared Address Space Platforms | {doc}`19 <../lectures/19-pthreads-openmp>` |
| Grama and colleagues | 8 | Dense Matrix Algorithms | {doc}`24 <../lectures/24-dense-matrix-cannon>` |
| Grama and colleagues | 9 | Sorting | {doc}`25 <../lectures/25-sorting-merging>` |
| Grama and colleagues | 10 | Graph Algorithms | {doc}`26 <../lectures/26-graphs-sparse-partitioning>` |
| Grama and colleagues | 11 | Search Algorithms for Discrete Optimization Problems | {doc}`27 <../lectures/27-search-dynamic-programming>` |
| Grama and colleagues | 12 | Dynamic Programming | {doc}`27 <../lectures/27-search-dynamic-programming>` |
| Grama and colleagues | 13 | Fast Fourier Transform | {doc}`28 <../lectures/28-parallel-fft>` |
| Grama and colleagues | A | Appendix A: Complexity of Functions and Order Analysis | {doc}`18 <../lectures/18-decomposition-task-graphs>`, {doc}`23 <../lectures/23-performance-models>` |
| Hwu, Kirk, and El Hajj | 1 | Introduction | {doc}`31 <../lectures/31-cuda-execution>` |
| Hwu, Kirk, and El Hajj | 2 | Heterogeneous data parallel computing | {doc}`31 <../lectures/31-cuda-execution>` |
| Hwu, Kirk, and El Hajj | 3 | Multidimensional grids and data | {doc}`31 <../lectures/31-cuda-execution>` |
| Hwu, Kirk, and El Hajj | 4 | Compute architecture and scheduling | {doc}`32 <../lectures/32-scheduling-occupancy>` |
| Hwu, Kirk, and El Hajj | 5 | Memory architecture and data locality | {doc}`33 <../lectures/33-memory-tiling>` |
| Hwu, Kirk, and El Hajj | 6 | Performance considerations | {doc}`34 <../lectures/34-gpu-performance>` |
| Hwu, Kirk, and El Hajj | 7 | Convolution | {doc}`35 <../lectures/35-convolution>` |
| Hwu, Kirk, and El Hajj | 8 | Stencil | {doc}`36 <../lectures/36-stencils>` |
| Hwu, Kirk, and El Hajj | 9 | Parallel histogram | {doc}`37 <../lectures/37-histograms-atomics>` |
| Hwu, Kirk, and El Hajj | 10 | Reduction | {doc}`38 <../lectures/38-reductions>` |
| Hwu, Kirk, and El Hajj | 11 | Prefix sum (scan) | {doc}`39 <../lectures/39-prefix-scan>` |
| Hwu, Kirk, and El Hajj | 12 | Merge | {doc}`40 <../lectures/40-merge-sort>` |
| Hwu, Kirk, and El Hajj | 13 | Sorting | {doc}`40 <../lectures/40-merge-sort>` |
| Hwu, Kirk, and El Hajj | 14 | Sparse matrix computation | {doc}`41 <../lectures/41-sparse-graphs>` |
| Hwu, Kirk, and El Hajj | 15 | Graph traversal | {doc}`41 <../lectures/41-sparse-graphs>` |
| Hwu, Kirk, and El Hajj | 16 | Deep learning | {doc}`42 <../lectures/42-deep-learning>` |
| Hwu, Kirk, and El Hajj | 17 | Iterative magnetic resonance imaging reconstruction | {doc}`43 <../lectures/43-mri-electrostatics>` |
| Hwu, Kirk, and El Hajj | 18 | Electrostatic potential map | {doc}`43 <../lectures/43-mri-electrostatics>` |
| Hwu, Kirk, and El Hajj | 19 | Parallel programming and computational thinking | {doc}`44 <../lectures/44-streams-clusters>`, {doc}`46 <../lectures/46-gpu-numerics-project>` |
| Hwu, Kirk, and El Hajj | 20 | Programming a heterogeneous computing cluster | {doc}`44 <../lectures/44-streams-clusters>` |
| Hwu, Kirk, and El Hajj | 21 | CUDA dynamic parallelism | {doc}`45 <../lectures/45-dynamic-parallelism-future>` |
| Hwu, Kirk, and El Hajj | 22 | Advanced practices and future evolution | {doc}`45 <../lectures/45-dynamic-parallelism-future>` |
| Hwu, Kirk, and El Hajj | 23 | Conclusion and outlook | {doc}`45 <../lectures/45-dynamic-parallelism-future>` |
| Hwu, Kirk, and El Hajj | A | Appendix A: Numerical considerations | {doc}`46 <../lectures/46-gpu-numerics-project>`, {doc}`1 <../lectures/01-errors-floating-point>` |

## The course-specific material has its own map.

| Source unit | Study location | Coverage boundary |
|---|---|---|
| The CSCI 5302 syllabus covers errors, equations, interpolation, least squares, optimization, quadrature, and ODEs. | Lessons 1–10 explain the core sequence. | Lessons 11–16 extend the sequence to the remaining book topics. |
| The supplementary Taylor and bicubic notes explain local multivariate models and two-dimensional interpolation. | Lessons 8 and 16 explain those constructions. | The examples use new teaching data. |
| The supplementary QP/LP and SVM materials connect optimization to constrained models. | Lesson 7 explains the objectives, constraints, and optimality conditions. | The notes do not reproduce the instructor's handouts. |
| The MATLAB references compare array operations and programming conventions. | Lesson 16 explains the important MATLAB/Python differences. | The original reference sheets remain external sources. |
| Mooney's 26 lecture decks cover architectures, OpenMP, MPI, modeling, and CUDA. | Lessons 17–46 explain that sequence and its textbook extensions. | The final-lecture hardware and ML topics are clearly marked surveys. |
| HW1 uses shared-memory parallelism and dynamic time warping. | Lessons 19, 27, and 30 explain dependencies, implementation checks, and reporting. | The examples teach the methods without redistributing the assignment package. |
| HW2 and HW3 use distributed matrix multiplication and block-cyclic layouts. | Lessons 21, 22, 24, and 30 explain the ownership and communication. | The new examples have their own data and assertions. |
| HW4 uses CUDA convolution and profiling. | Lessons 34 and 35 explain memory reuse, boundaries, and performance evidence. | Native CUDA performance remains unverified locally. |
| HW5 uses pairwise batched matrix products. | Lesson 46 explains the algebra, numerical consequences, and validation choices. | Algebraic simplification does not establish permission under a particular assignment rubric. |
| Saad's 19 lecture sets add sorting, sparse and graph algorithms, hybrid programming, and quantum concepts. | Lessons 17–29 and 40–41 cover these subjects. | The quantum material is an introductory model. |
| The final project requires an implementation and evidence. | Lessons 30, 46, and 47 explain reproducibility, correctness, and performance reporting. | The capstone is an original teaching problem. |

## Some advanced subjects remain survey-level treatments.

A short section introduces a method's purpose, assumptions, main mechanism, and tradeoffs. It does not establish mastery of that method. The compressed areas include specialized eigensolver variants, finite-element and spectral implementation details, quantum algorithms, MRI reconstruction, dynamic parallelism, and evolving accelerator platforms. The handoff identifies these areas for expansion with authorized textbook access and hardware evidence.

The intended next step is to deepen those sections, not to change their status labels without adding explanations and checks. A future edition should also map textbook subsections and exercises when the actual editions become available.
