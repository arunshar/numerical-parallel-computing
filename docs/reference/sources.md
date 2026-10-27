# Sources and attribution

The notes use public course material to identify scope and primary references to check mechanisms. The explanations, worked examples, quizzes, and teaching programs in this repository are original study material. The repository links to instructors' files instead of republishing their PDF collections or homework archives.

## CSCI 5302 provides the numerical course sequence.

[Prof. Daniel Boley's Spring 2025 course](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/) provides the syllabus, weekly sequence, and links to [notes and supplementary material](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The supplementary topics include multivariate Taylor expansion, bicubic interpolation, QP/LP examples, FFTs, SVMs, and MATLAB references.

Michael T. Heath wrote *Scientific Computing: An Introductory Survey*. The course names the second edition, and its SIAM link points to the revised second edition. The [author's table of contents](https://heath.cs.illinois.edu/scicomp/contents/index.html) and [lecture slides](https://heath.cs.illinois.edu/scicomp/notes/) provide the public topic outline. The [SIAM book page](https://epubs.siam.org/doi/book/10.1137/1.9781611975581) identifies the revised edition.

Uri M. Ascher and Chen Greif wrote *A First Course in Numerical Methods* (SIAM, 2011). The [publisher's chapter list](https://epubs.siam.org/doi/book/10.1137/9780898719987) supplies the 16-chapter outline. The full textbook was not downloaded for this study edition.

## CSCI 5451 provides the parallel course sequences.

[James Mooney's Fall 2025 course](https://jimtmooney.github.io/Courses/F25/index.html) provides the dated lecture sequence, 26 PDF decks, five homework packages, and team-project expectations. [Prof. Yousef Saad's Spring 2021 notes](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) add a second sequence with sorting, sparse matrices, graphs, hybrid programming, and introductory quantum computing.

Ananth Grama, Anshul Gupta, George Karypis, and Vipin Kumar wrote *Introduction to Parallel Computing*, second edition (Addison-Wesley, 2003). The [author-hosted contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) provide a preprint chapter outline. That page reverses the final edition's chapters 6 and 7. The author's [chapter 6 slides](https://www.cs.purdue.edu/homes/ayg/book/Slides/chap6_slides.pdf) identify message passing, which agrees with the course's reading references. This edition uses chapter 6 for MPI and chapter 7 for shared-memory programming. This repository does not redistribute the authors' restricted preprint chapters.

Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj wrote *Programming Massively Parallel Processors: A Hands-on Approach*, fourth edition (Morgan Kaufmann, 2022). The [publisher's page](https://www.educate.elsevier.com/book/details/9780323912310) lists 23 chapters and Appendix A. The author list includes El Hajj, whose name was absent from the initial course summary supplied in chat.

[Simon Boehm's CUDA matrix-multiplication worklog](https://siboehm.com/articles/22/CUDA-MMM) is an additional performance case study linked by the course. The new lessons explain the relevant optimization ideas using their own examples. They do not reproduce his article or its figures.

## Primary technical references support the implementation details.

The [NumPy documentation](https://numpy.org/doc/stable/) describes the array and linear algebra functions used in the CPU examples. The [SciPy documentation](https://docs.scipy.org/doc/scipy/) provides additional numerical method references.

The [MPI Forum](https://www.mpi-forum.org/docs/) publishes the MPI standard. [OpenMP](https://www.openmp.org/specifications/) publishes the OpenMP specification. [NVIDIA's CUDA documentation](https://docs.nvidia.com/cuda/) covers execution, memory behavior, synchronization, and native development tools. Individual lessons cite narrower pages where useful.

## Public availability does not transfer copyright.

The MIT license applies to the original code and prose contributed to this repository. Linked books, course slides, assignment files, and articles retain their own copyright and terms. The local archives created before this repository are separate study downloads. They are not included in the public Git history.

## You can fetch the public source files locally.

The following command downloads cataloged source PDFs into `downloaded-sources/`, which Git ignores. The source sites can change or withdraw a file. The script records access failures and compares downloaded bytes with the cataloged hashes.

```bash
python scripts/fetch_public_sources.py --course all --pdf-only
```

You can omit `--pdf-only` to include the other cataloged course files. The script does not download full textbooks, sign into Canvas, or execute downloaded programs. You should keep the downloaded directory outside public commits.
