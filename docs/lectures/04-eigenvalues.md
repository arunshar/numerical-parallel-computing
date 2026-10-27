# 4. Eigenvalues, invariant subspaces, and matrix transformations

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`03-least-squares-svd` described how a matrix stretches input directions. Eigenvalues ask which directions remain on their own line after multiplication. These directions explain repeated matrix application, vibration modes, stability, and many iterative algorithms. We will compute a dominant eigenpair and use its residual to assess progress.

:::{admonition} Learning goals
- You will interpret eigenpairs and invariant subspaces.
- You will explain power, inverse, Rayleigh quotient, and QR iteration.
- You will distinguish symmetric problems from general nonsymmetric problems.
- You will use residuals without confusing them with guaranteed eigenvector accuracy.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## An eigenvector keeps its direction

An **eigenpair** consists of a nonzero vector v and a scalar lambda satisfying A v equals lambda v. Multiplying by A changes the vector's scale but keeps it in the same one-dimensional subspace. A negative real eigenvalue reverses its orientation, while a complex eigenvalue can describe rotation and scaling in a real two-dimensional representation.

$$
Av=\lambda v.
$$

For the matrix with rows two, one and one, two, the vectors one, one and one, minus one are eigenvectors. Their eigenvalues are three and one. Repeated multiplication magnifies the first direction three times per application and leaves the second direction unchanged. A generic starting vector therefore becomes increasingly aligned with the first direction.

The characteristic polynomial provides a formal definition through the determinant of A minus lambda I. Computing that polynomial and then finding its roots is usually a poor numerical strategy. Its coefficients can be extremely sensitive, and it discards useful matrix structure.

## Multiplicity does not guarantee enough eigenvectors

A repeated eigenvalue has an algebraic multiplicity in the characteristic polynomial. Its geometric multiplicity is the dimension of its eigenspace. A matrix is **defective** when it lacks a full basis of eigenvectors. The two-by-two matrix with ones on the diagonal and one above the diagonal has only one independent eigenvector.

A defective or nearly defective matrix can have very sensitive eigenvectors and eigenvalues. A small matrix perturbation can change the spectrum much more than a comparable perturbation would change a symmetric matrix. The condition of the eigenvector basis matters for a diagonalizable nonsymmetric matrix.

An **invariant subspace** is a collection of directions that multiplication keeps within that collection. A cluster of nearby eigenvalues can define a stable subspace even when its individual eigenvectors are difficult to identify. Many practical algorithms target the subspace rather than insist on a particular basis inside it.

## Symmetry gives strong guarantees

A real symmetric matrix has real eigenvalues and an orthonormal eigenvector basis. Orthogonal changes of coordinates preserve Euclidean distances, which makes symmetric eigenproblems numerically attractive. A general real matrix can instead require complex eigenpairs or a real block representation.

The **Schur decomposition** writes a square matrix as Q T Q conjugate-transpose with unitary Q and upper triangular T in the complex case. The diagonal of T gives the eigenvalues. Real arithmetic can use two-by-two diagonal blocks for conjugate pairs. Schur form exists even when an eigenvector basis does not.

Gershgorin disks provide inexpensive eigenvalue location bounds. Each disk is centered at a diagonal entry, with radius equal to the sum of absolute off-diagonal entries in that row. The bounds can be loose, but they connect diagonal dominance to spectral location without a complete decomposition.

## Power iteration follows the strongest direction

**Power iteration** repeatedly multiplies a vector by A and normalizes it. If one eigenvalue has strictly largest magnitude and the starting vector has a component in its direction, that component can dominate. For a diagonalizable matrix, the asymptotic direction error depends on the magnitude ratio between the second and first eigenvalues.

The Rayleigh quotient estimates the eigenvalue associated with the current vector. For a unit vector, it is x transpose A x. The eigenpair residual is A x minus the quotient times x.

$$
\rho(x)=\frac{x^T A x}{x^T x},\qquad r=Ax-\rho(x)x.
$$

The lab's eigenvalues are three and one, so the direction error contracts at roughly one-third per iteration after the relevant assumptions hold. The quotient converges faster for this symmetric problem because its first-order variation vanishes near an eigenvector. The printed residual gives a direct check that the eigenpair equation is becoming more accurate.

:::{admonition} Try it
:class: exercise
You can change the diagonal entries to make the two eigenvalues closer in magnitude. Predict how the iteration count changes. Then choose the starting vector exactly perpendicular to the dominant eigenvector and explain why power iteration misses that eigenpair.
:::

## Inverse iteration targets a chosen region

**Inverse iteration** applies the inverse of A minus a shift times I to the current vector. The method solves a linear system at each step rather than forming an inverse. Eigenvalues close to the shift become dominant after inversion. A fixed shift lets the algorithm reuse one factorization.

**Rayleigh quotient iteration** updates the shift from the current quotient. It can converge very quickly near a simple eigenpair of a symmetric matrix. The shifted solve becomes nearly singular near convergence, so implementation requires care. Rapid local convergence does not imply reliable convergence from every starting vector.

Deflation removes or separates a converged component so the algorithm can find another one. Simultaneous iteration advances a block of vectors and repeatedly orthogonalizes them. A block method handles an invariant subspace more naturally than a single-vector method when several eigenvalues are clustered.

## QR iteration transforms the whole matrix

An unshifted QR step factors the current matrix into Q R and forms R Q. The new matrix is orthogonally similar to the old one, so the eigenvalues remain unchanged. Shifts accelerate convergence. Practical dense solvers first reduce a general matrix to Hessenberg form or a symmetric matrix to tridiagonal form.

That reduction makes later iterations far cheaper than repeatedly factoring an arbitrary dense matrix. A nearly zero subdiagonal entry allows the problem to split, which is also called deflation. Symmetric tridiagonal problems support specialized methods such as bisection, divide-and-conquer, and relatively robust representations. This lecture names their purpose without reproducing their full implementation machinery.

## Large problems need selected spectral information

A **Krylov subspace** contains a starting vector and its repeated matrix products. Arnoldi iteration builds an orthonormal basis for a general matrix. Lanczos iteration exploits symmetry to obtain a shorter recurrence. These methods project a large problem into a smaller one whose eigenpairs approximate selected eigenpairs of the original.

Loss of orthogonality can create duplicate or misleading approximations, so robust implementations monitor it. The matrix can be represented by a product routine rather than an explicit dense array. Generalized problems, A v equals lambda B v, add another matrix and require attention to whether B is positive definite, singular, or indefinite.

:::{admonition} Try it
:class: exercise
You can calculate residuals for every reference eigenpair returned by the symmetric solver. Then perturb the defective matrix's lower-left entry by a tiny positive number. Compare the eigenvalue change with the perturbation size and explain why this is a sensitivity experiment rather than a timing benchmark.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/04_eigenvalues.py
```

```python
"""Compare power iteration and a symmetric eigensolver."""
import numpy as np

A = np.array([[2., 1.], [1., 2.]])
x = np.array([1., 0.])
for k in range(1, 9):
    x = A@x
    x /= np.linalg.norm(x)
    rayleigh = x@A@x
    residual = np.linalg.norm(A@x-rayleigh*x)
    print(f"iteration {k}: lambda={rayleigh:.9f}, residual={residual:.6e}")
values, vectors = np.linalg.eigh(A)
print("reference eigenvalues:", values)
assert np.allclose(values, [1., 3.])
B = np.array([[1., 1.], [0., 1.]])
print("defective-matrix eigenvalues:", np.linalg.eigvals(B))
print("rank of B-I:", np.linalg.matrix_rank(B-np.eye(2)))
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
iteration 1: lambda=2.800000000, residual=6.000000e-01
iteration 2: lambda=2.975609756, residual=2.195122e-01
iteration 3: lambda=2.997260274, residual=7.397260e-02
iteration 4: lambda=2.999695215, residual=2.468760e-02
iteration 5: lambda=2.999966130, residual=8.230313e-03
iteration 6: lambda=2.999996237, residual=2.743479e-03
iteration 7: lambda=2.999999582, residual=9.144946e-04
iteration 8: lambda=2.999999954, residual=3.048316e-04
reference eigenvalues: [1. 3.]
defective-matrix eigenvalues: [1. 1.]
rank of B-I: 1
```

The Rayleigh estimate approaches the dominant eigenvalue three, and the eigenpair residual falls from 0.6 to about 3.05e-4 across eight iterations. The defective example has two repeated eigenvalues but only a one-dimensional eigenspace because B-I has rank one. These checks separate an eigenvalue list from a complete eigenvector basis.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Power iteration | A dominant magnitude and a suitable starting component support convergence. |
| Schur form | A unitary transformation remains available when diagonalization fails. |
| Krylov methods | Selected eigeninformation can be approximated through matrix-vector products. |

The next lecture, {doc}`05-nonlinear-equations`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Which eigenvalue does ordinary power iteration usually target?",
    "options": [
      "It targets the eigenvalue of largest magnitude under suitable assumptions.",
      "It always targets the smallest algebraic eigenvalue."
    ],
    "answer": 0,
    "explain": "Repeated multiplication amplifies the component with greatest eigenvalue magnitude."
  },
  {
    "q": "Why can power iteration miss the dominant eigenvector?",
    "options": [
      "Normalization always removes the dominant component.",
      "The starting vector can have zero component in that direction."
    ],
    "answer": 1,
    "explain": "The diagonalizable exact-arithmetic iteration cannot create a missing eigenvector component."
  },
  {
    "q": "What does a small eigenpair residual check?",
    "options": [
      "It checks the equation A v approximately equals lambda v.",
      "It proves the eigenvector is insensitive to all perturbations."
    ],
    "answer": 0,
    "explain": "Eigenvector sensitivity also depends on separation and matrix structure."
  },
  {
    "q": "Which decomposition exists even for a defective square matrix?",
    "options": [
      "A complete eigenvector diagonalization always exists.",
      "A Schur decomposition exists."
    ],
    "answer": 1,
    "explain": "Schur form uses a triangular or real block-triangular representation rather than requiring independent eigenvectors."
  },
  {
    "q": "Why can inverse iteration target a chosen eigenvalue?",
    "options": [
      "A shift near that eigenvalue makes its inverse-transformed magnitude large.",
      "Inversion makes all eigenvalues equal."
    ],
    "answer": 0,
    "explain": "The transformed eigenvalues are reciprocals of the shifted eigenvalues."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 4 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap04.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
