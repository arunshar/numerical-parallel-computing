# 28. Parallel Fourier transforms and data redistribution

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`27-search-dynamic-programming` exposed parallelism by following a recurrence. The fast Fourier transform also follows a structured recurrence, but its data exchanges form regular butterfly patterns. This lecture connects the numerical FFT to binary-exchange and transpose-based parallel algorithms.

:::{admonition} Learning goals
- You can derive an FFT butterfly from even and odd input subsequences.
- You can identify which butterfly stages require communication under a block layout.
- You can explain the role of transposes in distributed multidimensional transforms.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/28-parallel-fft.py`.

## The transform separates a signal into frequencies

The discrete Fourier transform of $n$ complex samples is $X_k=\sum_{j=0}^{n-1}x_j\exp(-2\pi i jk/n)$. The sign and normalization are conventions that must match the inverse transform. A direct implementation evaluates $n$ sums of $n$ terms and performs order $n^2$ arithmetic. The FFT is a family of algorithms that computes the same transform more efficiently by reusing structure in the roots of unity.

For even $n$, split the input into even and odd indices. Their transforms E and O have length $n/2$. The output satisfies $X_k=E_k+\omega_n^kO_k$ and $X_{k+n/2}=E_k-\omega_n^kO_k$, where $\omega_n=\exp(-2\pi i/n)$. This pair of updates is a **butterfly**. Recursing to small transforms produces order $n\log n$ work. The recursion changes the evaluation order, so a floating-point FFT need not match direct summation bit for bit.

Radix-2 assumes a power-of-two factorization. Mixed-radix methods use other factors, and algorithms such as Bluestein handle awkward lengths through convolution. Padding a signal to a convenient length changes the sampled frequency grid and can change the application semantics. Padding is appropriate for some convolution calculations when enough zeros are added, but it is not a universal way to replace an arbitrary transform with an equivalent one.

## An iterative FFT has stages with different partners

An iterative radix-2 implementation commonly begins with a bit-reversal permutation and then doubles the butterfly width at each stage. For eight samples, the stages combine groups of 2, 4, and 8. Within one stage, butterflies are independent. Between stages, outputs become inputs for the next stage. A stage therefore offers data parallelism, while the stage sequence creates the span.

Bit reversal maps the binary representation of an index to its reversed bit order. With three bits, index 3 is 011 and maps to 110, which is 6. The permutation can be performed explicitly or absorbed into another layout convention. An implementation must document whether its input or output is in natural order. A numerically plausible but permuted spectrum is still an incorrect result for a caller expecting natural order.

Twiddle factors are the roots of unity used by each butterfly. Precomputing them saves repeated trigonometric evaluation but consumes memory. Recurrence-based generation can reduce storage while accumulating additional rounding error. Libraries choose among many decompositions and small kernels because the best arithmetic count alone does not determine cache behavior or vectorization efficiency.

## Binary exchange follows the butterfly graph

Suppose $n$ samples are distributed in contiguous blocks across $p=2^q$ ranks, with $n/p$ samples per rank. Some butterfly stages pair values inside one rank's block, while later stages pair values on different ranks. A binary-exchange algorithm communicates with partners whose rank identifiers differ in one selected bit. The pattern mirrors a hypercube even when the physical network is not a hypercube.

For $n=16$ and $p=4$, each rank owns four consecutive positions in the simplified layout. A butterfly pairing indices that differ in bit 0 or bit 1 remains local. Pairings that differ in bit 2 or bit 3 cross rank boundaries. The implementation may choose a different input permutation or cyclic layout, which changes which stages are local. The important method is to derive ownership for both endpoints of every butterfly rather than memorize a fixed stage rule.

Each rank performs order $(n/p)\log n$ arithmetic. A simple block-based exchange model has order $\log p$ communicating stages with order $n/p$ data per stage. This gives a useful first approximation, but exact volume depends on packing, duplication, and the chosen FFT decomposition. Network contention and the cost of permutations can dominate at high process counts.

## Transpose algorithms move whole pencils of data

A multidimensional transform can be computed through one-dimensional transforms along each axis. A rank must have the relevant line of data locally or communicate during that transform. A **slab decomposition** distributes whole planes or thick slabs. It can limit the number of useful ranks when one dimension is small. A **pencil decomposition** distributes thinner two-dimensional partitions and supports a larger process grid, but it needs more redistribution stages.

A distributed transpose changes which dimension is contiguous and locally owned. The application first performs local transforms along a resident axis, redistributes data, then performs transforms along the next axis. All-to-all or subgroup all-to-all operations often implement that redistribution. Packing into destination-contiguous buffers can improve transfer efficiency but adds memory traffic and temporary storage.

A transpose-based one-dimensional FFT can reinterpret a long transform through a matrix-shaped decomposition, local transforms, twiddle multiplication, and transposes. The algebra must account for the chosen index factorization. Simply taking a matrix FFT without the required twiddle terms does not reproduce an arbitrary long one-dimensional DFT. The decomposition is valuable because bulk transposes can replace many fine-grained exchanges.

## Transform applications impose extra checks

Circular convolution corresponds to pointwise multiplication in the Fourier domain. Linear convolution requires enough zero padding to prevent wraparound. Real-input transforms exploit conjugate symmetry and store fewer complex coefficients, but the packed layout must be understood before combining it with another kernel. Frequency bins also depend on sample spacing, so a peak index alone is not a physical frequency.

The experiment compares an original radix-2 implementation with NumPy's FFT for eight samples and checks the inverse reconstruction. A separate ownership count classifies butterfly pairs as local or remote for a four-rank block layout. These are algorithmic checks. The script does not call distributed FFT software or measure interconnect traffic.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
import numpy as np
def fft(x):
 x=np.asarray(x,dtype=complex);n=len(x)
 if n==1:return x.copy()
 assert n%2==0
 e,o=fft(x[::2]),fft(x[1::2])
 w=np.exp(-2j*np.pi*np.arange(n//2)/n)*o
 return np.concatenate([e+w,e-w])
x=np.array([1,2,0,-1,3,0,2,1.])
y=fft(x)
error=float(np.max(np.abs(y-np.fft.fft(x))))
assert error<1e-12
n,p=16,4;local=n//p;stages=[]
for bit in range(4):
 pairs=[(i,i^(1<<bit)) for i in range(n) if i<(i^(1<<bit))]
 remote=sum(a//local!=b//local for a,b in pairs)
 stages.append({'partner_bit':bit,'remote_pairs':remote})
assert [s['remote_pairs'] for s in stages]==[0,0,8,8]
print(json.dumps({'max_reference_error':error,'reconstruction_ok':bool(np.allclose(np.fft.ifft(y),x)),
                  'ownership_stages':stages},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "max_reference_error": 6.280369834735101e-16,
  "reconstruction_ok": true,
  "ownership_stages": [
    {
      "partner_bit": 0,
      "remote_pairs": 0
    },
    {
      "partner_bit": 1,
      "remote_pairs": 0
    },
    {
      "partner_bit": 2,
      "remote_pairs": 8
    },
    {
      "partner_bit": 3,
      "remote_pairs": 8
    }
  ]
}
```

The reference error stays below **1e-12** for this small example, and the inverse reconstructs the input within floating-point tolerance. The ownership model has zero remote pairs in the first two partner-bit stages and eight in each of the next two. That count depends on the stated contiguous-block layout.

:::{admonition} Try it
:class: exercise
You can change the ownership rule to cyclic assignment `owner = index % p`. Predict which partner bits become remote and verify the count.
:::

:::{admonition} Try it
:class: exercise
You can convolve two length-four signals directly and through FFTs. Compare transform lengths 4 and 8 and explain why only adequate padding gives the full linear convolution.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| FFT structure | Even-odd decomposition reuses smaller transforms through butterflies. |
| Distributed ownership | A butterfly requires communication when its endpoints have different owners. |
| Transpose design | Bulk redistribution can make the next transform axis local. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What does the FFT compute?",
    "options": [
      "The same DFT through a faster factorization.",
      "A different approximate frequency definition."
    ],
    "answer": 0,
    "explain": "Rounding differs, but the mathematical transform is the same."
  },
  {
    "q": "What is radix-2 FFT work for n samples?",
    "options": [
      "Order n log n.",
      "Order n cubed."
    ],
    "answer": 0,
    "explain": "There are logarithmically many stages with linear work per stage."
  },
  {
    "q": "Why can an FFT need a distributed transpose?",
    "options": [
      "The next transform axis may be split across ranks.",
      "A transpose changes addition into multiplication."
    ],
    "answer": 0,
    "explain": "Redistribution changes local ownership."
  },
  {
    "q": "Does zero padding always preserve the original transform samples?",
    "options": [
      "No, it changes the frequency grid.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Padding has application-specific meaning."
  },
  {
    "q": "What is the modeled remote-pair count in partner bit 2?",
    "options": [
      "8.",
      "0."
    ],
    "answer": 0,
    "explain": "Bit 2 changes the owner under four-element contiguous blocks."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
