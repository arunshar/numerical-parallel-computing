# 13. Fourier transforms, fast algorithms, and convolution

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`12-pde` used Fourier modes to explain the stability of a periodic stencil. The Fourier transform makes those modes into a practical coordinate system for signals and operators. We will compare a direct discrete transform with an FFT, demonstrate aliasing, and compute a linear convolution without accidentally wrapping its ends together.

:::{admonition} Learning goals
- You will explain the discrete Fourier transform as a change of basis.
- You will distinguish the DFT from the FFT algorithm that computes it.
- You will interpret frequency bins, aliasing, and normalization conventions.
- You will use zero-padding to compute linear convolution with an FFT.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## The DFT measures agreement with periodic patterns

A **discrete Fourier transform** expresses a vector as a combination of complex sinusoidal basis vectors. Each coefficient measures how strongly the samples align with one periodic pattern. The common forward convention uses a negative sign in the exponential, while the inverse uses a positive sign and divides by the number of samples.

$$
X_k=\sum_{j=0}^{N-1}x_j e^{-2\pi i jk/N},\qquad
x_j=\frac1N\sum_{k=0}^{N-1}X_k e^{2\pi i jk/N}.
$$

A different normalization convention can split the scaling between forward and inverse transforms. Both conventions are valid when used consistently. A transform library's definition determines how you interpret coefficient magnitudes, energy identities, and convolution scaling.

The complex exponential combines cosine and sine. Real input creates conjugate symmetry, so negative-frequency information is redundant with positive-frequency information. A real-input FFT can exploit this symmetry to reduce storage and work. The zero-frequency coefficient is the sum of the samples under the displayed convention.

## Frequency indices become physical frequencies through sampling

If the sampling interval is delta t, the total record duration is N times delta t. The spacing between DFT frequency bins is one divided by that duration. The bin index alone is not a frequency in hertz until the sampling interval is specified.

The DFT treats the finite record as one period of a periodic sequence. A mismatch between the first and last parts of a record can create an apparent discontinuity at the periodic join. **Spectral leakage** distributes energy across bins when a sinusoid does not align with a bin or when the record boundary is discontinuous.

A window tapers the record to reduce some leakage, but it changes amplitude scaling and frequency resolution. Zero-padding samples the discrete-time Fourier transform on a denser frequency grid. It can make a spectrum easier to interpolate visually, but it does not create new measurement information or improve the fundamental resolution set by the record duration.

## Aliasing makes different continuous signals look identical

Sampling cannot distinguish frequencies separated by integer multiples of the sampling frequency. Real cosines also identify positive and negative frequency counterparts. The lab samples sixteen points and compares cosine bins three and thirteen. They produce the same real samples because thirteen is equivalent to minus three modulo sixteen.

The **Nyquist limit** describes the highest recoverable frequency for a band-limited signal under ideal uniform sampling. An antialiasing filter removes higher-frequency content before sampling. Filtering after aliasing cannot generally recover which original high frequency produced the observed samples.

This distinction matters in spatial grids as well as time series. A fine oscillation in a simulated field can appear as a different coarse oscillation after downsampling. A numerical method that moves information between grids should therefore consider frequency content, not merely array shape.

:::{admonition} Try it
:class: exercise
You can sample cosine bins two and fourteen on the same sixteen-point grid. Then sample a frequency of 2.5 cycles per record and inspect which bins receive energy. Explain why the second experiment concerns leakage while the first demonstrates aliasing.
:::

## An FFT reuses structure in the DFT sum

A direct DFT evaluates N sums with N terms each, giving quadratic arithmetic work. A **fast Fourier transform** reorganizes the same calculation to reuse repeated subproblems. A radix-two Cooley-Tukey decomposition splits the input into even and odd indices.

The two smaller transforms combine through twiddle factors, which are roots of unity. Each recursion level performs work proportional to N, and there are logarithmically many levels for powers of two. The resulting work is proportional to N log N.

$$
T(N)=2T(N/2)+O(N)=O(N\log N).
$$

The FFT is an algorithm family, while the DFT is the mathematical transform. Mixed-radix algorithms handle lengths with other small factors. Prime-size methods use other identities, including reductions to convolution. Practical performance depends on transform length, memory access, planning, vectorization, and communication.

The lab constructs the dense DFT matrix and compares its product with NumPy's FFT. Their answers agree to rounding error. That check establishes the chosen sign and normalization conventions for a small example. It is not a performance benchmark, and the dense matrix approach should not be used for large production transforms.

## Multiplication in frequency corresponds to circular convolution

The DFT convolution theorem relates pointwise multiplication of transforms to **circular convolution** of sequences. Circular convolution wraps indices modulo the transform length. This is appropriate for periodic signals, but ordinary finite sequences usually require linear convolution.

Two sequences of lengths m and n have a linear convolution of length m plus n minus one. Padding both sequences to at least that length prevents the nonzero result from wrapping onto itself. The lab multiplies the transforms of the sequences one, two, three and four, five, then inverts the product.

The result is four, thirteen, twenty-two, fifteen. A shorter transform would fold the tail into earlier entries. Numerical roundoff can leave tiny imaginary parts even when the mathematical result is real, so the lab takes the real part only after using a real-input problem with a known identity.

FFT convolution supports polynomial multiplication, signal filtering, and some integral operators. For very short filters, direct convolution can be faster because FFT setup and extra memory movement outweigh the asymptotic saving. Overlap-add and overlap-save methods process long streams in blocks while handling the convolution boundaries carefully.

## Fourier methods connect calculus and numerical models

Differentiating a complex Fourier mode multiplies it by its frequency times the imaginary unit. A spectral derivative therefore transforms a periodic smooth function, multiplies coefficients by frequency factors, and transforms back. High frequencies receive larger multipliers, so differentiation still amplifies noise.

Fast Poisson solvers can diagonalize certain regular-grid operators with Fourier, sine, or cosine transforms depending on the boundary conditions. The transform choice must match those conditions. Using a periodic FFT for a nonperiodic boundary problem solves a different mathematical problem.

**Wavelets** provide another basis that localizes information in both scale and position. A wavelet transform can represent abrupt localized changes more compactly than a global Fourier expansion. The simplest Haar transform combines neighboring averages and differences. Wavelet methods require their own boundary, scaling, and reconstruction conventions.

:::{admonition} Try it
:class: exercise
You can repeat the convolution with a transform length of three and compare it with the length-four result. Identify exactly which output terms wrap together. Then increase the padding beyond four and confirm that the meaningful linear-convolution entries stay unchanged up to rounding.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/13_fourier_fft.py
```

```python
"""Check a DFT convention, aliasing, and convolution padding."""
import numpy as np

n = 16
j = np.arange(n)
signal = np.cos(2*np.pi*3*j/n)
F = np.exp(-2j*np.pi*np.outer(j,j)/n)
direct = F@signal
fast = np.fft.fft(signal)
print(f"DFT versus FFT difference: {np.max(np.abs(direct-fast)):.6e}")
print("nonzero frequency bins:", np.flatnonzero(np.abs(fast)>1e-10))
print(f"inverse error: {np.max(np.abs(np.fft.ifft(fast)-signal)):.6e}")
aliased = np.cos(2*np.pi*13*j/n)
print(f"sample difference for cosine bins 3 and 13: {np.max(np.abs(signal-aliased)):.6e}")
a = np.array([1.,2.,3.]);b = np.array([4.,5.])
length = len(a)+len(b)-1
conv = np.fft.ifft(np.fft.fft(a,length)*np.fft.fft(b,length)).real
print("padded FFT convolution:", np.round(conv, 6))
assert np.allclose(conv, np.convolve(a,b))
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
DFT versus FFT difference: 1.317398e-14
nonzero frequency bins: [ 3 13]
inverse error: 1.359740e-16
sample difference for cosine bins 3 and 13: 7.660539e-15
padded FFT convolution: [ 4. 13. 22. 15.]
```

The dense DFT and FFT differ by about 1.32e-14 on the toy vector. The nonzero bins are three and thirteen, and the inverse reconstructs the input near floating-point precision. Padding to four samples gives the linear convolution [4,13,22,15].

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Transform | The DFT specifies a coordinate change and normalization. |
| Algorithm | An FFT reduces the work needed to compute that transform. |
| Sampling | Aliasing and leakage arise from different sampling and record effects. |

The next lecture, {doc}`14-random-simulation`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is the difference between DFT and FFT?",
    "options": [
      "The DFT is a transform, and an FFT is a fast algorithm for it.",
      "They necessarily compute different mathematical quantities."
    ],
    "answer": 0,
    "explain": "The fast algorithm reorganizes the same sums."
  },
  {
    "q": "What does zero-padding do to spectral information?",
    "options": [
      "It restores frequencies already lost to aliasing.",
      "It samples the spectrum more densely without creating new measurements."
    ],
    "answer": 1,
    "explain": "Padding changes interpolation in frequency, not the original sampling information."
  },
  {
    "q": "What convolution does a length-N DFT product naturally produce?",
    "options": [
      "It produces circular convolution modulo N.",
      "It always produces unwrapped linear convolution."
    ],
    "answer": 0,
    "explain": "The transform treats the sequence as periodic."
  },
  {
    "q": "What padding length suffices for sequences of lengths m and n?",
    "options": [
      "A length of one always suffices.",
      "A length of at least m plus n minus one suffices."
    ],
    "answer": 1,
    "explain": "The full linear convolution has m plus n minus one nonzero output positions at most."
  },
  {
    "q": "Why do cosine bins three and thirteen match on sixteen samples?",
    "options": [
      "Thirteen is equivalent to negative three modulo sixteen.",
      "The cosine function has no frequency dependence."
    ],
    "answer": 0,
    "explain": "Real cosine identifies the positive and negative frequency pair under this sampling."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 12 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap12.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
