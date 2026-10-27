# 14. Random numbers, simulation, and uncertainty

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`13-fourier-fft` exploited deterministic algebraic structure. Some numerical problems are easier to solve by sampling, especially when the dimension is high or the quantity itself is an expectation. Random simulation replaces a deterministic approximation error with a statistical uncertainty that must be measured and reported. We will estimate an integral and compare random samples with a simple low-discrepancy sequence.

:::{admonition} Learning goals
- You will distinguish pseudorandom generation from independent ideal randomness.
- You will estimate Monte Carlo uncertainty and interpret a standard error.
- You will transform uniform samples into other distributions.
- You will compare variance reduction with quasi-Monte Carlo sampling.
:::

## Setup

This lecture uses the CPU setup from {doc}`01-errors-floating-point`. The following standalone experiment needs only Python and NumPy.

## A generator produces a reproducible sequence

A **pseudorandom number generator** evolves a finite internal state through a deterministic rule. A seed initializes that state. The output is reproducible under the same generator and relevant implementation conditions, which is valuable for debugging and comparing algorithms.

Reproducibility does not make the sequence truly independent or suitable for every use. A finite-state generator eventually repeats. Period length alone does not measure quality because correlations or lattice structure can harm a simulation long before the period is exhausted. Scientific simulation and cryptographic security also require different properties.

A linear congruential generator updates an integer state by multiplying, adding, and taking a modulus. Its simplicity makes the mechanism easy to study, but poor parameters can create severe patterns. Lagged Fibonacci generators combine older sequence values. Modern libraries provide better-designed generators and APIs for managing independent streams.

Parallel simulation should not assign seeds casually and assume independence. A supported stream-splitting or counter-based scheme can make the relationship between workers explicit. The reproducibility record should include the generator family, seed, library version, and sampling design.

## Monte Carlo turns an integral into an expectation

If U is uniform on zero to one, the expectation of f of U equals the integral of f over that interval. The sample mean of independent evaluations estimates that expectation. For the toy function U squared, the exact integral is one-third.

$$
\hat{I}_N=\frac1N\sum_{i=1}^{N}f(U_i).
$$

The estimator is unbiased when the sampling assumptions hold and the expectation exists. Its variance is the variance of one sample divided by N. Its standard error therefore decreases proportionally to one divided by square-root of N.

$$
\operatorname{SE}(\hat{I}_N)=\frac{\sigma_f}{\sqrt{N}}.
$$

Reducing the standard error by a factor of ten requires about one hundred times as many independent samples. This slow convergence is a real cost. Its rate does not directly worsen with dimension in the same way as a full tensor-product grid, which makes Monte Carlo attractive for high-dimensional integrals.

## A standard error is an uncertainty estimate

The sample standard deviation divided by square-root of N estimates the standard error when observations are independent and the variance is finite. A normal approximation can support an approximate confidence interval for sufficiently regular large samples. The interval is a statement about the sampling procedure, not a guarantee that one realized estimate lies near the truth.

The lab prints the absolute error against the known one-third reference and the estimated standard error. These are different quantities. The actual error is available only because the toy integral is known, while the standard error is what a real simulation would estimate from its samples.

An individual run need not improve monotonically as sample count grows. A lucky small run can beat an unlucky larger run. Comparing methods requires multiple replications or an appropriate paired design when the claim concerns typical statistical performance. The fixed seed in this tutorial makes the demonstration reproducible rather than proving a universal ranking.

## Transformations create useful distributions

The **inverse transform method** applies an inverse cumulative distribution function to a uniform sample. For an exponential distribution with rate lambda, the transform is minus the logarithm of one minus U divided by lambda.

$$
X=-\frac{\log(1-U)}{\lambda}.
$$

The lab uses `log1p` to evaluate the logarithm accurately when its argument is near one. The expected exponential value is one divided by the rate, so a rate of two gives mean one-half. The simulated mean fluctuates around that value.

Rejection sampling draws from an easier proposal distribution and accepts points according to a bounded density ratio. A poor proposal can reject most samples. Box-Muller transforms two uniform samples into normal samples through polar geometry. Correlated normal vectors can be formed by applying a covariance factor, provided the covariance is positive semidefinite and factored appropriately.

:::{admonition} Try it
:class: exercise
You can change the exponential rate from two to four. Predict the mean before rerunning the sample. Then compare the median with the mean and explain why a skewed distribution need not place half its mass on each side of its mean.
:::

## Variance reduction spends samples more carefully

**Stratified sampling** divides the domain into regions and samples within each region. It reduces the chance that a finite sample misses a whole important part of the domain. **Antithetic sampling** pairs related samples whose errors tend to cancel, such as U and one minus U for some monotone integrands.

A **control variate** subtracts a correlated quantity with known expectation and adds that expectation back exactly. The method can reduce variance without changing the desired mean when applied correctly. The control coefficient and its estimation affect the practical benefit.

**Importance sampling** draws more often from important regions and corrects for the changed distribution through weights. The proposal must cover the target contribution. Extremely variable weights can produce a poor estimator even when every formula is formally correct. Effective sample size and weight diagnostics help expose this failure.

Markov chain Monte Carlo produces correlated samples from a target distribution through a transition rule. The ordinary independent-sample standard-error formula is then inappropriate. Autocorrelation, convergence diagnostics, and effective sample size become part of the numerical analysis.

## Low-discrepancy points replace randomness with coverage

A **quasi-Monte Carlo sequence** tries to cover the domain evenly instead of mimicking independent randomness. The lab uses the base-two radical inverse, also called a one-dimensional van der Corput sequence. Reversing the binary digits of an integer places points at fractions that progressively fill gaps.

For some smooth low-dimensional integrands, low-discrepancy sampling can converge faster than ordinary random sampling. Its deterministic error does not have the same standard-error interpretation as independent Monte Carlo. Randomized quasi-Monte Carlo can combine useful coverage with repeated randomizations for uncertainty estimation.

The benefit depends on dimension, effective dimension, smoothness, coordinate ordering, and the specific integrand. The lab's simple one-dimensional polynomial is favorable and should not be treated as evidence that one sequence wins for every problem. Discontinuities or poorly aligned high-dimensional structure can reduce the advantage.

:::{admonition} Try it
:class: exercise
You can replace the squared function with an indicator that is one below 0.37 and zero elsewhere. Its exact integral is 0.37. Compare random and low-discrepancy estimates, and explain which smoothness assumption changed.
:::

## Runnable experiment

You can run this experiment from the repository root with the following command. Its complete implementation appears below.

```bash
python labs/numerical/14_random_simulation.py
```

```python
"""Estimate an integral and contrast random with low-discrepancy points."""
import numpy as np

def radical_inverse(k, base=2):
    value, factor = 0., 1/base
    while k:
        value += factor*(k % base)
        k //= base
        factor /= base
    return value

rng = np.random.default_rng(2026)
for n in [64,256,1024,4096]:
    u = rng.random(n)
    values = u*u
    estimate = values.mean()
    se = values.std(ddof=1)/np.sqrt(n)
    quasi = np.array([radical_inverse(k) for k in range(1,n+1)])
    print(f"n={n:4d}: random error={abs(estimate-1/3):.6e}, estimated SE={se:.6e}, quasi error={abs(np.mean(quasi**2)-1/3):.6e}")
u = rng.random(10000)
exponential = -np.log1p(-u)/2
print(f"exponential sample mean: {exponential.mean():.6f}; exact mean: 0.500000")
assert np.all(exponential>=0)
```

### Captured output

The following output comes from the saved local CPU run. Tiny floating-point differences can vary across platforms.

```text
n=  64: random error=3.252338e-02, estimated SE=3.114520e-02, quasi error=7.770856e-03
n= 256: random error=4.356464e-03, estimated SE=1.883435e-02, quasi error=1.950567e-03
n=1024: random error=2.493155e-02, estimated SE=9.457994e-03, quasi error=4.881221e-04
n=4096: random error=1.885137e-03, estimated SE=4.649878e-03, quasi error=1.220604e-04
exponential sample mean: 0.503652; exact mean: 0.500000
```

The fixed-seed Monte Carlo errors are not monotonic as sample count grows, even though the estimated standard error generally decreases. The low-discrepancy sequence performs well on this smooth one-dimensional example. The exponential sample mean is about 0.504, close to its exact mean 0.5, without being required to equal it exactly.

## Summary

| Concept | What the experiment and explanation establish |
|---|---|
| Random generation | A seed controls reproducibility but does not prove statistical independence. |
| Uncertainty | Monte Carlo standard error typically decreases as the inverse square root of samples. |
| Coverage | Low-discrepancy designs target even coverage with a different error interpretation. |

The next lecture, {doc}`15-iterative-solvers`, extends the numerical toolkit with a different mathematical structure.

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "How many samples are typically needed to reduce Monte Carlo standard error tenfold?",
    "options": [
      "About one hundred times as many independent samples are needed.",
      "Only twice as many samples are needed."
    ],
    "answer": 0,
    "explain": "Standard error scales as one divided by square-root of N."
  },
  {
    "q": "Does a fixed seed establish that a generator is statistically adequate?",
    "options": [
      "It proves independence of every generated value.",
      "It establishes reproducibility, not adequacy by itself."
    ],
    "answer": 1,
    "explain": "Generator quality and the sampling design require additional analysis."
  },
  {
    "q": "Why can actual error increase in one larger random run?",
    "options": [
      "Sampling fluctuations need not improve monotonically.",
      "The law of large numbers forbids every finite fluctuation."
    ],
    "answer": 0,
    "explain": "Asymptotic convergence does not impose monotonic error on individual realizations."
  },
  {
    "q": "What is the mean of the exponential distribution with rate two?",
    "options": [
      "Its mean is two.",
      "Its mean is one-half."
    ],
    "answer": 1,
    "explain": "The exponential mean is the reciprocal of its rate."
  },
  {
    "q": "Can the independent Monte Carlo standard-error formula be used unchanged for a deterministic low-discrepancy sequence?",
    "options": [
      "It needs a different error or randomization analysis.",
      "It always applies without modification."
    ],
    "answer": 0,
    "explain": "The sequence does not provide independent identically distributed random samples."
  }
]
```

## Source mapping and scope

This is an original teaching explanation with original toy code. The topic mapping was checked against [Heath’s chapter 13 slides](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/Slides3/chap13.pdf), the [author’s contents](https://heath.cs.illinois.edu/scicomp/contents/index.html), and the [CSCI 5302 public notes](https://classpages.cselabs.umn.edu/Spring-2025/csci5302/Notes/). The [Ascher and Greif chapter listing](https://epubs.siam.org/doi/book/10.1137/9780898719987) provides a second reading route. The full textbook texts were not available for a line-by-line coverage audit.

The implemented experiment verifies the mechanism identified in its explanation. Other named algorithm families receive explanatory survey coverage here and need dedicated derivations and implementations for mastery. The coverage ledger records that distinction.
