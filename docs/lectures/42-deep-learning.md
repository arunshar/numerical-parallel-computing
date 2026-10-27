# 42. Convolutional networks, GEMM, and library choices

Prepared for Arun Sharma. Codex drafted this study edition.

The previous lecture, {doc}`41-sparse-graphs`, established the ideas that this lesson extends. This lesson develops an original explanation and a small executable experiment. The numerical inputs are teaching examples, not measurements from the instructor’s machines.

:::{admonition} Learning goals
- You will map a convolutional layer to a matrix multiplication.
- You will account for the memory cost of an explicit patch matrix.
- You will separate numerical precision choices from an operation-count comparison. :::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The reference experiment needs Python and NumPy and finishes on a CPU.

```bash
python labs/gpu/42_deep_learning.py
```

:::{warning} **The experiment runs on a CPU.** It checks arithmetic, indexing, or a stated cost model. It does not demonstrate GPU speed, GPU occupancy, CUDA race freedom, or successful CUDA compilation. Native CUDA execution is **UNVERIFIED** in this edition. :::

## A layer is a structured collection of dot products

A convolutional layer takes an input tensor, a bank of filters, and often a bias. Each output channel applies one filter across the input’s channel and spatial dimensions. A batch dimension repeats the operation for several examples. The output shape depends on stride, padding, dilation, filter size, and input layout.

For one spatial dimension, the output length is

$$L_{out}=\left\lfloor\frac{L_{in}+2p-d(k-1)-1}{s}\right\rfloor+1,$$

where $p$ is padding, $d$ is dilation, $k$ is filter width, and $s$ is stride. The same formula applies separately to height and width. A kernel must implement the same convention as its reference framework, including whether the operation reverses the filter or uses cross-correlation.

A direct implementation assigns an output element to a thread and loops over channels and filter positions. This ownership rule is correct and easy to test. It can reread many input values and coefficients, which motivates reuse, tiling, and matrix-multiplication formulations.

## The patch matrix exposes GEMM

An **im2col** transformation turns each input patch into a column. If each patch has $K=C_{in}R S$ values and there are $P$ output positions across the batch, the patch matrix has shape $K\times P$. Flattened filters form a matrix with shape $C_{out}\times K$. Their product has shape $C_{out}\times P$ and can be reshaped into the output tensor.

The arithmetic is approximately $2C_{out}KP$ FLOP. A one-channel three-by-three filter gives $K=9$. A large image repeats many input values across neighboring patch columns. The explicit patch matrix can therefore be much larger than the original input.

An **implicit GEMM** implementation computes the patch addresses as needed and avoids materializing the complete patch matrix. It retains the matrix-multiplication structure while trading extra address calculations for less intermediate traffic. The best implementation depends on the convolution shape and the hardware’s favored tile structure.

## Other algorithms change the cost profile

A pointwise one-by-one convolution has no spatial neighborhood and often maps especially naturally to GEMM. A depthwise convolution processes each input channel separately and has much less cross-channel reuse. A kernel optimized for dense multi-channel convolution may therefore perform poorly on a depthwise layer.

FFT-based convolution transforms the input and filter, multiplies in the frequency domain, and transforms back. It can reduce arithmetic for sufficiently large filters, but it introduces transforms, padding, temporary storage, and numerical differences. Winograd-style algorithms reduce multiplication counts for selected small filters while increasing transformation work and sensitivity. A lower multiplication count is not enough to establish lower runtime or acceptable accuracy.

A high-quality library selects among implementations using the actual tensor shape, layout, workspace limit, precision, and target GPU. cuDNN is the primary CUDA deep-learning primitive library for such operations. A hand-written teaching kernel should be compared with an appropriate library baseline rather than treated as the default production implementation.

## Fusion removes intermediate traffic

A layer can apply a bias and activation after its convolution. Separate kernels write the convolution output, reread it for the bias, write it again, and reread it for the activation. A fused implementation can apply these operations while the value is still in a register and write only the final result.

Fusion can also enlarge the kernel and increase register demand. An operation that needs a global reduction, such as some normalization schemes, may not fit a simple local fusion. The dependency graph determines which intermediate values can remain local without changing the computation.

Training adds backward passes for inputs and weights. The weight-gradient calculation accumulates contributions across examples and output positions. This accumulation creates different parallelism and numerical-order issues from the forward pass. A complete training benchmark must include the actual forward, backward, optimizer, and communication work that belongs to the workload.

## Precision is a contract

Tensor-core paths can use specialized input and accumulation formats. The precise supported modes depend on the GPU and library version. A throughput comparison must report these formats and the allowed error. A faster low-precision path is useful only when it satisfies the application’s accuracy requirements.

A stable test uses more than one random tensor. It includes zeros, large and small magnitudes, unusual channel counts, nonmultiple tile shapes, and asymmetric filters. It checks both absolute and relative error and records the reference precision. The same shape and arithmetic convention must be used for every implementation being compared.

The reference constructs a small patch matrix and compares its product with direct cross-correlation. It reports the patch-storage expansion and the number of multiplication-addition operations. It does not claim a cuDNN or tensor-core benchmark.

## Run the reference experiment

The complete implementation is shown below. The source file is also available as {download}`42_deep_learning.py <../../labs/gpu/42_deep_learning.py>`.

```python
import numpy as np
x = np.arange(25,dtype=np.float64).reshape(5,5)
filters = np.array([[[1,0,-1],[1,0,-1],[1,0,-1]],
                    [[0,1,0],[1,2,1],[0,1,0]]],dtype=float)
patches = np.column_stack([x[r:r+3,c:c+3].reshape(-1) for r in range(3) for c in range(3)])
y_gemm = (filters.reshape(2,9) @ patches).reshape(2,3,3)
y_direct = np.zeros((2,3,3))
for f in range(2):
    for r in range(3):
        for c in range(3):
            y_direct[f,r,c] = np.sum(filters[f]*x[r:r+3,c:c+3])
assert np.array_equal(y_gemm,y_direct)
print(f"input_values={x.size}, patch_values={patches.size}, expansion={patches.size/x.size:.2f}")
print(f"gemm_shape=(2,9)@(9,9), nominal_flops={2*2*9*9}")
print("first_output_channel=",y_gemm[0].astype(int).tolist())
```

The recorded run produced the following output.

```text
input_values=25, patch_values=81, expansion=3.24
gemm_shape=(2,9)@(9,9), nominal_flops=324
first_output_channel= [[-6, -6, -6], [-6, -6, -6], [-6, -6, -6]]
```

**The patch-matrix product exactly matches the direct calculation for these integer-valued inputs.** The patch matrix stores 81 values for a 25-value input, demonstrating the cost that implicit GEMM seeks to avoid.

:::{admonition} Try it 1 :class: exercise

Add stride two and derive the output dimensions before editing the patch extraction. Check that direct and GEMM formulations still agree. :::

:::{admonition} Try it 2 :class: exercise

Replace the filter bank with a one-by-one convolution over several channels. Explain which duplication disappears and which GEMM dimensions remain. :::

## Summary

| Decision | What you should remember |
|---|---|
| Convolution mapping | Each output is a dot product over a channel-spatial patch. |
| im2col | An explicit patch matrix exposes GEMM but duplicates input values. |
| Libraries | A production comparison should use the same shape, precision, and operation contract. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What does one im2col column represent?",
    "options": [
      "It represents one input patch.",
      "It represents the entire training dataset."
    ],
    "answer": 0,
    "explain": "Each column contains the values needed for one output position."
  },
  {
    "q": "How many values are in the toy patch matrix?",
    "options": [
      "There are 81 values.",
      "There are 25 values."
    ],
    "answer": 0,
    "explain": "Nine patches each contain nine values."
  },
  {
    "q": "What does implicit GEMM avoid?",
    "options": [
      "It avoids materializing the full patch matrix.",
      "It avoids every input load."
    ],
    "answer": 0,
    "explain": "The kernel generates patch addresses while retaining GEMM-like computation."
  },
  {
    "q": "Does fewer multiplications always imply faster convolution?",
    "options": [
      "Yes, no other cost matters.",
      "No, transformations and memory traffic also matter."
    ],
    "answer": 1,
    "explain": "Alternative algorithms can trade multiplication count for other work."
  },
  {
    "q": "Why must precision be stated in a benchmark?",
    "options": [
      "Different formats can change both speed and numerical error.",
      "Precision affects only file naming."
    ],
    "answer": 0,
    "explain": "The arithmetic contract determines whether two results are comparable."
  }
]
```

## Sources and coverage

This lesson maps to *Programming Massively Parallel Processors*, fourth edition, chapters 16. The book is by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The [publisher’s contents page](https://www.educate.elsevier.com/book/details/9780323912310) establishes the chapter mapping. The complete book text was not available for this work, so this lesson does not claim to reproduce or exhaust every passage or exercise.

The [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) provide the primary reference for programming semantics and optimization guidance. Hardware limits must be checked for the actual device. These notes use original examples and preserve the distinction between a model, an executable CPU reference, and an unverified GPU implementation.

The [cuDNN documentation](https://docs.nvidia.com/deeplearning/cudnn/) describes the library interface and supported execution paths. The lesson’s NumPy experiment does not execute cuDNN.
