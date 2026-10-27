# A glossary for both courses

The definitions below use the same terms as the lessons. You should return to the relevant lesson for assumptions, equations, and examples.

## Numerical analysis

Absolute error
: The absolute error is the magnitude of the difference between an approximation and its reference value.

Relative error
: The relative error divides an absolute error by the magnitude of a nonzero reference value.

Conditioning
: Conditioning describes how much the exact answer can change when the problem data change slightly.

Stability
: Numerical stability describes how an algorithm's errors compare with the effect of small changes to the input problem.

Backward error
: The backward error measures how much the original data must change to make the computed answer exact for a nearby problem.

Residual
: A residual measures the discrepancy left when a candidate answer is substituted into the original equation.

Truncation error
: Truncation error arises when a finite approximation replaces an infinite process or an exact mathematical operation.

Roundoff error
: Roundoff error arises when a value is represented or computed with finite precision.

Cancellation
: Cancellation removes leading significant digits when nearly equal values are subtracted, which can expose earlier errors.

Pivoting
: Pivoting changes the order of elimination to avoid unsuitable pivots and improve numerical behavior.

Orthogonality
: Two vectors are orthogonal when their inner product is zero under the chosen inner product.

Singular value
: A singular value describes a stretching factor of a linear map along appropriate orthogonal directions.

Spectral radius
: The spectral radius is the largest magnitude among a matrix's eigenvalues.

Preconditioner
: A preconditioner transforms a system so an iterative method can converge more effectively while preserving the recoverable solution.

Interpolation
: Interpolation constructs an approximation that matches specified data values exactly at the interpolation nodes.

Approximation
: Approximation constructs a simpler representation whose error is controlled according to a chosen criterion.

Quadrature
: Quadrature estimates an integral with a weighted sum of function values.

Stiffness
: Stiffness describes a differential-equation setting in which stability forces some explicit methods to take much smaller steps than accuracy alone would require.

Consistency
: Consistency means that a numerical discretization approaches the target equation in the appropriate small-step limit.

Convergence
: Convergence means that a sequence of approximations approaches the desired mathematical object under stated assumptions.

Convexity
: Convexity means that a function's value on a line segment does not exceed the corresponding linear interpolation of endpoint values.

KKT conditions
: The Karush–Kuhn–Tucker conditions combine stationarity, feasibility, and complementary slackness for constrained optimization under suitable assumptions.

## Parallel computing

Task
: A task is a unit of computational work that the algorithm can schedule subject to its dependencies.

Work
: Work counts the total operations required by a parallel algorithm under a stated cost model.

Span
: Span measures the longest dependency path through the task graph under that model.

Speedup
: Speedup divides a reference execution time by the parallel execution time for the specified comparison.

Efficiency
: Parallel efficiency divides speedup by the number of processors used in the comparison.

Strong scaling
: Strong scaling studies execution time as processor count grows while the total problem size stays fixed.

Weak scaling
: Weak scaling studies execution time while problem size grows with processor count to keep work per processor approximately fixed.

Latency
: Latency is the delay before a requested operation or transfer produces its result.

Bandwidth
: Bandwidth is the rate at which data can move through a specified interface.

Arithmetic intensity
: Arithmetic intensity divides an operation count by bytes moved across a specified memory boundary.

Race condition
: A race condition occurs when the result depends on an uncontrolled ordering of concurrent operations.

Data race
: A data race involves conflicting concurrent memory accesses without the synchronization required by the language or memory model.

Deadlock
: Deadlock occurs when participants wait for conditions that their current waits prevent from occurring.

False sharing
: False sharing occurs when independent variables share a coherence unit and cause unnecessary coherence traffic between processors.

Collective operation
: An MPI collective operation involves the ranks in a communicator according to the operation's matching rules.

Nonblocking operation
: A nonblocking operation returns control before the operation has necessarily completed and requires a later completion check before protected resources can be reused.

Communicator
: An MPI communicator identifies a group of ranks and a communication context.

Load balance
: Load balance describes how evenly useful work is distributed among available workers.

## GPU computing

Kernel
: A CUDA kernel is a device function launched over a configured collection of threads.

Thread block
: A thread block is a group of CUDA threads that can cooperate through block-level synchronization and shared memory.

Warp
: A warp is the group of threads scheduled together under NVIDIA's execution model, subject to the architecture's control-flow rules.

Occupancy
: Occupancy compares active warps with the hardware limit for the relevant multiprocessor configuration.

Coalescing
: Coalescing combines suitable thread memory accesses into fewer memory transactions under the device's access rules.

Shared-memory bank conflict
: A bank conflict occurs when a group of shared-memory accesses requires extra service because its addresses compete for a bank under the hardware's rules.

Tiling
: Tiling groups data and operations into blocks so a loaded value can serve several computations before it is discarded.

Atomic operation
: An atomic operation performs a specified read-modify-write action indivisibly at the scope guaranteed by its API.

Privatization
: Privatization gives workers separate partial state and later combines the results to reduce contention.

Reduction
: A reduction combines many values into fewer values with a specified operation and ordering policy.

Scan
: A scan computes every prefix aggregate of an ordered sequence, using an inclusive or exclusive convention.

Stream
: A CUDA stream orders submitted operations within that stream and interacts with other streams according to explicit and implicit dependency rules.

Pinned memory
: Pinned host memory remains resident so device transfers can use the supported transfer path without ordinary pageable-memory handling.

Dynamic parallelism
: CUDA dynamic parallelism allows eligible device code to launch additional device work under the toolkit and hardware rules.

Roofline model
: A roofline model bounds attainable performance using computational throughput, memory bandwidth, and arithmetic intensity for a chosen memory level.
