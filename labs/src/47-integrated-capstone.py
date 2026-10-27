# %% [markdown]
# # 47. A numerical and parallel computing capstone
#
# Prepared for Arun Sharma. Codex drafted this study edition.
#
# The earlier lessons considered numerical accuracy, decomposition, communication, and hardware separately. This capstone puts those decisions into one small example. We solve a steady one-dimensional diffusion problem, check its error, and model how its iterative update could be divided among workers.
#
# A useful solver needs three kinds of evidence. The discrete equation should approximate the intended continuous equation. The algorithm should solve the discrete system accurately. The implementation should perform the required work without invalid data dependencies. A speed claim becomes meaningful only after these conditions are checked.
#
# :::{admonition} Learning goals
# - You will separate discretization error from algebraic error.
# - You will compare a partitioned update with a serial reference.
# - You will estimate when communication limits strong scaling.
# - You will write a reproducibility record that preserves the scope of each result.
# :::
#
# ## Setup
#
# This notebook uses NumPy and the shared quiz helper. The helper is loaded locally when the notebook runs inside the repository. A fresh Colab session downloads the same public helper. The experiment runs on a CPU and does not launch MPI or CUDA.
#
# :::{note}
# The partitioned update is a synchronous CPU simulation. It checks the ownership and halo arithmetic. It does not measure a distributed network or a GPU.
# :::
# %%
from pathlib import Path
import sys
import urllib.request
import importlib.util
import numpy as np

helper_url = "https://raw.githubusercontent.com/arunshar/numerical-parallel-computing/main/labs/tutorial_helpers.py"
candidates = [Path("../../labs/tutorial_helpers.py"), Path("labs/tutorial_helpers.py")]
helper = next((p.resolve() for p in candidates if p.exists()), None)
if helper is None:
    helper = Path("tutorial_helpers.py").resolve()
    urllib.request.urlretrieve(helper_url, helper)
spec = importlib.util.spec_from_file_location("tutorial_helpers", helper)
th = importlib.util.module_from_spec(spec)
spec.loader.exec_module(th)
SYSTEM = {"interior_points": [15, 31, 63], "workers": 4, "iterations": 100}
# %% [markdown]
# ## The continuous problem gives us an independent reference.
#
# We solve the boundary-value problem
#
# $$-u''(x)=\pi^2\sin(\pi x),\qquad 0<x<1,\qquad u(0)=u(1)=0.$$
#
# The exact solution is $u(x)=\sin(\pi x)$. We chose this manufactured problem because its answer is known independently of the numerical solver. We can therefore measure an actual solution error, not just the residual of a discrete equation.
#
# A uniform grid with $n$ interior points has spacing $h=1/(n+1)$. The centered difference approximation gives
#
# $$\frac{-u_{i-1}+2u_i-u_{i+1}}{h^2}=\pi^2\sin(\pi x_i).$$
#
# The matrix has a diagonal of $2/h^2$ and adjacent diagonals of $-1/h^2$. The boundary values are zero, so they add no contribution to the right-hand side. A nonzero boundary value would change the first or last equation and must be handled explicitly.
#
# The centered formula has second-order truncation error for a sufficiently smooth solution. Halving the grid spacing should therefore reduce the leading discretization error by about a factor of four. That prediction depends on reaching the asymptotic regime and solving the discrete system accurately enough.
# %%
def problem(n):
    h = 1.0 / (n + 1)
    x = np.arange(1, n + 1) * h
    A = (2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / h**2
    b = np.pi**2 * np.sin(np.pi*x)
    return h, x, A, b

errors = []
for n in SYSTEM["interior_points"]:
    h, x, A, b = problem(n)
    u = np.linalg.solve(A, b)
    error = np.max(np.abs(u - np.sin(np.pi*x)))
    residual = np.linalg.norm(A @ u - b, ord=np.inf)
    errors.append(error)
    print(f"n={n:2d} h={h:.6f} solution_error={error:.8f} residual={residual:.3e}")
ratios = np.array(errors[:-1]) / np.array(errors[1:])
print("successive error ratios:", np.round(ratios, 5))
assert np.all((ratios > 3.9) & (ratios < 4.1))
# %% [markdown]
# The error ratios should be close to **four**. The residual is much smaller than the solution error because the linear solver nearly satisfies the discrete equations. The remaining error comes mainly from replacing the continuous derivative with a finite difference.
#
# A small residual alone would not demonstrate second-order accuracy. We needed the exact solution and a grid-refinement experiment. On a problem without an exact solution, we would need a different independent check, such as a trusted finer-grid result or a manufactured test problem with similar structure.
#
# :::{admonition} Try it
# :class: exercise
# You should replace the three grid sizes with 7, 15, and 31. Predict whether the ratios move closer to four as the grid becomes finer. Then change the forcing function without changing the claimed exact solution and observe why the reference check fails.
# :::
#
# ## A Jacobi update exposes the communication boundary.
#
# A Jacobi iteration computes each new value from the previous iterate. The update is
#
# $$u_i^{(k+1)}=\tfrac12\left(u_{i-1}^{(k)}+u_{i+1}^{(k)}+h^2b_i\right).$$
#
# Each interior point needs two neighbors. A worker that owns a consecutive block already has most of those values. It needs one old value from each adjacent worker at an internal boundary. Those borrowed boundary values are called a **halo**.
#
# The implementation below keeps the old array immutable during an iteration. Each worker writes a disjoint slice of a new array. This separation makes the update mathematically identical to the serial Jacobi step. Updating the old array in place would change the method and introduce an ordering dependency.
# %%
n = 32
h, x, A, b = problem(n)
serial = np.zeros(n)
partitioned = np.zeros(n)
blocks = np.array_split(np.arange(n), SYSTEM["workers"])
for step in range(SYSTEM["iterations"]):
    padded = np.pad(serial, (1, 1))
    serial = 0.5 * (padded[:-2] + padded[2:] + h*h*b)
    old = np.pad(partitioned, (1, 1))
    new = np.empty_like(partitioned)
    for ids in blocks:
        start, stop = int(ids[0]), int(ids[-1]) + 1
        new[start:stop] = 0.5*(old[start:stop] + old[start+2:stop+2] + h*h*b[start:stop])
    partitioned = new
print("largest serial/partitioned difference:", np.max(np.abs(serial-partitioned)))
print("Jacobi residual after 100 updates:", np.linalg.norm(A @ partitioned-b, ord=np.inf))
assert np.array_equal(serial, partitioned)
# %% [markdown]
# The two update paths should agree exactly in this implementation because each output uses the same arithmetic expression and operand order. This agreement is a correctness check for the partition logic. It does not demonstrate parallel execution or a speedup.
#
# The residual after 100 iterations remains nonzero. That result is expected because Jacobi can converge slowly on a fine diffusion grid. A faster halo exchange would not fix its convergence rate. A better iterative method or preconditioner could reduce the number of communication rounds as well as the arithmetic work.
#
# A real MPI implementation would exchange boundary values before computing the boundary cells. It could start nonblocking communication, compute cells that do not need the incoming values, wait for completion, and then compute the boundaries. The overlap depends on MPI progress and the available interior work. A nonblocking call alone does not prove useful overlap.
#
# ## A cost model predicts a scaling limit.
#
# Suppose one update costs $c$ seconds per point and a halo exchange costs approximately $2\alpha+2\beta s$. Here $\alpha$ is a startup delay, $\beta$ is seconds per byte, and $s$ is the payload per neighbor. An interior worker's simple estimate is
#
# $$T_p\approx cN/p+2\alpha+2\beta s.$$
#
# This model ignores overlap, endpoint differences, network contention, and reductions used for convergence checks. It is a first diagnostic. The compute term shrinks with worker count, but the startup term does not. Strong scaling eventually runs out of enough work per worker to hide that cost.
#
# The next cell uses invented teaching parameters. Its times are model predictions, not benchmark measurements.
# %%
N = 1_000_000
c = 2e-9
alpha = 5e-6
beta = 1 / 10e9
s = 8
baseline = c*N
for workers in [1, 2, 4, 16, 64, 256]:
    communication = 0 if workers == 1 else 2*alpha + 2*beta*s
    estimate = c*N/workers + communication
    print(f"workers={workers:3d} predicted_us={1e6*estimate:9.3f} predicted_speedup={baseline/estimate:8.3f}")
# %% [markdown]
# The predicted speedup grows less rapidly as the fixed communication term becomes important. The model uses one scalar per neighboring boundary because the problem is one-dimensional. A two-dimensional decomposition has a different surface-to-volume relationship, so its halo term must be derived again.
#
# :::{admonition} Try it
# :class: exercise
# You should increase the startup delay by a factor of ten and rerun the model. Predict which worker counts become inefficient. Then derive the halo payload for a square subdomain with one layer of ghost cells and eight-byte values.
# :::
#
# ## A useful report separates the evidence.
#
# Your report should state the continuous problem, discretization, solver, stopping rule, and implementation. It should record the grid size, precision, processor count, compiler, and hardware. It should then show numerical error and runtime separately.
#
# You should compare methods at a comparable accuracy target. A solver that stops earlier can look faster while solving a different effective problem. You should also include setup costs when they matter to the intended use. A preconditioner that costs seconds to build may be valuable across many right-hand sides and wasteful for one tiny solve.
#
# This notebook establishes a grid-refinement trend, an exact agreement between two CPU update paths, and a communication-cost prediction under invented parameters. Native MPI scaling and CUDA performance remain separate experiments. That separation makes the notebook a useful starting point for a later implementation rather than an unsupported performance claim.
#
# ## Summary
#
# | Check | What the check establishes |
# |---|---|
# | Grid refinement compares numerical solutions with the manufactured exact solution. | It tests the expected second-order discretization trend. |
# | The serial and partitioned updates use the same arithmetic. | Their agreement tests the ownership and halo-index logic. |
# | The latency-bandwidth formula predicts iteration cost. | It explains a possible strong-scaling limit under stated assumptions. |
# | A native benchmark would record hardware and actual elapsed time. | It would provide a different kind of evidence that this CPU model does not supply. |
#
# ## Check your understanding
#
# Pick an answer to see whether it is right. After a wrong answer, you can try another option.
# %% tags=["remove-input"]
#@title Run this cell to show the quiz
th.quiz([
 {"q":"What does a tiny linear-system residual establish?","options":["It establishes exact agreement with the continuous PDE.","It establishes near-satisfaction of the discrete equations."],"answer":1,"explain":"Discretization error remains even when the discrete equations are solved accurately."},
 {"q":"What error ratio is expected when a second-order method halves its grid spacing?","options":["The ratio is about two.","The ratio is about four.","The ratio is about eight."],"answer":1,"explain":"The leading error is proportional to h squared, so halving h reduces it by about four."},
 {"q":"Why does the Jacobi implementation preserve an old array?","options":["Every new value must use the previous iteration.","The array makes the network faster."],"answer":0,"explain":"An in-place update would change the algorithm and its dependency order."},
 {"q":"Does exact agreement between the two CPU update paths prove MPI speedup?","options":["Yes.","No."],"answer":1,"explain":"The paths run on a CPU without MPI communication, so they only check the modeled computation."},
 {"q":"Which term prevents ideal strong scaling in the simple model?","options":["The startup cost remains even as local arithmetic shrinks.","The grid spacing is always one."],"answer":0,"explain":"The fixed communication delay eventually becomes large compared with each worker's computation."}
])
# %% [markdown]
# ## Sources and next implementation
#
# The finite-difference and iterative-method ideas connect Heath's boundary-value and PDE chapters with Ascher and Greif's iterative methods. The communication model connects to Grama and colleagues' analytical modeling chapter. The source bibliography appears in {doc}`../reference/sources`.
#
# A next implementation can replace the simulated blocks with MPI ranks or a CUDA stencil. It should preserve the independent numerical checks and add hardware-specific evidence before making a performance claim.
