# 29. Hybrid programming and an introduction to quantum computation

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`28-parallel-fft` combined local arithmetic with distributed communication. Hybrid programs deliberately combine several execution models. Saad’s final notes also introduce quantum computation, which uses a different mathematical state model. This lecture treats those topics separately so that ordinary parallel speedup is not confused with quantum interference.

:::{admonition} Learning goals
- You can explain MPI thread-support levels and a safe hybrid design.
- You can describe CPU-GPU communication ownership and completion.
- You can calculate a small quantum-state evolution and explain the Deutsch-Jozsa promise.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/29-hybrid-quantum.py`.

## Hybrid programs use different models at different scales

A common cluster design uses one or a few MPI ranks per node and several OpenMP threads inside each rank. MPI handles data exchange between address spaces, while threads share local arrays. This can reduce the number of MPI endpoints and copies of rank-private data. It can also introduce thread synchronization, NUMA placement issues, and competition between communication progress and computation. More software layers do not automatically improve performance.

The application should decide which level owns each loop and each data partition. A node might own one spatial subdomain, with threads splitting its interior. The rank exchanges halo data after the required local computation. If a library also creates threads, the total concurrency can exceed available cores. A configuration of four ranks times four threads means up to 16 application threads before counting runtime helpers. Resource accounting must include nested libraries and progress threads.

`MPI_Init_thread` requests a thread-support level and returns the level actually provided. `MPI_THREAD_SINGLE` assumes one thread. `MPI_THREAD_FUNNELED` permits multiple threads but requires the thread that initialized MPI to make MPI calls. `MPI_THREAD_SERIALIZED` permits calls from multiple threads as long as they do not overlap. `MPI_THREAD_MULTIPLE` permits concurrent MPI calls, subject to MPI's remaining rules. A program must inspect the provided level and choose a compatible path.

A funnelled design often keeps MPI calls outside OpenMP parallel regions or uses a carefully controlled main-thread section. A serialized design protects MPI calls with a protocol that prevents overlap. Multiple support can simplify some designs, but its overhead and progress behavior depend on the implementation. It should be requested because the algorithm needs it rather than assumed to be free.

## GPU participation adds another completion boundary

An MPI-plus-CUDA program can compute a local subproblem on a GPU and exchange boundary data through MPI. A host-staging design copies data from device to host, performs MPI communication, then copies incoming data to the device. A CUDA-aware MPI implementation may accept device pointers directly for supported operations. This removes some explicit staging code, but it does not imply that every datatype, stream interaction, or transport path is supported equally.

CUDA kernels and copies can be asynchronous with respect to the host. MPI must not read a device send buffer before the producer operation is complete under the supported interoperability contract. A GPU consumer must likewise wait until incoming data is ready. Events, stream synchronization, and library-specific interoperability features help express these dependencies. A successful run with an implicit synchronization does not prove that a later asynchronous version is safe.

The hybrid performance model should separate local CPU work, GPU work, transfers, communication, and synchronization. Overlap can reduce the critical path when operations are independent and the hardware supports concurrent progress. Adding their isolated runtimes gives a serialized estimate. Taking their maximum gives an ideal overlap estimate when stages can progress independently without resource contention. Neither is a measured pipeline without a timeline or elapsed-time evidence.

## A qubit is a normalized complex state

A **qubit** has a state $\alpha|0\rangle+\beta|1\rangle$ with $|\alpha|^2+|\beta|^2=1$. Measurement in the computational basis returns 0 or 1 with those respective probabilities. The amplitudes can have phases, and those phases influence later interference. A global phase does not change measurement probabilities, while relative phase can change the result of a later gate.

An $n$-qubit pure state has $2^n$ complex amplitudes in a classical state-vector representation. This does not let a user read all those amplitudes from one measurement. Measurement returns an outcome with $n$ classical bits. A quantum algorithm must arrange interference so that the measurement distribution reveals useful information. Calling the state a collection of simultaneously readable classical answers is misleading.

A quantum gate acts through a unitary transformation in the ideal circuit model. The X gate exchanges the 0 and 1 amplitudes. The Hadamard gate maps $|0\rangle$ to an equal superposition and maps relative phase into different outcomes when applied again. A controlled-NOT flips its target conditional on its control. Applying H to one qubit followed by controlled-NOT can create an entangled Bell state. Entanglement means the joint state cannot be factored into independent single-qubit states.

## Deutsch-Jozsa relies on a promised oracle

The Deutsch-Jozsa problem assumes a Boolean function is promised to be either constant or balanced. Balanced means it returns one on exactly half its inputs. The algorithm does not solve arbitrary function classification outside that promise. A phase-oracle form maps $|x\rangle$ to $(-1)^{f(x)}|x\rangle$. Hadamard gates prepare a uniform superposition before the oracle and transform it again afterward.

The amplitude of the all-zero output after the final Hadamards is proportional to $\sum_x(-1)^{f(x)}$. A constant function makes all terms have the same sign, while a balanced function makes them cancel. In the ideal promised setting, measurement distinguishes these cases with one oracle query. The query model treats the oracle as available. Building the oracle and implementing error correction are separate costs, so this result does not prove an exponential end-to-end speedup for arbitrary real applications.

The experiment constructs the ideal state vector for two input qubits and evaluates a constant function and a balanced parity function. It also creates a Bell state and prints its probabilities. This is classical simulation of four amplitudes. Its cost grows exponentially with the number of simulated qubits, and it does not demonstrate quantum hardware performance. The value of the exercise is to expose normalization, phase, and interference with numbers you can inspect.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
import numpy as np
H=np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
H2=np.kron(H,H)
start=np.array([1,0,0,0],dtype=complex)
results={}
for name,phases in [('constant',[1,1,1,1]),('balanced',[1,-1,-1,1])]:
 state=H2@(np.array(phases)*(H2@start))
 probs=np.abs(state)**2
 assert np.isclose(probs.sum(),1)
 results[name]=np.round(probs,12).tolist()
bell=np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
assert np.isclose(results['constant'][0],1) and np.isclose(results['balanced'][0],0)
print(json.dumps({'deutsch_jozsa_probabilities':results,
                  'bell_probabilities':(np.abs(bell)**2).tolist()},indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "deutsch_jozsa_probabilities": {
    "constant": [
      1.0,
      0.0,
      0.0,
      0.0
    ],
    "balanced": [
      0.0,
      0.0,
      0.0,
      1.0
    ]
  },
  "bell_probabilities": [
    0.4999999999999999,
    0.0,
    0.0,
    0.4999999999999999
  ]
}
```

The constant oracle gives all-zero output probability **1**, while the balanced parity oracle gives probability **0** for that output. The Bell state gives equal probability to 00 and 11. These results follow from ideal unitary arithmetic in a tiny classical simulator. They are not observations from a quantum processor.

:::{admonition} Try it
:class: exercise
You can replace the balanced oracle by another balanced truth table. Verify that the all-zero probability stays zero even when the other output probabilities change.
:::

:::{admonition} Try it
:class: exercise
You can draw the dependency chain for a GPU-produced send buffer, MPI exchange, and GPU consumer. Mark which event or completion check protects each transition.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Hybrid execution | Each model needs explicit ownership and completion boundaries. |
| Quantum state | Amplitudes determine probabilities through interference and measurement. |
| Quantum claim | A query-model advantage does not establish an arbitrary application speedup. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "May a funnelled MPI program call MPI from any worker thread?",
    "options": [
      "No, only the initializing thread may call MPI.",
      "Yes."
    ],
    "answer": 0,
    "explain": "MPI_THREAD_FUNNELED has a specific calling-thread restriction."
  },
  {
    "q": "Does requesting MPI_THREAD_MULTIPLE guarantee it was provided?",
    "options": [
      "No, the returned level must be checked.",
      "Yes."
    ],
    "answer": 0,
    "explain": "The implementation reports the available support."
  },
  {
    "q": "Can one measurement reveal all 2^n amplitudes?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Measurement yields an outcome rather than a complete amplitude table."
  },
  {
    "q": "What promise does Deutsch-Jozsa require?",
    "options": [
      "The function is constant or balanced.",
      "The function is any real-valued program."
    ],
    "answer": 0,
    "explain": "The distinction depends on the promised Boolean-function class."
  },
  {
    "q": "What creates the balanced-case zero amplitude?",
    "options": [
      "Cancellation of positive and negative phase contributions.",
      "Reading every branch as a classical answer."
    ],
    "answer": 0,
    "explain": "Interference produces the useful measurement pattern."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text. The [MPI standard documents](https://www.mpi-forum.org/docs/) and [OpenMP specifications](https://www.openmp.org/specifications/) define the relevant API contracts. Saad’s sets 18 and 19 provide the hybrid-programming and quantum-computing source topics.
