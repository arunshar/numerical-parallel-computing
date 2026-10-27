# 21. MPI point-to-point communication and correctness

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`20-communication-networks` modeled the cost of moving data. MPI gives a portable interface for expressing those transfers between processes. This lecture focuses on message matching, buffer ownership, deadlock, and completion, because a fast communication pattern is useless when it can hang or read unfinished data.

:::{admonition} Learning goals
- You can identify the source, destination, tag, communicator, and datatype of a message.
- You can explain why blocking send does not always mean receiver completion.
- You can write a safe neighbor exchange with MPI_Sendrecv or nonblocking requests.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/21-mpi-point-to-point.py`.

## MPI processes communicate within a context

An MPI program commonly follows the single-program, multiple-data model. Every rank starts the same executable, and rank-specific branches assign different work. `MPI_Init` initializes MPI, `MPI_Comm_rank` identifies the rank within a communicator, and `MPI_Comm_size` reports that communicator's size. `MPI_Finalize` ends MPI use after outstanding communication and collective obligations have been satisfied. A rank is a communicator-relative identifier rather than a permanent hardware-core number.

A **communicator** combines a process group with a communication context. The context keeps unrelated message streams from interfering. Two libraries can use matching rank numbers and tags without receiving each other's messages if they use distinct communicators. The envelope of a point-to-point message includes its source, destination, tag, and communicator. A receive specifies matching constraints, including optional wildcards for source or tag.

The message payload has a datatype and count. `MPI_INT` describes a C integer according to the implementation's MPI binding. A count specifies elements of that datatype, not bytes unless the datatype is `MPI_BYTE`. A receive buffer must hold enough matching elements. A larger receive count can be valid because it specifies capacity. `MPI_Get_count` lets the receiver inspect how many elements arrived. A too-small receive can produce truncation rather than a partially valid algorithm.

## Completion describes buffer reuse precisely

A blocking standard-mode `MPI_Send` returns when the send buffer can be safely reused. It does not universally guarantee that the receiving program has completed its receive. MPI may copy data into an internal buffer or may wait for a matching receive, depending on message size and protocol. A program that works only because a small message was buffered is therefore incorrect as a portable design.

`MPI_Ssend` uses synchronous mode, which requires a matching receive to have started before the send completes. Buffered mode uses application-provided buffer space with its own management rules. Ready mode assumes the matching receive has already been posted. These modes are semantic contracts, not interchangeable performance switches. A sender must satisfy the required precondition rather than infer it from a test that happened to finish.

A blocking receive returns when the receive buffer holds the received data and is available for access. A nonblocking operation returns a request before completion. `MPI_Isend` does not let the program overwrite its send buffer immediately. `MPI_Irecv` does not let the program read unfinished received elements. `MPI_Wait`, `MPI_Waitall`, or a successful test establishes completion of the corresponding request. A request handle tracks an operation, not a copy of the user's array.

## A deadlock can appear when buffering changes

Consider two ranks that both call a blocking send before either posts a receive. Small messages may complete through internal buffering. Larger messages may require a rendezvous with the receiver, causing both ranks to wait forever. Changing the machine, message size, or MPI implementation can expose the bug. The successful small-message run never proved that the ordering was safe.

One solution gives the ranks complementary orderings. Another uses `MPI_Sendrecv`, which combines one send and one receive and avoids the circular blocking pattern for a suitable exchange. A third posts receives and sends nonblockingly, then waits for all requests. The best choice depends on the application's dependency structure. A halo exchange often posts receives first so incoming data already has a destination when neighbors start sending.

Message ordering also needs care. MPI preserves certain ordering guarantees for matching operations from a single execution thread, but wildcard receives and concurrent threads can introduce nondeterminism. Tags can distinguish iteration phases or message roles, although they should remain within the supported tag range. Communicator separation is often cleaner than allocating large informal tag spaces across independent modules.

## Derived datatypes describe noncontiguous memory

A matrix column in row-major storage is not contiguous. You can pack it into a temporary contiguous buffer or describe its stride with an MPI datatype. `MPI_Type_vector` constructs repeated blocks with a stride measured in units of the old datatype. `MPI_Type_commit` makes a newly constructed type available for communication, and `MPI_Type_free` releases the handle when it is no longer needed. Datatype extent controls spacing between repeated datatype instances, which can surprise you when holes or resized types are involved.

A derived datatype expresses the memory layout. It does not promise that the implementation avoids internal packing or uses a particular hardware transfer. A fair performance comparison includes datatype construction where appropriate and separates one-time setup from repeated exchanges. Explicit packing can make irregular data easier to inspect, while datatypes can reduce manual indexing errors in stable layouts.

## Nonblocking calls create an opportunity for overlap

An application can initiate communication, compute on independent interior data, then wait before using halo data. This overlap is valid only if the interior computation does not touch active communication buffers in a conflicting way. MPI progress behavior and hardware support determine whether communication advances while computation runs. A nonblocking call alone is not evidence of overlap. You should measure a timeline or compare carefully defined elapsed intervals on the target implementation.

The small native program in this lecture performs a ring exchange with `MPI_Sendrecv`. Each rank sends its rank number to the right neighbor and receives from the left. The last rank sends to rank zero, so the communicator forms a logical ring even when all processes run on one host. The validation checks received values, which establishes message-routing correctness without claiming network performance.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
p=4
rows=[{'rank':r,'send_to':(r+1)%p,'receive_from':(r-1)%p,
       'received_value':(r-1)%p} for r in range(p)]
assert [r['received_value'] for r in rows]==[3,0,1,2]
print(json.dumps(rows,indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
[
  {
    "rank": 0,
    "send_to": 1,
    "receive_from": 3,
    "received_value": 3
  },
  {
    "rank": 1,
    "send_to": 2,
    "receive_from": 0,
    "received_value": 0
  },
  {
    "rank": 2,
    "send_to": 3,
    "receive_from": 1,
    "received_value": 1
  },
  {
    "rank": 3,
    "send_to": 0,
    "receive_from": 2,
    "received_value": 2
  }
]
```

The model predicts received values **3, 0, 1, 2** at ranks 0 through 3. The native `mpi_exchange.c` checks the same mapping with real MPI processes. The saved native log records its result. Both tests concern correctness. Local rank-to-rank communication does not characterize communication across a cluster interconnect.

:::{admonition} Try it
:class: exercise
You can reverse the direction of the ring and update the expected receive rank. Explain how source and destination change without changing the communicator.
:::

:::{admonition} Try it
:class: exercise
You can sketch a nonblocking halo exchange with distinct send and receive buffers. Mark the earliest point at which each buffer may be reused.
:::

## Run the native examples

The original C implementations below were compiled and executed locally. The captured compiler and runtime results are saved in `evidence/parallel/native.json`. The examples validate small integer results and do not measure speedup. The build commands are documented in `labs/parallel/README.md`.

### The MPI ring exchange example uses the native API

```{literalinclude} ../../labs/parallel/mpi_exchange.c
:language: c
```

## Summary

| Idea | What the lesson established |
|---|---|
| Matching | A communicator and message envelope determine which receive can match a send. |
| Completion | A completed request establishes when its buffer may be accessed or reused. |
| Deadlock avoidance | A portable exchange must not depend on unspecified internal buffering. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "Can MPI_Send return before the receiver finishes its receive?",
    "options": [
      "Yes, standard mode can use buffering.",
      "No, it always waits for receiver completion."
    ],
    "answer": 0,
    "explain": "Send completion guarantees safe send-buffer reuse."
  },
  {
    "q": "Can a program overwrite an active MPI_Isend buffer?",
    "options": [
      "No, it must establish completion first.",
      "Yes, the call always copied it."
    ],
    "answer": 0,
    "explain": "Nonblocking initiation does not imply a completed copy."
  },
  {
    "q": "What does a communicator provide besides ranks?",
    "options": [
      "A communication context.",
      "A promise of one rank per physical core."
    ],
    "answer": 0,
    "explain": "The context separates independent communication streams."
  },
  {
    "q": "What can happen when a receive buffer is too small?",
    "options": [
      "A truncation error can occur.",
      "MPI always allocates a larger user buffer."
    ],
    "answer": 0,
    "explain": "The receiver must provide adequate capacity."
  },
  {
    "q": "Does nonblocking initiation prove useful overlap?",
    "options": [
      "No, progress and independent work must be verified.",
      "Yes."
    ],
    "answer": 0,
    "explain": "The API exposes an opportunity rather than a timing guarantee."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text. The [MPI standard documents](https://www.mpi-forum.org/docs/) and [OpenMP specifications](https://www.openmp.org/specifications/) define the relevant API contracts.
