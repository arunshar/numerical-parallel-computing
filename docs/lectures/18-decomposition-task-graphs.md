# 18. Decomposition, dependency graphs, and load balance

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`17-architecture-cache` explained the hardware resources that can execute work together. This lecture asks whether an algorithm exposes enough independent work to use those resources. We represent an algorithm as tasks and dependencies, then decide how to group and place those tasks.

:::{admonition} Learning goals
- You can compute work and span for a dependency graph.
- You can compare data, recursive, exploratory, and speculative decomposition.
- You can explain the tradeoff between load balance and locality.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/18-decomposition-task-graphs.py`.

## A dependency graph states what must happen first

A **task** is a unit of computation that the scheduler can assign. A directed edge from task A to task B means that B needs something A produces. A valid schedule respects every edge. Task size is a design choice. One multiplication per task may expose enormous theoretical parallelism but require more scheduling work than useful computation. One entire application per task hides almost all internal parallelism.

**Work**, written $T_1$, is the total execution cost of all tasks on one processor under the chosen cost model. **Span**, written $T_\infty$, is the cost of the longest dependency path. An execution on $p$ identical processors must take at least $\max(T_1/p,T_\infty)$. The ratio $T_1/T_\infty$ is average parallelism. This bound assumes the task costs are comparable across processors and ignores communication unless communication appears explicitly in the graph.

Consider two independent tasks costing 4 and 2 units. Each feeds a separate 3-unit task, and a final 1-unit task combines their results. Total work is 13 units. The longer path costs $4+3+1=8$ units, so unlimited processors cannot finish in fewer than eight units. Two processors can achieve eight units for this graph. Four processors cannot improve the critical path merely by waiting beside it.

## Decomposition follows the problem's structure

**Data decomposition** splits an input, output, or intermediate array. A row decomposition of matrix multiplication assigns output rows to tasks. Output decomposition often avoids conflicting writes because each task owns its result region. Input decomposition can be useful when input blocks are naturally distributed, but it may require a reduction when several tasks contribute to the same output. Intermediate-data decomposition creates reusable partial results and must account for their storage and movement.

**Recursive decomposition** splits a problem into smaller instances until each leaf is large enough to compute efficiently. Merge sort and divide-and-conquer matrix algorithms provide familiar examples. The recursion creates a tree of tasks, but the merge stage can become the bottleneck. A task scheduler benefits from a cutoff that switches small subproblems to a sequential implementation. The cutoff depends on task overhead, cache behavior, and the operation inside each leaf.

**Exploratory decomposition** gives workers different branches of a search space. The amount of work is often unknown until a branch is explored. **Speculative decomposition** computes alternatives before knowing which result will be needed. Speculation trades extra work for a shorter dependency chain. It is useful only when the saved waiting time exceeds the cost of discarded computation. Search, branch prediction, and optimistic execution use related ideas under different correctness rules.

## Task interactions affect the placement policy

Two tasks may exchange data in a regular nearest-neighbor pattern, a global collective, or an irregular graph. Interaction may be static and known before execution, or dynamic and discovered from the input. The amount of data and the frequency of interaction both matter. Sending a tiny message a million times can cost more than sending the same total bytes in a few batches because each message has startup overhead.

**Static scheduling** assigns work before execution. Block scheduling assigns contiguous ranges, which often preserves locality. Cyclic scheduling spreads nearby iteration indices across workers, which can balance a predictable cost trend. Block-cyclic scheduling uses small contiguous blocks and distributes those blocks cyclically. A block size of one is pure cyclic distribution, while one block per worker approaches block distribution. This family gives a useful knob between locality and balance.

**Dynamic scheduling** assigns new work to idle workers. A central queue is simple but may become a contention point. Distributed queues and work stealing reduce central pressure, although stolen tasks may move away from their data. Work stealing commonly removes a large pending subtree so the stealing overhead supports substantial computation. The correct policy depends on whether uncertainty in task cost is more expensive than queue access and lost locality.

## An imbalance model makes the tradeoff concrete

Suppose task durations are 1, 2, 3, 4, 5, 6, 7, 8 units and four workers are available. A contiguous assignment of two tasks per worker produces loads 3, 7, 11, 15. The makespan is 15. A cyclic assignment produces loads 6, 8, 10, 12, so its makespan is 12. A greedy assignment that sorts tasks from largest to smallest and repeatedly uses the least-loaded worker achieves 9 in this case. Total work is 36, so 9 is the work lower bound.

The greedy result is a property of this small list, not proof that greedy scheduling is always optimal. It also assumes every task is ready at time zero and moving a task has no communication cost. A graph with dependencies can prevent the largest task from being ready. A matrix partition can make a balanced assignment slower if it duplicates data transfers. A realistic placement objective must include computation, communication, and memory capacity.

The **owner-computes rule** assigns an operation to the worker that owns its output or dominant input. It can reduce transfers, but it does not automatically balance load. Graph partitioning addresses a related objective by balancing vertex or edge work while reducing communication across partitions. Later lectures apply that idea to sparse matrices. The design process starts with a dependency graph, estimates task costs, records interactions, and only then chooses a mapping.


## Asymptotic notation compares growth rates

The textbook appendix uses big-O, big-Omega, and big-Theta to separate algorithm growth from one machine's clock rate. The statement $f(n)=O(g(n))$ means that fixed positive constants eventually bound $f$ above by a multiple of $g$. Big-Omega provides a corresponding eventual lower bound, and big-Theta means both bounds hold. These definitions concern sufficiently large inputs. They do not say that the hidden constant is small or that the asymptotically better method wins at every practical size.

For example, $3n^2+20n+7$ is $\Theta(n^2)$. It is also $O(n^3)$, but that upper bound is less informative. A logarithm's base changes a constant factor when the bases are fixed, so order notation commonly omits the base. The number of processors is another independent variable in a parallel algorithm. A claim such as $T_p(n,p)=O(n/p+\log p)$ must state the valid range of $p$ and account for available concurrency. Substituting more processors than useful tasks does not create fractional indivisible work.

Work and span describe two different functions of input size. A balanced reduction has linear work and logarithmic span. A sequential recurrence can have linear work and linear span. Both algorithms can therefore have the same sequential big-Theta complexity while offering very different parallel speedup. This is why a parallel analysis reports total work, critical path, and communication separately.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json
costs = list(range(1, 9))
p = 4
block = [sum(costs[2*i:2*i+2]) for i in range(p)]
cyclic = [sum(costs[i::p]) for i in range(p)]
greedy = [0] * p
for cost in sorted(costs, reverse=True):
    i = min(range(p), key=lambda k: greedy[k])
    greedy[i] += cost
out = {name: {'loads': loads, 'makespan': max(loads)}
       for name, loads in [('block', block), ('cyclic', cyclic), ('greedy', greedy)]}
assert max(block) == 15 and max(cyclic) == 12 and max(greedy) == 9
print(json.dumps(out, indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
{
  "block": {
    "loads": [
      3,
      7,
      11,
      15
    ],
    "makespan": 15
  },
  "cyclic": {
    "loads": [
      6,
      8,
      10,
      12
    ],
    "makespan": 12
  },
  "greedy": {
    "loads": [
      9,
      9,
      9,
      9
    ],
    "makespan": 9
  }
}
```

The modeled makespans are **15, 12,and 9 cost units** for block,cyclic,and greedy assignment. The best schedule matches the work lower bound for this particular independent-task list. The experiment does not execute tasks concurrently. It verifies the arithmetic of an assignment model, which you can later compare with a real runtime.

:::{admonition} Try it
:class: exercise
You can add a two-unit migration cost to each task assigned away from its initial block. Determine whether the greedy assignment still wins after that transfer cost.
:::

:::{admonition} Try it
:class: exercise
You can change the costs to eight identical values, then explain why a dynamic scheduler has little useful imbalance to fix. Consider its queue overhead separately.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Work and span | Work limits total capacity, and span limits dependency speedup. |
| Static mapping | A predictable workload can benefit from cheap scheduling and good locality. |
| Dynamic mapping | An irregular workload can benefit from reassignment despite scheduling costs. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is the lower bound on runtime for work 36 and four processors?",
    "options": [
      "9 cost units.",
      "4 cost units."
    ],
    "answer": 0,
    "explain": "The work bound is 36 divided by 4."
  },
  {
    "q": "Can more processors shorten a fixed dependency chain?",
    "options": [
      "No, the chain still sets the span.",
      "Yes, every additional processor halves it."
    ],
    "answer": 0,
    "explain": "Dependent tasks must preserve their order."
  },
  {
    "q": "Which decomposition explores separate search branches?",
    "options": [
      "Exploratory decomposition.",
      "Only output-array decomposition."
    ],
    "answer": 0,
    "explain": "Search branches are tasks whose sizes may not be known in advance."
  },
  {
    "q": "Does perfect load balance guarantee low runtime?",
    "options": [
      "No, communication can dominate.",
      "Yes, placement never matters."
    ],
    "answer": 0,
    "explain": "Balance is only one part of the cost."
  },
  {
    "q": "What does a task cutoff avoid?",
    "options": [
      "Excessive overhead on tiny tasks.",
      "All numerical error."
    ],
    "answer": 0,
    "explain": "Small recursive tasks may cost more to schedule than to execute."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
