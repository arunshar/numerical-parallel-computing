# 23. Speedup, efficiency, scaling, and isoefficiency

Prepared for Arun Sharma. Codex drafted this study edition.

{doc}`22-mpi-collectives-topology` showed how group operations organize communication. We now ask when the parallel algorithm is worth using. A useful performance claim states the baseline, problem size, resource count, data movement, and measured interval instead of reporting one unexplained speedup number.

:::{admonition} Learning goals
- You can calculate speedup, efficiency, cost, and overhead.
- You can distinguish strong scaling, weak scaling, and scaled speedup.
- You can derive a simple isoefficiency condition and identify its assumptions.
:::

## Setup

You can use the Python environment described in {doc}`../getting-started/run-locally`. The experiment below uses only Python and NumPy, runs on a CPU, and uses small deterministic inputs. You can run the complete file from the repository root with `python labs/parallel/23-performance-models.py`.

## A speedup needs a well-defined baseline

Let $T_s(n)$ be the runtime of a chosen sequential algorithm on a specified machine and $T_p(n,p)$ the runtime of a parallel implementation using $p$ processors. **Speedup** is $S=T_s/T_p$, and **efficiency** is $E=S/p$. A speedup of 6 on 8 processors gives efficiency 0.75. This does not mean that each processor is idle exactly 25% of the time, because different forms of overhead and baseline differences can produce the same ratio.

The **parallel cost** is $pT_p$. An implementation is cost-optimal in the asymptotic sense when this cost has the same order as the best relevant sequential work. A method can run quickly while wasting enough resources to be poor for a throughput-limited service. Conversely, a modest speedup can still be valuable when the sequential job misses a deadline. The objective determines whether latency, total core-hours, memory capacity, or energy matters most.

Define overhead as $T_o=pT_p-T_s$. It can include communication, redundant work, waiting, scheduling, and other costs. This definition compares entire runtimes, so it can become negative when the parallel execution improves cache behavior relative to the baseline. Such a result needs explanation rather than an automatic claim that all standard limits were defeated. A different baseline algorithm or a different memory regime may be responsible.

## Amdahl's law holds the workload fixed

Suppose a fraction $f$ of sequential time is inherently serial, while the remaining fraction scales ideally. The model gives

$$S(p)=\frac{1}{f+(1-f)/p}.$$

For $f=0.1$ and $p=4$, speedup is about 3.077. As $p$ increases without bound, the limit is 10. This is a model for a fixed workload with an unchanged serial fraction. Communication overhead would lower the prediction further. Optimizing the serial phase changes $f$ and therefore changes the bound.

The serial fraction must be defined from the baseline workload. A measured parallel execution can spend a large fraction of its wall time in a short serial phase because the parallel phase has become faster. Substituting that observed parallel fraction directly into the fixed-work Amdahl formula mixes two denominators. Derivations should write down which runtime each fraction uses before manipulating it.

## Gustafson's model changes the workload

A scaled workload can grow as resources increase. If a parallel execution spends fraction $\alpha$ of its time in a serial part and fraction $1-\alpha$ using $p$ processors, a scaled-speedup model is $S_G=p-\alpha(p-1)$. With $\alpha=0.1$ and $p=4$, this gives 3.7. This does not contradict the Amdahl value 3.077 because the experiments hold different things fixed.

**Strong scaling** fixes the global problem size and adds resources. Each processor gets less useful work, while communication and synchronization eventually dominate. **Weak scaling** keeps some measure of work per processor fixed while growing the global problem. Ideal weak-scaling time is constant under that definition. A two-dimensional domain can grow in area while each partition's boundary-to-interior ratio changes, so the exact scaling rule must be stated.

Memory capacity is another reason to scale. A job may become possible only when aggregate memory grows. Comparing its parallel runtime with a hypothetical sequential run that cannot fit in memory requires care. You can report throughput, time-to-solution, or a model-based baseline, but you should label a modeled baseline clearly. A speedup computed from a measurement that never occurred is not an empirical speedup.

## Isoefficiency asks how fast the problem must grow

From the overhead definition, efficiency is $E=T_s/(T_s+T_o)$. Holding a target efficiency fixed requires sequential work to grow proportionally to overhead. This relation defines an **isoefficiency function** when the overhead can be expressed in terms of problem size and processor count. The result describes scalability of an algorithm-machine model, not a universal hardware score.

Consider a model with work $W$ and runtime $T_p=W/p+\log_2p$ in normalized units. Its overhead is $p\log_2p$. To maintain efficiency 0.8, the relation $W/(W+p\log_2p)=0.8$ gives $W=4p\log_2p$. A workload that grows only proportionally to $p$ does not maintain the same efficiency in this model. Concurrency limits and memory capacity can impose additional growth requirements, so the largest relevant requirement controls the useful regime.

The target efficiency 0.8 is inapplicable when $p=1$. The overhead is zero, so every positive workload has ideal efficiency 1. The output uses `null` for this case because no positive work attains exactly 0.8.

A work-span bound adds another limit. If the critical path grows almost as quickly as total work, adding processors cannot create enough independent tasks. A communication-volume model may reveal a different limit even when the graph has abundant parallelism. A complete analysis checks work, span, communication, and memory together.

## A measurement protocol makes the claim reproducible

A trustworthy benchmark records input dimensions, data type, compiler flags, library versions, process and thread counts, affinity, warmup, repetition count, and the timed region. The result should include variability. MPI jobs often report the maximum rank time for a phase because the slowest rank determines completion. Summing rank times answers a different resource-use question. A barrier before timing can align a start, but it also changes the experiment and should be documented.

A performance model is useful before timing because it predicts which curve should bend and why. After timing, a mismatch is evidence about an assumption. It may reveal memory bandwidth saturation, a hidden serial section, contention, or a weak baseline. It is not a reason to silently redefine the model until the graph looks right. The experiment below uses exact formulas and makes no claim about measured machine speed.

## Run the experiment

The complete experiment is shown below so that you can inspect every formula and indexing choice before running it.

```python
import json, math
rows=[]
for p in (1,2,4,8,16):
    amdahl=1/(0.1+0.9/p)
    gustafson=p-0.1*(p-1)
    required=4*p*math.log2(p) if p>1 else None
    rows.append({'processors':p,'amdahl_speedup':round(amdahl,4),
                 'gustafson_scaled_speedup':round(gustafson,4),
                 'work_for_efficiency_0_8':required})
assert rows[0]['work_for_efficiency_0_8'] is None
assert abs(rows[2]['amdahl_speedup']-3.0769)<1e-4
for row in rows[1:]:
    p=row['processors']
    work=row['work_for_efficiency_0_8']
    efficiency=work/(work+p*math.log2(p))
    assert abs(efficiency-0.8)<1e-12
print(json.dumps(rows,indent=2))
```

The local verification produced this output. Tiny floating-point residuals can vary across numerical-library builds while remaining within the stated tolerance.

```text
[
  {
    "processors": 1,
    "amdahl_speedup": 1.0,
    "gustafson_scaled_speedup": 1.0,
    "work_for_efficiency_0_8": null
  },
  {
    "processors": 2,
    "amdahl_speedup": 1.8182,
    "gustafson_scaled_speedup": 1.9,
    "work_for_efficiency_0_8": 8.0
  },
  {
    "processors": 4,
    "amdahl_speedup": 3.0769,
    "gustafson_scaled_speedup": 3.7,
    "work_for_efficiency_0_8": 32.0
  },
  {
    "processors": 8,
    "amdahl_speedup": 4.7059,
    "gustafson_scaled_speedup": 7.3,
    "work_for_efficiency_0_8": 96.0
  },
  {
    "processors": 16,
    "amdahl_speedup": 6.4,
    "gustafson_scaled_speedup": 14.5,
    "work_for_efficiency_0_8": 256.0
  }
]
```

The four-processor row gives **3.0769 fixed-work speedup** and **3.7 scaled speedup** under different assumptions. The table is a calculation, not a benchmark. For the isoefficiency example, the required work at four processors is 32 normalized units, which gives 0.8 efficiency in the stated runtime model.

:::{admonition} Try it
:class: exercise
You can reduce the serial fraction from 0.1 to 0.01. Calculate the asymptotic limit and compare the benefit with doubling processors from 8 to 16.
:::

:::{admonition} Try it
:class: exercise
You can add a communication term proportional to the square root of the work. Derive the new condition for fixed efficiency instead of guessing it from a plot.
:::

## Summary

| Idea | What the lesson established |
|---|---|
| Strong scaling | The global problem stays fixed while resources increase. |
| Weak scaling | The chosen work-per-resource quantity stays fixed while the global problem grows. |
| Isoefficiency | The workload must grow fast enough to keep overhead from dominating. |

## Check your understanding

Pick an answer to see whether it is right. After a wrong answer, you can try another option.

```{quiz}
[
  {
    "q": "What is efficiency for speedup 6 on 8 processors?",
    "options": [
      "0.75.",
      "1.33."
    ],
    "answer": 0,
    "explain": "Efficiency divides speedup by processor count."
  },
  {
    "q": "What is the Amdahl limit for serial fraction 0.1?",
    "options": [
      "10.",
      "100."
    ],
    "answer": 0,
    "explain": "The ideal asymptotic limit is 1 divided by the serial fraction."
  },
  {
    "q": "Do Amdahl and Gustafson hold the same workload fixed?",
    "options": [
      "No.",
      "Yes."
    ],
    "answer": 0,
    "explain": "Their different assumptions explain their different results."
  },
  {
    "q": "Which rank duration determines a synchronized phase's completion?",
    "options": [
      "The maximum relevant rank duration.",
      "The minimum rank duration."
    ],
    "answer": 0,
    "explain": "The phase cannot finish before its slowest required participant."
  },
  {
    "q": "Is a formula table a measured speedup result?",
    "options": [
      "No, it is a model calculation.",
      "Yes."
    ],
    "answer": 0,
    "explain": "An empirical claim requires saved measured runtime evidence."
  }
]
```

## Sources and scope

[Mooney’s public course schedule](https://jimtmooney.github.io/Courses/F25/index.html) and [Saad’s lecture index](https://www-users.cse.umn.edu/~saad/csci5451/notes.html) establish the course context. [The authors’ textbook contents](https://www.cs.purdue.edu/homes/ayg/book/toc.html) supplies the chapter roadmap. These notes are original explanations, and they do not claim access to the complete textbook text.
