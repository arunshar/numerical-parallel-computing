# Validation and evidence

The final build records its checks in `evidence/validation.json`. Each portable lab entry contains the source path, source hash, return code, and captured output. The track-specific evidence directories preserve additional numerical and native-runtime checks.

## The recorded checks cover these artifacts.

| Check | Recorded result | Scope |
|---|---|---|
| The structure and quiz checks inspect 48 lessons. | All 240 quiz questions parse, and the required lesson sections are present. | These checks establish structure, not proof correctness. |
| The portable execution check runs 46 numbered Python labs. | All labs exit successfully with their assertions enabled. | The experiments use small synthetic inputs on the recorded CPU environment. |
| The integrated capstone executes as a notebook. | Its numerical and decomposition assertions pass. | Its communication costs are invented model inputs. |
| The native checks compile and run four C programs. | pthreads, OpenMP, an MPI ring, and MPI collectives pass. | The run uses at most four local threads or ranks and records no speed measurements. |
| The documentation preflight uses a fresh Python 3.12 environment. | All 59 source pages pass the strict build and page checks. | The preflight checks canonical links, quizzes, and notebook launch URLs. |
| The external reference check requests the linked source pages. | Two SIAM book pages return HTTP 403 to the automated checker. | These references remain listed as access-limited. The check does not establish full-book access. |
| The CUDA suite has source-level review. | Native compilation and execution remain UNVERIFIED. | The machine has no compatible NVIDIA CUDA runtime. |

The preflight transcript appears in `evidence/documentation-preflight.txt`. The external link report appears in `evidence/external-links.json`.

## The portable checks have a defined scope.

The Python checks execute small teaching cases. They test the assertions stated by each lab. The capstone checks grid-refinement behavior, agreement between two update paths, and a cost model with invented parameters. It does not measure MPI or CUDA performance.

The documentation build uses Sphinx with `-W`, which treats warnings as errors. The structure check verifies lesson sections, quiz schemas, and unresolved placeholders. The combined document is generated from the same authored lessons.

## Native execution needs separate records.

The native examples require their stated compiler and runtime. Successful CPU reference checks do not establish native execution. CUDA compilation and execution remain UNVERIFIED in this edition because a local CUDA compiler and compatible NVIDIA runtime were not available.

The relevant track evidence reports whether pthread, OpenMP, and MPI examples compiled and ran. Those results apply to the recorded local configuration and small test inputs. They do not establish performance on a university cluster.

## Publication requires remote verification.

A local build does not prove that a public website is live. The final publication report separately checks the pushed commit, GitHub Actions status, and the live site. The Read the Docs configuration is provided, but the service requires an account import before its hosted site can be claimed.

## Visual and substantive review have limits.

The initial review checks equations, output agreement, representative content, and navigation. It does not certify every theorem or replace expert review of all 48 lessons. The coverage ledger identifies the advanced survey sections that should receive another technical pass.
