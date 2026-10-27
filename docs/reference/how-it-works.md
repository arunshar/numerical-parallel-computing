# How the study repository is built

The repository keeps authored explanations, teaching code, and validation evidence together. The documentation build turns Markdown and notebooks into a navigable website. A separate builder creates one continuous reading document.

## The source inventory defines the scope.

The inventory records the public CSCI 5302 and CSCI 5451 material inspected for this edition. Official textbook tables of contents provide a second checklist. The coverage matrix maps these topics to lessons and marks compressed surveys.

The four books were not read in full through an authenticated library session. The repository therefore makes a topic-coverage claim based on public slides, public chapter outlines, and primary technical references. It does not claim that every exercise, proof, or paragraph in the books has been reproduced.

## The examples test small claims.

The numerical examples compare computed results with an exact value, a residual, or an independently computed reference. The parallel models compare operation counts or communication estimates under stated assumptions. Native examples test program behavior only on the recorded local configuration.

A GPU indexing model can expose a missing boundary check. It cannot measure memory coalescing, occupancy, or instruction throughput on a physical GPU. A CPU implementation of a CUDA algorithm checks its arithmetic and indexing at a different execution layer.

## The build separates success states.

A syntax check confirms that a parser accepts a file. A numerical assertion checks a specified property on the tested input. A documentation build confirms that Sphinx can resolve the authored pages. A GitHub push confirms remote source state. A deployed website requires a successful deployment and a live page check.

The validation record reports these states separately. Future editors should preserve that separation when adding results or changing examples.
