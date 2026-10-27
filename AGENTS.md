# Instructions for future editors

Read CLAUDE_HANDOFF.md, docs/reference/coverage.md, and docs/reference/validation.md before changing claims. The study lessons explain public course topics and textbook outlines. Full textbook text was not available during the initial build.

Write complete subject-first sentences in plain language. Do not use em dashes. Preserve technical terms, equations, and exact claims when editing. Write new examples instead of copying source slides or textbook paragraphs.

Edit docs/lectures and labs as the primary sources. Rebuild dist/STUDY_GUIDE.md after changes. Keep coverage entries aligned with actual lesson depth. A short survey must remain marked as a survey.

Run python scripts/verify_course.py and python -m sphinx -W --keep-going -b html docs docs/_build/html before proposing changes. Record native MPI, OpenMP, and CUDA validation separately. Never describe a CPU model as a GPU benchmark.

Create a branch and a draft pull request for changes after initial publication. Do not merge without Arun's instruction. Do not add the previously downloaded instructor PDFs, textbook files, homework ZIPs, credentials, or private local paths.
