"""Build navigation and a chapter-level coverage map from the authored lessons."""
from pathlib import Path
import json,re,csv
R=Path(__file__).resolve().parents[1]
lectures=sorted((R/'docs/lectures').glob('*.md'))+sorted((R/'docs/lectures').glob('*.ipynb'))
lectures.sort(key=lambda p:p.name)
paths={int(p.name.split('-')[0]):p for p in lectures}
heath=['Scientific Computing','Systems of Linear Equations','Linear Least Squares','Eigenvalue Problems','Nonlinear Equations','Optimization','Interpolation','Numerical Integration and Differentiation','Initial Value Problems for Ordinary Differential Equations','Boundary Value Problems for Ordinary Differential Equations','Partial Differential Equations','Fast Fourier Transform','Random Numbers and Simulation']
ascher=['Numerical Algorithms','Roundoff Errors','Nonlinear Equations in One Variable','Linear Algebra Background','Linear Systems: Direct Methods','Least Squares','Linear Systems: Iterative Methods','Eigenvalues and Singular Values','Nonlinear Systems and Optimization','Polynomial Interpolation','Piecewise Polynomial Interpolation','Best Approximation','Fourier Transform','Numerical Differentiation','Numerical Integration','Differential Equations']
grama=['Introduction','Parallel Programming Platforms','Principles of Parallel Algorithm Design','Basic Communication Operations','Analytical Modeling of Parallel Programs','Programming Using the Message-Passing Paradigm','Programming Shared Address Space Platforms','Dense Matrix Algorithms','Sorting','Graph Algorithms','Search Algorithms for Discrete Optimization Problems','Dynamic Programming','Fast Fourier Transform','Appendix A: Complexity of Functions and Order Analysis']
pmpp=['Introduction','Heterogeneous data parallel computing','Multidimensional grids and data','Compute architecture and scheduling','Memory architecture and data locality','Performance considerations','Convolution','Stencil','Parallel histogram','Reduction','Prefix sum (scan)','Merge','Sorting','Sparse matrix computation','Graph traversal','Deep learning','Iterative magnetic resonance imaging reconstruction','Electrostatic potential map','Parallel programming and computational thinking','Programming a heterogeneous computing cluster','CUDA dynamic parallelism','Advanced practices and future evolution','Conclusion and outlook','Appendix A: Numerical considerations']
sets=[('Heath',heath,[[1],[2,15],[3],[4],[5],[6,7],[8],[9],[10],[11],[12],[13],[14]],'https://heath.cs.illinois.edu/scicomp/contents/index.html'),('Ascher and Greif',ascher,[[1,16],[1],[5],[2,3,4],[2],[3],[15],[4],[5,6,7],[8],[8],[8,16],[13],[9],[9],[10,11,12]],'https://epubs.siam.org/doi/book/10.1137/9780898719987'),('Grama and colleagues',grama,[[17],[17],[18],[20,22],[23],[21,22],[19],[24],[25],[26],[27],[27],[28],[18,23]],'https://www.cs.purdue.edu/homes/ayg/book/toc.html'),('Hwu, Kirk, and El Hajj',pmpp,[[31],[31],[31],[32],[33],[34],[35],[36],[37],[38],[39],[40],[40],[41],[41],[42],[43],[43],[44,46],[44],[45],[45],[45],[46,1]],'https://www.educate.elsevier.com/book/details/9780323912310')]
rows=[]
for book,chapters,mapping,url in sets:
 for i,(title,numbers) in enumerate(zip(chapters,mapping),1):
  targets=[str(paths[n].relative_to(R)) for n in numbers if n in paths]
  rows.append({'book':book,'chapter':'A' if title.startswith('Appendix A:') else str(i),'chapter_title':title,'lessons':numbers,'paths':targets,'source_url':url,'scope':'Topic explanations and selected experiments; advanced mechanisms may be surveys.'})
(R/'coverage/book-chapters.json').write_text(json.dumps(rows,indent=2))
with (R/'coverage/book-chapters.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,lineterminator='\n',fieldnames=['book','chapter','chapter_title','lessons','paths','source_url','scope']);w.writeheader();w.writerows(rows)
intro='''# Course and textbook coverage

This ledger maps the public course topics and all chapter outlines from the four associated books to the study lessons. A mapped chapter means that the notes explain its subject and include selected examples. It does not mean that every proof, exercise, historical note, or software variant in the full text has been reproduced.

The full textbooks were not available during the initial build. Public slides, author and publisher chapter outlines, and primary API documentation supplied the verifiable scope. The track-specific JSON ledgers in `coverage/` distinguish explanations from compressed surveys.

## The textbook map covers the complete chapter outlines.

| Book | Chapter | Topic | Lessons |
|---|---|---|---|
'''
for row in rows:
 links=', '.join('{doc}`'+str(n)+' <../lectures/'+paths[n].stem+'>`' for n in row['lessons'] if n in paths)
 intro+=f'| {row["book"]} | {row["chapter"]} | {row["chapter_title"]} | {links} |\n'
intro+='''
## The course-specific material has its own map.

| Source unit | Study location | Coverage boundary |
|---|---|---|
| The CSCI 5302 syllabus covers errors, equations, interpolation, least squares, optimization, quadrature, and ODEs. | Lessons 1–10 explain the core sequence. | Lessons 11–16 extend the sequence to the remaining book topics. |
| The supplementary Taylor and bicubic notes explain local multivariate models and two-dimensional interpolation. | Lessons 8 and 16 explain those constructions. | The examples use new teaching data. |
| The supplementary QP/LP and SVM materials connect optimization to constrained models. | Lesson 7 explains the objectives, constraints, and optimality conditions. | The notes do not reproduce the instructor's handouts. |
| The MATLAB references compare array operations and programming conventions. | Lesson 16 explains the important MATLAB/Python differences. | The original reference sheets remain external sources. |
| Mooney's 26 lecture decks cover architectures, OpenMP, MPI, modeling, and CUDA. | Lessons 17–46 explain that sequence and its textbook extensions. | The final-lecture hardware and ML topics are clearly marked surveys. |
| HW1 uses shared-memory parallelism and dynamic time warping. | Lessons 19, 27, and 30 explain dependencies, implementation checks, and reporting. | The examples teach the methods without redistributing the assignment package. |
| HW2 and HW3 use distributed matrix multiplication and block-cyclic layouts. | Lessons 21, 22, 24, and 30 explain the ownership and communication. | The new examples have their own data and assertions. |
| HW4 uses CUDA convolution and profiling. | Lessons 34 and 35 explain memory reuse, boundaries, and performance evidence. | Native CUDA performance remains unverified locally. |
| HW5 uses pairwise batched matrix products. | Lesson 46 explains the algebra, numerical consequences, and validation choices. | Algebraic simplification does not establish permission under a particular assignment rubric. |
| Saad's 19 lecture sets add sorting, sparse and graph algorithms, hybrid programming, and quantum concepts. | Lessons 17–29 and 40–41 cover these subjects. | The quantum material is an introductory model. |
| The final project requires an implementation and evidence. | Lessons 30, 46, and 47 explain reproducibility, correctness, and performance reporting. | The capstone is an original teaching problem. |

## Some advanced subjects remain survey-level treatments.

A short section introduces a method's purpose, assumptions, main mechanism, and tradeoffs. It does not establish mastery of that method. The compressed areas include specialized eigensolver variants, finite-element and spectral implementation details, quantum algorithms, MRI reconstruction, dynamic parallelism, and evolving accelerator platforms. The handoff identifies these areas for expansion with authorized textbook access and hardware evidence.

The intended next step is to deepen those sections, not to change their status labels without adding explanations and checks. A future edition should also map textbook subsections and exercises when the actual editions become available.
'''
(R/'docs/reference/coverage.md').write_text(intro)
# Navigation follows the actual authored filenames.
def title(p):
 if p.suffix=='.md':return p.read_text().splitlines()[0].lstrip('# ')
 return ''.join(json.loads(p.read_text())['cells'][0]['source']).splitlines()[0].lstrip('# ')
content='''# Numerical and Parallel Computing

Prepared for Arun Sharma. Codex drafted this edition, and Claude refinement is pending.

**This course explains numerical algorithms and parallel computing through worked examples and runnable experiments.**

The study path brings together CSCI 5302 and CSCI 5451. The lessons use original explanations and code, with source links to the public course materials and four associated textbook outlines. You can read the {doc}`getting-started/index` page first or choose a lesson below.

You can also open the [complete interactive study reader](_static/STUDY_GUIDE.html). It combines every lesson into one file that you can save for offline study.

The {doc}`reference/coverage` ledger records coverage and depth. The {doc}`reference/validation` page separates verified CPU experiments from hardware-dependent examples. The {doc}`reference/claude-handoff` page gives the next editor a concrete continuation plan.

![The figure separates measured numerical error from a modeled parallel speedup.](_static/accuracy-and-scaling.svg)

The left panel checks a manufactured diffusion problem. The right panel uses invented communication costs. Lesson 47 explains both experiments.

## You can study the topics in three connected tracks.

| Lesson | Topic |
|---|---|
'''
for p in lectures:content+=f'| {p.name.split("-")[0]} | {{doc}}`{title(p)} <lectures/{p.stem}>` |\n'
for caption,files in [('Getting Started',['getting-started/index','getting-started/run-locally','getting-started/run-in-browser']),('Numerical methods',[f'lectures/{p.stem}' for p in lectures if int(p.name.split('-')[0])<=16]),('CPU and distributed computing',[f'lectures/{p.stem}' for p in lectures if 17<=int(p.name.split('-')[0])<=30]),('GPU computing',[f'lectures/{p.stem}' for p in lectures if 31<=int(p.name.split('-')[0])<=46]),('Integration',[f'lectures/{p.stem}' for p in lectures if int(p.name.split('-')[0])>=47]),('Reference',[f'reference/{p.stem}' for p in sorted((R/'docs/reference').glob('*.md'))])]:
 content+='\n```{toctree}\n:hidden:\n:maxdepth: 2\n:caption: '+caption+'\n\n'+'\n'.join(files)+'\n```\n'
(R/'docs/index.md').write_text(content)
print(f'Mapped {len(rows)} textbook chapters and appendices across {len(lectures)} lesson files.')
