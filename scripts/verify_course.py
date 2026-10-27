"""Check course structure, quiz schemas, source links, and original CPU labs."""
from pathlib import Path
import json,re,subprocess,sys,hashlib,platform,os
import yaml
import numpy as np
import scipy
import ast
ROOT=Path(__file__).resolve().parents[1]
errors=[];pages=[]
for p in sorted((ROOT/'docs/lectures').glob('*.md')):
 s=p.read_text();count=0
 if '@@' in s:errors.append(f'{p.name}: unresolved placeholder')
 if '\u2014' in s:errors.append(f'{p.name}: em dash')
 for block in re.findall(r'```\{quiz\}\s*\n(.*?)```',s,re.S):
  qs=yaml.safe_load(block)
  for q in qs:
   assert isinstance(q['q'],str) and isinstance(q['options'],list)
   assert 0<=q['answer']<len(q['options']) and isinstance(q['explain'],str)
  count+=len(qs)
 if count<5:errors.append(f'{p.name}: only {count} quiz questions')
 for title in ['Learning goals','Setup','Summary','Check your understanding']:
  if title.lower() not in s.lower():errors.append(f'{p.name}: missing {title}')
 pages.append({'path':str(p.relative_to(ROOT)),'words':len(s.split()),'quiz_questions':count})
for p in sorted((ROOT/'docs/lectures').glob('*.ipynb')):
 nb=json.loads(p.read_text());count=0;texts=[];executed=0
 for cell in nb['cells']:
  source=''.join(cell['source']);texts.append(source)
  if cell['cell_type']=='code':
   if cell.get('execution_count') is None:errors.append(f'{p.name}: unexecuted code cell')
   else:executed+=1
   if any(o.get('output_type')=='error' for o in cell.get('outputs',[])):errors.append(f'{p.name}: error output')
   if 'th.quiz(' in source:
    for node in ast.walk(ast.parse(source)):
     if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='quiz':
      questions=ast.literal_eval(node.args[0]);count+=len(questions)
      for q in questions:
       assert isinstance(q['q'],str) and 0<=q['answer']<len(q['options']) and isinstance(q['explain'],str)
 if count<5:errors.append(f'{p.name}: only {count} quiz questions')
 if any('@@' in t or '\u2014' in t for t in texts):errors.append(f'{p.name}: placeholder or em dash')
 pages.append({'path':str(p.relative_to(ROOT)),'words':sum(len(t.split()) for t in texts),'quiz_questions':count,'executed_code_cells':executed})
results=[]
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
for folder in ['numerical','parallel','gpu']:
 for p in sorted((ROOT/'labs'/folder).glob('*.py')):
  if not re.match(r'^\d{2}[-_]',p.name):continue
  run=subprocess.run([sys.executable,str(p)],cwd=ROOT,env=env,text=True,capture_output=True,timeout=90)
  results.append({'path':str(p.relative_to(ROOT)),'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  if run.returncode:errors.append(f'{p.name}: failed with {run.stderr[-300:]}')
report={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'pages':pages,'cpu_labs':results,'errors':errors,'claims':'CPU examples and analytic models do not validate CUDA execution or hardware performance.'}
(ROOT/'evidence').mkdir(exist_ok=True);(ROOT/'evidence/validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'lectures':len(pages),'words':sum(p['words'] for p in pages),'quiz_questions':sum(p['quiz_questions'] for p in pages),'cpu_labs':len(results),'errors':errors},indent=2))
sys.exit(bool(errors))
