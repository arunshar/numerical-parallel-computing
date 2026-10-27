"""Build a single study document from the same sources as the course website."""
from pathlib import Path
import re,json,subprocess,ast
import yaml
ROOT=Path(__file__).resolve().parents[1]
def convert(text,source):
 def role(m):
  content=m[1];match=re.match(r'(.*?)\s*<([^>]+)>$',content)
  label,target=(match[1],match[2]) if match else (content.split('/')[-1],content)
  path=(source.parent/target).resolve().relative_to(ROOT/'docs')
  return '['+label+'](https://arunshar.github.io/numerical-parallel-computing/'+str(path)+'.html)'
 text=re.sub(r'\{doc\}`([^`]+)`',role,text)
 def download(m):
  match=re.match(r'(.*?)\s*<([^>]+)>$',m[1])
  label,target=(match[1],match[2]) if match else (m[1],m[1])
  path=(source.parent/target).resolve().relative_to(ROOT)
  return '['+label+'](https://github.com/arunshar/numerical-parallel-computing/blob/main/'+str(path)+')'
 text=re.sub(r'\{download\}`([^`]+)`',download,text)
 def quiz(m):
  questions=yaml.safe_load(m[1]);out=['### Review questions and answers\n']
  for i,q in enumerate(questions,1):
   out += [f'**Question {i}. {q["q"]}**\n']
   out += [f'{j+1}. {opt}' for j,opt in enumerate(q['options'])]
   out += ['\n**Answer.** '+str(q['options'][q['answer']])+' '+q['explain']+'\n']
  return '\n'.join(out)
 text=re.sub(r'```\{quiz\}\s*\n(.*?)```',quiz,text,flags=re.S)
 def include(m):
  raw=m[1].strip();opts=m[2];path=(source.parent/raw).resolve()
  if not path.is_relative_to(ROOT):raise ValueError(raw)
  lang=re.search(r':language:\s*(\S+)',opts)
  return '```'+(lang[1] if lang else '')+'\n'+path.read_text()+'\n```'
 text=re.sub(r'```\{literalinclude\}\s*([^\n]+)\n(.*?)```',include,text,flags=re.S)
 text=re.sub(r':::\{(?:admonition|note|warning|tip)\}\s*([^\n]*)',lambda m:'**'+(m[1] or 'Note')+'.**',text)
 text=re.sub(r'^:[\w-]+:.*$','',text,flags=re.M)
 text=re.sub(r'^:::\s*$','',text,flags=re.M)
 text=re.sub(r'```\{(?:code-block)\}\s*(\S*)',r'```\1',text)
 return text
parts=['# Numerical and Parallel Computing\n\nThis document combines original study lessons for CSCI 5302 and CSCI 5451. The course website and repository contain the same lessons, runnable labs, source maps, and verification evidence.\n\nThe notes explain the public course topics and the major topics in all four associated textbook outlines. Some advanced topics are surveys. The coverage ledger records those distinctions. This document does not reproduce or replace the textbooks.\n\nPrepared for Arun Sharma. Codex drafted this edition. Claude refinement is pending.\n']
for p in sorted((ROOT/'docs/lectures').glob('*.md')):
 parts.append(convert(p.read_text(),p))
for p in sorted((ROOT/'docs/lectures').glob('*.ipynb')):
 nb=json.loads(p.read_text());sections=[]
 for c in nb['cells']:
  src=''.join(c['source'])
  if c['cell_type']=='markdown':sections.append(convert(src,p))
  elif 'remove-input' in c.get('metadata',{}).get('tags',[]) and 'th.quiz(' in src:
   call=ast.parse(src).body[0].value
   questions=ast.literal_eval(call.args[0])
   text='```{quiz}\n'+yaml.safe_dump(questions,sort_keys=False)+'```'
   sections.append(convert(text,p))
  elif 'remove-input' not in c.get('metadata',{}).get('tags',[]):
   sections.append('```python\n'+src+'\n```')
   for o in c.get('outputs',[]):
    if 'text' in o:sections.append('```text\n'+''.join(o['text'])+'\n```')
 parts.append('\n\n'.join(sections))
for name in ['coverage','sources','glossary','claude-handoff','validation']:
 p=ROOT/'docs/reference'/f'{name}.md'
 if p.exists():parts.append(convert(p.read_text(),p))
out=ROOT/'dist';out.mkdir(exist_ok=True)
(out/'STUDY_GUIDE.md').write_text('\n\n---\n\n'.join(parts))
print(f'Wrote {out / "STUDY_GUIDE.md"}')
