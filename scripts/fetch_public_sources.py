"""Fetch cataloged public course files into an ignored local study directory."""
import argparse,hashlib,json,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--course',choices=['csci5302','csci5451','all'],default='all')
p.add_argument('--pdf-only',action='store_true')
a=p.parse_args()
rows=json.loads((ROOT/'coverage/public-source-inventory.json').read_text())
dest=ROOT/'downloaded-sources';dest.mkdir(exist_ok=True)
seen=set();log=[]
for row in rows:
 if a.course!='all' and row['course']!=a.course:continue
 u=row['url']
 if a.pdf_only and not urlsplit(u).path.lower().endswith('.pdf'):continue
 if u in seen:continue
 seen.add(u)
 filename=hashlib.sha256(u.encode()).hexdigest()[:12]+'-'+Path(row['source_filename']).name
 target=dest/row['course']/filename;target.parent.mkdir(exist_ok=True)
 result={'url':u,'local_path':str(target.relative_to(ROOT))}
 try:
  if target.exists():data=target.read_bytes()
  else:
   with urlopen(Request(u,headers={'User-Agent':'NumericalParallelStudy/1.0'}),timeout=30) as response:
    data=response.read(50_000_001)
    if len(data)>50_000_000:raise ValueError('The response exceeds the per-file limit.')
   target.write_bytes(data)
   time.sleep(.15)
  digest=hashlib.sha256(data).hexdigest()
  result.update(status='saved',bytes=len(data),sha256=digest,matches_catalog=digest==row['sha256'])
 except Exception as exc:result.update(status='failed',error=str(exc))
 log.append(result);print(result['status'],u)
(dest/'retrieval-log.json').write_text(json.dumps(log,indent=2))
print(f'Saved retrieval results in {dest / "retrieval-log.json"}.')
