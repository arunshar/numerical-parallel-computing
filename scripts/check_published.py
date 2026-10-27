"""Compare every live course page with a local Sphinx build."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
import argparse,json,hashlib

class Headings(HTMLParser):
    def __init__(self):
        super().__init__();self.active=None;self.buffer=[];self.items=[];self.quizzes=0
    def handle_starttag(self,tag,attrs):
        attributes=dict(attrs)
        if tag in ('title','h2'):
            self.active=tag;self.buffer=[]
        if 'tq' in attributes.get('class','').split():self.quizzes+=1
    def handle_data(self,data):
        if self.active:self.buffer.append(data)
    def handle_endtag(self,tag):
        if tag==self.active:
            self.items.append((tag,' '.join(''.join(self.buffer).split())))
            self.active=None

def signature(text):
    parser=Headings();parser.feed(text);return parser.items,parser.quizzes

def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--base',default='https://arunshar.github.io/numerical-parallel-computing/')
    args.add_argument('--output',type=Path)
    options=args.parse_args();root=Path(__file__).resolve().parents[1]
    sources=[p for p in (root/'docs').rglob('*') if p.suffix in {'.md','.ipynb'} and '_build' not in p.parts]
    def check(source):
        relative=source.relative_to(root/'docs').with_suffix('.html')
        url=options.base.rstrip('/')+'/'+relative.as_posix()
        try:
            with urlopen(Request(url,headers={'User-Agent':'CoursePublicationCheck/1.0'}),timeout=25) as response:
                content=response.read().decode();status=response.status
            expected=(root/'docs/_build/html'/relative).read_text()
            actual_signature=signature(content);expected_signature=signature(expected)
            return {'path':str(relative),'url':url,'http_status':status,'matches_title_headings_quizzes':actual_signature==expected_signature,'quiz_questions':actual_signature[1],'sha256':hashlib.sha256(content.encode()).hexdigest()}
        except Exception as error:return {'path':str(relative),'url':url,'error':str(error)}
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(check,sorted(sources)))
    passed=all(r.get('matches_title_headings_quizzes',False) for r in results)
    report={'base':options.base,'pages':len(results),'passed':passed,'results':results}
    if options.output:options.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'pages':len(results),'passed':passed,'failures':[r for r in results if not r.get('matches_title_headings_quizzes',False)]},indent=2))
    raise SystemExit(0 if passed else 1)
if __name__=='__main__':main()
