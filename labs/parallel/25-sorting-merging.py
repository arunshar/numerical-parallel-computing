import json
source=[7,2,6,3,8,1,5,4]
a=source.copy();phases=[]
for phase in range(len(a)):
 for i in range(phase%2,len(a)-1,2):
  if a[i]>a[i+1]:a[i],a[i+1]=a[i+1],a[i]
 phases.append(a.copy())
assert a==sorted(source)
print(json.dumps({'phases':phases,'sorted':a,'preserves_multiset':True},indent=2))
