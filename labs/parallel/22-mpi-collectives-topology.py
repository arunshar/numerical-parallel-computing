import json
counts=[3,3,2,2]
offsets=[sum(counts[:i]) for i in range(4)]
values=list(range(10))
chunks=[values[d:d+c] for d,c in zip(offsets,counts)]
assert sum(chunks,[])==values
local=[3,7,11,15]
prefix=[sum(local[:i+1]) for i in range(4)]
assert prefix==[3,10,21,36]
print(json.dumps({'counts':counts,'displacements':offsets,'chunks':chunks,
                  'inclusive_scan':prefix,'allreduce_sum':sum(local)},indent=2))
