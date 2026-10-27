import json, math
p, alpha, bandwidth = 8, 2e-6, 10e9
rows=[]
for m in (1024, 1024*1024):
    beta=1/bandwidth
    tree=2*math.log2(p)*(alpha+beta*m)
    ring=2*(p-1)*alpha+2*(p-1)/p*beta*m
    rows.append({'bytes':m,'tree_allreduce_us':round(tree*1e6,3),
                 'ring_allreduce_us':round(ring*1e6,3)})
assert rows[0]['tree_allreduce_us'] < rows[0]['ring_allreduce_us']
assert rows[1]['ring_allreduce_us'] < rows[1]['tree_allreduce_us']
print(json.dumps(rows,indent=2))
