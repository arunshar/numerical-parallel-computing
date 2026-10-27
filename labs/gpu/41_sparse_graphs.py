"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
import numpy as np
row_ptr = np.array([0,2,2,5,6])
cols = np.array([0,3,0,1,3,2])
vals = np.array([2.,1.,-1.,3.,2.,4.])
x = np.array([1.,2.,3.,4.])
y = np.zeros(4); dense = np.zeros((4,4))
for row in range(4):
    for k in range(row_ptr[row],row_ptr[row+1]):
        y[row] += vals[k]*x[cols[k]]
        dense[row,cols[k]] += vals[k]
assert np.allclose(y,dense@x)
lengths = np.diff(row_ptr)
print("csr_product=", y.tolist())
print(f"nonzeros={len(vals)}, ell_slots={len(lengths)*max(lengths)}, padding={len(lengths)*max(lengths)-len(vals)}")
graph = [[1,2],[3],[3,4],[5],[5],[],[]]
distance = [-1]*len(graph); distance[0]=0; frontier=[0]
while frontier:
    following=[]
    for vertex in frontier:
        for neighbor in graph[vertex]:
            if distance[neighbor]<0:
                distance[neighbor]=distance[vertex]+1;following.append(neighbor)
    frontier=following
assert distance == [0,1,1,2,2,3,-1]
print("bfs_distances=",distance)
