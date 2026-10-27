"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
def subdivide(size, threshold):
    pending=[size];leaves=[];internal=0
    while pending:
        current=pending.pop()
        if current<=threshold:leaves.append(current)
        else:
            internal+=1
            left=current//2;right=current-left
            pending.extend([left,right])
    return leaves,internal
for threshold in [8,32,128]:
    leaves,internal=subdivide(1000,threshold)
    assert sum(leaves)==1000 and max(leaves)<=threshold
    print(f"threshold={threshold}, leaves={len(leaves)}, internal_nodes={internal}, "
          f"leaf_range={min(leaves)}..{max(leaves)}")
