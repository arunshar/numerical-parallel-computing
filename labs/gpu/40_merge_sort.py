"""CPU reference experiment. This script does not execute CUDA or measure GPU performance."""
a, b = [1,3,3,8], [2,3,7,9]
def co_rank(k, a, b):
    low, high = max(0,k-len(b)), min(k,len(a))
    while low <= high:
        i = (low+high)//2; j = k-i
        if i > 0 and j < len(b) and a[i-1] > b[j]: high = i-1
        elif j > 0 and i < len(a) and b[j-1] >= a[i]: low = i+1
        else: return i,j
    raise AssertionError("No valid co-rank")
merged = sorted([(v,0,i) for i,v in enumerate(a)] + [(v,1,i) for i,v in enumerate(b)])
for k in range(len(a)+len(b)+1):
    i,j = co_rank(k,a,b)
    assert i == sum(origin==0 for _,origin,_ in merged[:k])
    assert i+j == k
print("middle_co_rank=", co_rank(4,a,b))
print("merged_values=", [v for v,_,_ in merged])
values = [13,2,9,0,15,7]
for shift in [0,2]:
    buckets = [[] for _ in range(4)]
    for value in values: buckets[(value >> shift)&3].append(value)
    values = [value for bucket in buckets for value in bucket]
assert values == [0,2,7,9,13,15]
print("two_pass_radix_result=", values)
