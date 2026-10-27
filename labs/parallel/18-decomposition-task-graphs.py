import json
costs = list(range(1, 9))
p = 4
block = [sum(costs[2*i:2*i+2]) for i in range(p)]
cyclic = [sum(costs[i::p]) for i in range(p)]
greedy = [0] * p
for cost in sorted(costs, reverse=True):
    i = min(range(p), key=lambda k: greedy[k])
    greedy[i] += cost
out = {name: {'loads': loads, 'makespan': max(loads)}
       for name, loads in [('block', block), ('cyclic', cyclic), ('greedy', greedy)]}
assert max(block) == 15 and max(cyclic) == 12 and max(greedy) == 9
print(json.dumps(out, indent=2))
