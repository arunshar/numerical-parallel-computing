import json
values = list(range(1, 1001))
partials = [sum(values[i::4]) for i in range(4)]
wrong = 0
read_a, read_b = wrong, wrong
wrong = read_a + 1
wrong = read_b + 1
assert sum(partials) == 500500 and wrong == 1
print(json.dumps({'partial_sums': partials, 'correct_sum': sum(partials),
                  'modeled_lost_update': wrong}, indent=2))
