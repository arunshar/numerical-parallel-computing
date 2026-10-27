import json
N, LINE, ITEM = 64, 64, 8
rows = []
for stride in (1, 2, 4, 8):
    addresses = [i * stride * ITEM for i in range(N)]
    lines = {a // LINE for a in addresses}
    rows.append({'stride_elements': stride, 'useful_bytes': N * ITEM,
                 'line_bytes': len(lines) * LINE,
                 'utilization': N * ITEM / (len(lines) * LINE)})
assert [r['line_bytes'] for r in rows] == [512, 1024, 2048, 4096]
print(json.dumps(rows, indent=2))
