import json, math
rows=[]
for p in (1,2,4,8,16):
    amdahl=1/(0.1+0.9/p)
    gustafson=p-0.1*(p-1)
    required=4*p*math.log2(p) if p>1 else None
    rows.append({'processors':p,'amdahl_speedup':round(amdahl,4),
                 'gustafson_scaled_speedup':round(gustafson,4),
                 'work_for_efficiency_0_8':required})
assert rows[0]['work_for_efficiency_0_8'] is None
assert abs(rows[2]['amdahl_speedup']-3.0769)<1e-4
for row in rows[1:]:
    p=row['processors']
    work=row['work_for_efficiency_0_8']
    efficiency=work/(work+p*math.log2(p))
    assert abs(efficiency-0.8)<1e-12
print(json.dumps(rows,indent=2))
