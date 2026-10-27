import json
p=4
rows=[{'rank':r,'send_to':(r+1)%p,'receive_from':(r-1)%p,
       'received_value':(r-1)%p} for r in range(p)]
assert [r['received_value'] for r in rows]==[3,0,1,2]
print(json.dumps(rows,indent=2))
