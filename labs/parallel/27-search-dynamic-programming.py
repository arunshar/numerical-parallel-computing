import json
s,t='kitten','sitting'
m,n=len(s),len(t)
d=[[0]*(n+1) for _ in range(m+1)]
for i in range(m+1):d[i][0]=i
for j in range(n+1):d[0][j]=j
widths=[]
for diagonal in range(2,m+n+1):
 cells=[]
 for i in range(1,m+1):
  j=diagonal-i
  if 1<=j<=n:
   d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(s[i-1]!=t[j-1]))
   cells.append((i,j))
 widths.append(len(cells))
assert d[m][n]==3
print(json.dumps({'edit_distance':d[m][n],'wavefront_widths':widths,
                  'cells_updated':sum(widths)},indent=2))
