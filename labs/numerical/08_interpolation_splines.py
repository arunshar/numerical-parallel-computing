"""Compare polynomial node choices and piecewise linear interpolation."""
import numpy as np

def barycentric(nodes, values, query):
    weights = np.array([1/np.prod(nodes[j]-np.delete(nodes, j)) for j in range(len(nodes))])
    output = []
    for t in query:
        hit = np.flatnonzero(np.isclose(t, nodes, atol=1e-14, rtol=0))
        if len(hit):
            output.append(values[hit[0]])
        else:
            terms = weights/(t-nodes)
            output.append((terms@values)/terms.sum())
    return np.array(output)

f = lambda x: 1/(1+25*x*x)
grid = np.linspace(-1, 1, 2001)
n = 15
for label, nodes in [("equally spaced", np.linspace(-1, 1, n)),
                     ("Chebyshev", np.cos(np.arange(n)*np.pi/(n-1)))]:
    error = np.max(np.abs(barycentric(nodes, f(nodes), grid)-f(grid)))
    print(f"{label} maximum error: {error:.6e}")
nodes = np.linspace(-1, 1, n)
linear = np.interp(grid, nodes, f(nodes))
print(f"piecewise linear maximum error: {np.max(np.abs(linear-f(grid))):.6e}")
assert np.allclose(barycentric(nodes, f(nodes), nodes), f(nodes))
