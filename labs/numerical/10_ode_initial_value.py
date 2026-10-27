"""Separate integration accuracy from absolute stability."""
import math

for n in [10, 20, 40, 80]:
    h = 1/n
    euler, rk4 = 1., 1.
    for _ in range(n):
        euler += h*(-euler)
        k1 = -rk4
        k2 = -(rk4+h*k1/2)
        k3 = -(rk4+h*k2/2)
        k4 = -(rk4+h*k3)
        rk4 += h*(k1+2*k2+2*k3+k4)/6
    print(f"n={n:2d}: Euler error={abs(euler-math.exp(-1)):.6e}, RK4 error={abs(rk4-math.exp(-1)):.6e}")
rate, h = -100., 0.03
forward_factor = 1+h*rate
backward_factor = 1/(1-h*rate)
print(f"forward Euler amplification: {forward_factor:.6f}")
print(f"backward Euler amplification: {backward_factor:.6f}")
print(f"exact amplification: {math.exp(h*rate):.6f}")
assert abs(forward_factor)>1 and abs(backward_factor)<1
