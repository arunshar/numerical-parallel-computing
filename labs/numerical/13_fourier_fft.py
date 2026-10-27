"""Check a DFT convention, aliasing, and convolution padding."""
import numpy as np

n = 16
j = np.arange(n)
signal = np.cos(2*np.pi*3*j/n)
F = np.exp(-2j*np.pi*np.outer(j,j)/n)
direct = F@signal
fast = np.fft.fft(signal)
print(f"DFT versus FFT difference: {np.max(np.abs(direct-fast)):.6e}")
print("nonzero frequency bins:", np.flatnonzero(np.abs(fast)>1e-10))
print(f"inverse error: {np.max(np.abs(np.fft.ifft(fast)-signal)):.6e}")
aliased = np.cos(2*np.pi*13*j/n)
print(f"sample difference for cosine bins 3 and 13: {np.max(np.abs(signal-aliased)):.6e}")
a = np.array([1.,2.,3.]);b = np.array([4.,5.])
length = len(a)+len(b)-1
conv = np.fft.ifft(np.fft.fft(a,length)*np.fft.fft(b,length)).real
print("padded FFT convolution:", np.round(conv, 6))
assert np.allclose(conv, np.convolve(a,b))
