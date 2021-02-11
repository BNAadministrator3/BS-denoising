import torch
import numpy as np
from matplotlib import pyplot as plt
from hht import hilbert_huang, hilbert_spectrum, plot_IMFs
from scipy.signal import chirp
# import IPython
import pdb

fs = 1000
duration = 2.0
delta_t = 1/fs
t = torch.arange(fs*duration) * delta_t
x = torch.from_numpy(chirp(t, 5, 0.8, 10, method = "quadratic", phi=100)) * torch.exp(-4*(t-1)**2) + \
    torch.from_numpy(chirp(t, 40, 1.2, 50, method = "linear")) * torch.exp(-4*(t-1)**2)
plt.plot(t, x)
plt.show()

imfs, imfs_env, imfs_freq = hilbert_huang(x, delta_t, num_extrema=3)

plot_IMFs(x, imfs, delta_t)

spectrum, t, f = hilbert_spectrum(imfs_env, imfs_freq, delta_t, time_range = (0.25, 1.75))

pdb.set_trace()