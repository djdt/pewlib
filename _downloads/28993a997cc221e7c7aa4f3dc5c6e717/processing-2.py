import matplotlib.pyplot as plt
import numpy as np
from pewlib.process import filters
a = np.sin(np.linspace(0, 1, 2500).reshape((50, 50)))
a += np.random.poisson(lam=0.01, size=(50, 50))
b = filters.rolling_median(a, (5, 5), threshold=3.0)

f, ax = plt.subplots(1, 2)
ax[0].imshow(a, vmax=1.0)
ax[0].set_title("raw image 'a'")
ax[1].imshow(b, vmax=1.0)
ax[1].set_title("filtered image 'b'")
plt.show()