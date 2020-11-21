import matplotlib.pyplot as plt
import numpy as np
from pew.lib import filters
import numpy as np
from pew.lib import filters
a = np.sin(np.linspace(0, 10, 50))
a[5::10] +=np.random.choice([-1, 1], size=5)
b = filters.rolling_mean(a, 3, threshold=3.0)

plt.plot(a, c="black")
plt.plot(b, ls=":", c="red", label="filtered")
plt.legend()
plt.show()