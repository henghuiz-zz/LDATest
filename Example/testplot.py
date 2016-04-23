import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

plt.plot(X, C, '-o', linewidth=2.0)
plt.plot(X, S)

plt.show()
plt.xlabel('x')
plt.ylabel('y')