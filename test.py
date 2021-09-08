import numpy as np
import matplotlib.pyplot as plt

B = np.random.random(500)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
l1, = plt.plot(B)
plt.plot(0.5*np.ones(len(B)), 'k--', label='B limit')
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.title('Magnet Flux Density - Daut et. al (2013)')
plt.xlabel('Time stamp')
plt.ylabel('B [T]')
plt.show()