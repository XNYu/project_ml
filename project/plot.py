import matplotlib.pyplot as plt
import numpy as np

x = np.random.random(20)
print(x)
y =np.random.random(20)
z= np.random.random(20)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(x, bins=np.arange(0, 1, 0.1), ls='dashed', alpha = 0.5, lw=3, color= 'b')
ax.hist(y, bins=np.arange(0, 1, 0.1), ls='dotted', alpha = 0.5, lw=3, color= 'r')
ax.hist(z, bins=np.arange(0, 1, 0.1), alpha = 0.5, lw=3, color= 'k')
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 7)
plt.show()