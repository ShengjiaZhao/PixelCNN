from matplotlib import pyplot as plt
import numpy as np

files = ['experiment_log/stein.csv', 'experiment_log/mmd.csv', 'experiment_log/adv.csv', 'experiment_log/elbo.csv']
names = ['stein', 'mmd', 'adv', 'elbo']


fig, ax = plt.subplots()
for file, name in zip(files, names):
    reader = open(file)
    iter_list = []
    val_list = []
    reader.readline()
    while True:
        line = reader.readline().split(',')
        if len(line) < 3:
            break
        iter_list.append(int(line[1]))
        val_list.append(float(line[2]))
    # Smooth
    box = np.ones(100) / 100.0
    smooth_val = np.convolve(val_list, box, mode='same')
    ax.plot(iter_list, smooth_val, label=name)

ax.set_yscale('log')
ax.set_xlabel('iterations')
ax.set_ylabel('mmd')
ax.legend()
plt.show()