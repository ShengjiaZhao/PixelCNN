from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import math

log_file = open("mutual_info_autoencoder_stein_large")
iter_list = []
info_list = []
error_list = []

smooth_step = 10
info_runavg = None
error_runavg = None

for iter in range(100000):
    line = log_file.readline().split()
    if len(line) < 3:
        break
    iter_list.append(int(line[0]))

    if info_runavg is None:
        info_runavg = float(line[1])
    elif iter < smooth_step:
        info_runavg = info_runavg * iter / (iter + 1) + float(line[1]) / (iter + 1)
    else:
        info_runavg = info_runavg * smooth_step / (smooth_step + 1) + float(line[1]) / (smooth_step + 1)

    if error_runavg is None:
        error_runavg = float(line[2])
    elif iter < smooth_step:
        error_runavg = error_runavg * iter / (iter + 1) + float(line[2]) / (iter + 1)
    else:
        error_runavg = error_runavg * smooth_step / (smooth_step + 1) + float(line[2]) / (smooth_step + 1)

    info_list.append(info_runavg / math.log(2.0))
    error_list.append(error_runavg)

x = np.exp(np.linspace(1, np.log(np.max(iter_list))-0.1, num=200, endpoint=True))
f = interp1d(iter_list, info_list, kind='cubic')

fig, ax1 = plt.subplots()
lns1 = ax1.plot(x, f(x), c='g', lw=4, label='mutual information')
ax1.set_xlabel('iteration')
ax1.set_ylabel('bits (mutual information)')
ax1.set_xscale('log')
ax1.set_ylim([0.0, 16.0])
# Make the y-axis label, ticks and tick labels match the line color.
# ax1.set_ylabel('')

f_error = interp1d(iter_list, error_list, kind='cubic')
ax2 = ax1.twinx()
ax2.set_ylabel('bits per dim (data nll)')
lns2 = ax2.plot(x, f_error(x), c='b', lw=4, label='train data nll')
# ax2.set_yscale('log')
ax2.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
ax2.set_ylim([0.05, 0.6])
# ax2.set_ylabel('sin')
# plt.xticks(x, labels, rotation='vertical')
# ax2.get_yaxis().get_major_formatter().set_useOffset(False)
# ax2.set_yticks([0.1, 0.2, 0.3, 0.4])


for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax2.yaxis.get_major_ticks():
    tick.label2.set_fontsize(14)
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax1.xaxis.label.set_size(16)
ax1.yaxis.label.set_size(16)
ax2.yaxis.label.set_size(16)

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=1, framealpha=0.5, prop={'size': 20})

'''
plt.xscale('log')
plt.xlabel('iteration')
plt.ylabel('bits per dim')
plt.ylim(0.0, 6.0)
plt.gca().yaxis.label.set_size(16)
plt.gca().xaxis.label.set_size(16)
for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
plt.legend(loc=3, )
plt.xlim([100, None])
plt.show()
'''
plt.show()