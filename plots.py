from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import numpy as np

plt.rcParams['font.size'] = 12

mmd_type = ['stein', 'mmd', 'adv', 'elbo']
mmd_data = [
    [2e-4, 8e-4, 1.15e-3, 4e-3, 1.4e-2],
    [2e-4, 7e-4, 5e-4, 8e-4, 1.08e-3],
    [0.052, 0.045, 0.0093, 0.008, 0.015],
    [2.7e-5, 3e-5, 4.9e-5, 3.9e-5, 4e-5]
]
logdet_type = ['stein', 'mmd', 'adv', 'elbo']
logdet_data = [
    [-0.123, -0.7, -1.32, -5, -17.92],
    [-0.088, 0.121, -0.22, -0.36, -2.341],
    [-8.0, -7.394, -7.6, -7.4, -13.7],
    [-3.3e-3, -0.013, -3.1e-3, -0.04, -0.13]
]
semi_1000_type = ['Stein', 'MMD', 'Adversarial', 'ELBO', 'Unregularized']
semi_1000_data = [
    [0.462, 0.681, 0.845, 0.839, 0.842],
    [0.58, 0.895, 0.956, 0.972, 0.975],
    [0.34, 0.37, 0.88, 0.901, 0.845],
    [0.1, 0.098, 0.1, 0.102, 0.11],
    [0.916, 0.923, 0.954, 0.967, 0.968]
]
ce_type = ['stein', 'mmd', 'adv', 'elbo']
ce_data = [
    [0.027, 0.027, 0.039, 0.049, 0.057],
    [0.015, 0.017, 0.019, 0.022, 0.036],
    [0.078, 0.042, 0.035, 0.028, 0.021],
    [0.028, 0.031, 0.028, 0.030, 0.033],
]

plot_data = mmd_data
plot_type = mmd_type
dims = [2, 5, 10, 20, 40]


def plot_mmd(ax):
    for type, value in zip(mmd_type, mmd_data):
        ax.plot(dims, value, lw=3, label=type)
    ax.set_title('(A) MMD Distance')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('latent dimension', )
    ax.set_ylabel('mmd')
    ax.xaxis.set_ticks(dims)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    # ax[0].legend()

def plot_covdet(ax):
    for type, value in zip(logdet_type, logdet_data):
        ax.plot(dims, value, lw=3, label=type)
    # ax[1].set_yscale('log')
    ax.set_title('(B) Covariance Log Determinant')
    ax.set_xscale('log')
    ax.set_xlabel('latent dimension')
    ax.set_ylabel('logdet of covaraince')
    ax.xaxis.set_ticks(dims)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.invert_yaxis()
    # ax[1].legend()

def plot_ce(ax):
    for type, value in zip(ce_type, ce_data):
        ax.plot(dims, value, lw=3, label=type)
    # ax[1].set_yscale('log')
    ax.set_title('(C) Class Distribution Mismatch')
    ax.set_xscale('log')
    ax.set_xlabel('latent dimension')
    ax.set_ylabel('class distribution cross entropy')
    ax.xaxis.set_ticks(dims)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))


def plot_semi(ax):
    for type, value in zip(semi_1000_type, semi_1000_data):
        ax.plot(dims, [1.0 - val for val in value], lw=3, label=type)
    ax.set_title('(E) Semi-supervised Classification Error')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('latent dimension')
    ax.set_ylabel('semi-supervised error')
    ax.xaxis.set_ticks(dims)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_ticks([0.01, 0.1, 1.0])


def plot_convergence(ax):
    files = ['experiment_log/stein.csv', 'experiment_log/mmd.csv', 'experiment_log/adv.csv', 'experiment_log/elbo.csv']
    names = ['stein', 'mmd', 'adv', 'elbo']
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
        box = np.ones(20) / 20.0
        smooth_val = np.convolve(val_list, box, mode='same')
        ax.plot(iter_list, smooth_val, lw=3, label=name)
    ax.set_xlim([None, 38000])
    ax.set_title('(D) MMD Loss Training Curve')
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('mmd')


fig, ax = plt.subplots(2, 3)
ax = ax.flatten()
plt.subplots_adjust(wspace=0.3, hspace=0.3)


plot_mmd(ax[0])
plot_covdet(ax[1])
plot_ce(ax[2])
plot_convergence(ax[3])
plot_semi(ax[4])
ax[4].legend(bbox_to_anchor=(1, 0), loc='lower left', prop={'size': 16})
ax[5].set_visible(False)

plt.show()