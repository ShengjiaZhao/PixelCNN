import numpy as np
import math
from matplotlib import pyplot as plt


size = 8000
dim = 50
covdets = []
for i in range(50):
    latent = np.random.normal(size=[size, dim])
    mu = np.mean(latent, axis=0)
    latent = latent - np.tile(np.reshape(mu, [1, mu.shape[0]]), [latent.shape[0], 1])
    cov = np.dot(np.transpose(latent), latent) / (latent.shape[0] - 1)
    covdets.append(np.exp(np.linalg.slogdet(cov)[1]))
    print(i),

plt.hist(covdets)
plt.show()