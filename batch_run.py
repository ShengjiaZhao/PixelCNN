import subprocess


reg_types = ['stein', 'mmd', 'elbo', 'adv', 'noreg']
latent_dims = [2, 5, 10, 20, 50]

for reg in reg_types:
    for latent_dim in latent_dims:
        subprocess.call(('python main.py --data=mnist --model=autoencoder_noreg_20 --latent_dim=20').split())


