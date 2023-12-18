from matplotlib import pyplot as plt
import torch
from model import VAE
from train import show_images

vae = VAE(3, 64, 64 * 64, 10)
vae.load_state_dict(torch.load('model_600'))
img = show_images(vae.generate_samples(100).detach().cpu())
plt.imshow(img)
plt.show()