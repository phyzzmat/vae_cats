import torchvision
import torch
from torch import nn
from matplotlib import pyplot as plt


def show_images(x, scale=15, line_width=10):
    plt.figure(figsize=(scale, scale / line_width *
               (x.shape[0] // line_width + 1)))
    x = x.view(-1, 3, 64, 64)
    mtx = torchvision.utils.make_grid(x, nrow=line_width, pad_value=1)
    return mtx.permute([1, 2, 0]).numpy()


def kl(q_mu, q_sigma, p_mu, p_sigma):
    d = q_mu.shape[-1]
    d_mu = q_mu - p_mu
    return (2 * torch.log(p_sigma).sum(dim=-1) - 2 * torch.log(q_sigma).sum(dim=-1) - d + (p_sigma ** -2 * q_sigma ** 2).sum(dim=-1) + (d_mu * p_sigma ** -2 * d_mu).sum(dim=-1)) / 2


class ClampLayer(nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max
        self.kwargs = {}
        if min is not None:
            self.kwargs['min'] = min
        if max is not None:
            self.kwargs['max'] = max

    def forward(self, input):
        return torch.clamp(input, **self.kwargs)


class ProposalNetwork(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):

        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Unflatten(-1, (num_input_channels, 64, 64)),
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3,
                      padding=1, stride=2), 
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3,
                      padding=1, stride=2), 
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3,
                      padding=1, stride=2), 
            act_fn(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(4 * c_hid, 4 * c_hid, kernel_size=3,
                      padding=1, stride=2),      
                    act_fn(),
            nn.Flatten(),
            nn.Linear(4 * 16 * c_hid, 2 * latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class GenerativeNetwork(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4 * 16 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4*c_hid, 4*c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(4*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2*c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3,
                               output_padding=1, padding=1, stride=2),
            ClampLayer(-5, 5),
            nn.Flatten(),
            nn.Tanh()
        )

    def forward(self, x):
        x0, x1 = x.shape[0], x.shape[1]
        x = self.linear(x.view(x0 * x1, -1))
        x = x.reshape(x0 * x1, -1, 4, 4)
        x = self.net(x).reshape(x0, x1, -1)
        return (x + 1) / 2
    

class VAE(nn.Module):

    def __init__(self, n_channels, d, D, beta=2):
        super(type(self), self).__init__()
        self.d = d
        self.D = D
        self.beta = beta
        self.n_channels = n_channels

        self.proposal_network = ProposalNetwork(n_channels, 64, d).to('cuda')
        self.generative_network = GenerativeNetwork(n_channels, 64, d).to('cuda')
        self.proposal_sigma_head = nn.Softplus().to('cuda')

    def proposal_distr(self, x):
        mu = self.proposal_network(x)[:, :self.d]
        sigma = self.proposal_sigma_head(self.proposal_network(x)[:, self.d:])
        return mu, sigma

    def prior_distr(self, x):
        return torch.zeros(x.shape[0], self.d, device='cuda'), torch.ones(x.shape[0], self.d, device='cuda')

    def sample_latent(self, mu, sigma, K=1):
        eps = torch.randn(mu.shape[0], K, mu.shape[1], device='cuda')
        return eps * sigma.unsqueeze(-2) + mu.unsqueeze(-2)

    def generative_distr(self, z):
        probs = self.generative_network(z)
        return probs

    def batch_vlb(self, batch):
        prop_mu, prop_sigma = self.proposal_distr(batch)
        vlb2 = -kl(prop_mu, prop_sigma, *self.prior_distr(batch))
        sample = self.sample_latent(prop_mu, prop_sigma)
        vlb1 = -((batch - self.generative_distr(sample).squeeze(-2)) ** 2).sum(dim=(-1, -2))
        vlb = vlb1 + self.beta * vlb2
        return vlb.mean(), vlb1.mean(), vlb2.mean()

    def generate_samples(self, num_samples):
        rnd = torch.randn(num_samples, 1, self.d, device='cuda')
        return self.generative_distr(rnd)
