import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [B, 32, 128, 128]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # [B, 64, 64, 64]
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 64 * 64, latent_dim)
        self.fc_logvar = nn.Linear(64 * 64 * 64, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 64 * 64 * 64)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 64, 64)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # [B, 32, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # [B, 3, 256, 256]
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        z = self.reparameterize(mu, logvar)
        x_decoded = self.decoder(self.decoder_fc(z))
        return x_decoded, mu, logvar
