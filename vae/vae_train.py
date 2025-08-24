import torch
import torch.optim as optim
from vae.vae_model import VAE
import torch.nn.functional as F

def vae_loss(reconstructed, original, mu, logvar):
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

def train_model(train_loader, device, num_epochs=10):
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images in train_loader:
            images = images.to(device)
            reconstructed, mu, logvar = model(images)
            loss = vae_loss(reconstructed, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.2f}")
    return model
