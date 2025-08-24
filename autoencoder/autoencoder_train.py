from autoencoder.autoencoder_model import SimpleAutoencoder
import torch
import torch.optim as optim



def train_model(train_loader, device, num_epochs=10):
    model = SimpleAutoencoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images in train_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")
    return model

