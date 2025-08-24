import torch
import torch.optim as optim
from fastflow.fastflow_model import FeatureExtractor, FastFlow

def train_model(train_loader, device, num_epochs=10):
    extractor = FeatureExtractor().to(device)
    flow_model = FastFlow(in_channels=256).to(device)  # خروجی layer3 در ResNet18

    optimizer = optim.Adam(flow_model.parameters(), lr=1e-3)

    extractor.eval()
    for epoch in range(num_epochs):
        flow_model.train()
        total_loss = 0
        for images in train_loader:
            images = images.to(device)
            with torch.no_grad():
                features = extractor(images)
            log_prob, z = flow_model(features)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    return extractor, flow_model
