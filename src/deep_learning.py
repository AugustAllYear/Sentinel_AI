"""Autoencoder for anomaly detection (PyTorch)."""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(X_train, epochs=50, batch_size=32, encoding_dim=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    tensor_data = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    model = Autoencoder(input_dim=X_scaled.shape[1], encoding_dim=encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = torch.utils.data.TensorDataset(tensor_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            data = batch[0]
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")

    return model, scaler

def predict_anomaly(model, scaler, X_new, threshold_percentile=95):
    device = next(model.parameters()).device
    X_scaled = scaler.transform(X_new)
    tensor_data = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructions = model(tensor_data).cpu().numpy()
    mse = np.mean((X_scaled - reconstructions)**2, axis=1)
    # Use a percentile of training reconstruction errors as threshold
    # (In production, you'd compute threshold on training set)
    threshold = np.percentile(mse, threshold_percentile)
    return (mse > threshold).astype(int), mse