import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle


class SNP_CNN(nn.Module):
    def __init__(self, n_components: int, n_classes: int, embed_dim: int = 128):
        super(SNP_CNN, self).__init__()

        # each PCA component is projected into embed_dim space
        self.embedding = nn.Linear(1, embed_dim)

        self.conv = nn.Sequential(
            # input: (batch, embed_dim, n_components)
            nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)    # (batch, 256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),              # (batch, 256)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (batch, n_components)
        x = x.unsqueeze(-1)            # (batch, n_components, 1)
        x = torch.relu(self.embedding(x))  # (batch, n_components, embed_dim)
        x = x.permute(0, 2, 1)        # (batch, embed_dim, n_components)
        x = self.conv(x)
        x = self.classifier(x)
        return x


def train_cnn(X_pca: np.ndarray, y: np.ndarray,
              n_epochs: int = 100,
              batch_size: int = 16,
              lr: float = 1e-3):

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    print(f"Training on: {torch.cuda.get_device_name(0)}")

    from sklearn.preprocessing import LabelEncoder
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")

    # input is (batch, n_components) — no unsqueeze needed here, forward handles it
    X_tensor = torch.tensor(X_pca, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    dataset    = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model     = SNP_CNN(n_components=X_pca.shape[1], n_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        total_loss    = 0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * len(y_batch)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += len(y_batch)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")

    torch.save(model.state_dict(), "cnn_model.pt")
    with open("cnn_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("CNN model saved to cnn_model.pt")

    return model, le

if __name__ == "__main__":
    X_pca     = np.load("/home/sashreekkumar/Documents/Projects/malariagen/cache/X_pca.npy")
    valid_ids = np.load("/home/sashreekkumar/Documents/Projects/malariagen/cache/valid_ids.npy", allow_pickle=True)

    df          = pd.read_csv("/home/sashreekkumar/Documents/Projects/malariagen/data/data_csv/sampled_100.csv", header=0)
    id_to_label = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 2]))
    y           = np.array([id_to_label[str(sid)] for sid in valid_ids])

    model, le = train_cnn(X_pca, y, n_epochs=100, batch_size=16, lr=1e-3)