import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


class SNP_CNN(nn.Module):
    def __init__(self, n_components: int, n_classes: int):
        super(SNP_CNN, self).__init__()

        self.conv = nn.Sequential(
            # treat 50 PCA components as a 1D sequence
            # input: (batch, 1, n_components)
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)  # global average pooling → (batch, 128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),            # (batch, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x

def train_cnn(X_pca: np.ndarray, y: np.ndarray,
              n_epochs: int = 100,
              batch_size: int = 16,
              lr: float = 1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # encode labels
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")

    # convert to tensors
    # reshape to (n_samples, 1, n_components) for Conv1d
    X_tensor = torch.tensor(X_pca, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    dataset    = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    model = SNP_CNN(n_components=X_pca.shape[1], n_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss  = 0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * len(y_batch)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += len(y_batch)

        scheduler.step()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")

    # save
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