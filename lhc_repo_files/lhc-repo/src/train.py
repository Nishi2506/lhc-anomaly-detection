import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train(model, X_train, X_val, epochs=60, batch_size=512,
          lr=1e-3, save_path="models/jet_autoencoder_best.pt", device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    loader    = DataLoader(TensorDataset(X_train_t), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    history  = {"train": [], "val": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(X_val_t.to(device)),
                X_val_t.to(device)
            ).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

    print(f"\nbest val loss: {best_val:.6f}")
    print(f"model saved to {save_path}")
    return history
