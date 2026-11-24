import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader


# ==============================
# 1. Logistic Regression
# ==============================

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ==============================
# 2. Neural Network Architecture
# ==============================

class CreditRiskMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ==============================
# 3. Neural Network Training
# ==============================

def train_mlp(X_train, y_train, epochs=20, batch_size=64, learning_rate=0.001):

    device = torch.device("cpu")

    X_train = torch.tensor(
        X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train),
        dtype=torch.float32
    )

    y_train = torch.tensor(
        np.asarray(y_train),
        dtype=torch.float32
    ).reshape(-1, 1)

    dataset = TensorDataset(X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = CreditRiskMLP(
        input_dim=X_train.shape[1]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"- Loss: {avg_loss:.4f}"
        )

    return model
def train_weighted_mlp(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    learning_rate=0.001
):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # Convert data to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)

    weight_0 = total_samples / (2 * class_counts[0])
    weight_1 = total_samples / (2 * class_counts[1])

    print(f"Class 0 weight: {weight_0:.4f}")
    print(f"Class 1 weight: {weight_1:.4f}")

    # Model and loss
    model = CreditRiskMLP(input_dim=X_train.shape[1])
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight_1 / weight_0], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Weighted Loss: {avg_loss:.4f}")

    return model