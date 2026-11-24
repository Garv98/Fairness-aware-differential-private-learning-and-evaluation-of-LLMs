import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

from src.models import CreditRiskMLP


def train_dp_mlp(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    learning_rate=0.001,
    noise_multiplier=1.0,
    max_grad_norm=1.0
):

    device = torch.device("cpu")

    # Convert training data into tensors
    X_train = torch.tensor(
        X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train),
        dtype=torch.float32
    )

    y_train = torch.tensor(
        np.asarray(y_train),
        dtype=torch.float32
    ).reshape(-1, 1)

    # Create dataset and loader
    dataset = TensorDataset(X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Initialize MLP
    model = CreditRiskMLP(
        input_dim=X_train.shape[1]
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Initialize privacy engine
    privacy_engine = PrivacyEngine(accountant="rdp")

    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True
    )

    # Training loop
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
            f"- DP Loss: {avg_loss:.4f}"
        )

    # Calculate privacy budget
    epsilon = privacy_engine.get_epsilon(delta=1e-5)

    print("\n===== PRIVACY ACCOUNTING =====")
    print(f"Noise multiplier : {noise_multiplier}")
    print(f"Max grad norm  : {max_grad_norm}")
    print(f"Privacy epsilon : {epsilon:.4f}")
    print("Delta          : 1e-5")

    return model, epsilon