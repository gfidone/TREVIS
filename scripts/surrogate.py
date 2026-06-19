import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPPredictor(nn.Module):
    def __init__(
        self,
        input_dim=60,
        hidden_dim=128,
        lr=1e-3,
        weight_decay=1e-5,
        device=None,
    ):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.mseloss = nn.MSELoss(reduction="sum")
        self.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def forward(self, x):
        return self.net(x)

    def fit(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        epochs=500,
        batch_size=64,
        verbose=True,
    ):
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        if X_test is not None and y_test is not None:
            X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device)

            if y_test.ndim == 1:
                y_test = y_test.unsqueeze(1)

        for epoch in range(epochs):
            self.train()

            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                pred = self.forward(xb)
                loss = self.mseloss(pred, yb) / xb.shape[0]
                loss.backward()
                self.optimizer.step()

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                train_metrics = self.metrics(X_train, y_train)

                msg = (
                    f"Epoch {epoch} | "
                    f"Train MSE: {train_metrics['mse']:.4f} | "
                    f"Train RMSE: {train_metrics['rmse']:.4f} | "
                    f"Train Pearson: {train_metrics['pearson']:.4f}"
                )

                if X_test is not None and y_test is not None:
                    test_metrics = self.metrics(X_test, y_test)
                    msg += (
                        f" | Test MSE: {test_metrics['mse']:.4f} | "
                        f"Test RMSE: {test_metrics['rmse']:.4f} | "
                        f"Test Pearson: {test_metrics['pearson']:.4f}"
                    )

                print(msg)

        return self

    def predict(self, X):
        self.eval()

        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.detach().float().to(self.device)

        with torch.no_grad():
            pred = self.forward(X)

        return pred.cpu().numpy()

    def metrics(self, X, y):
        self.eval()

        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.detach().float().to(self.device)

        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.detach().float().to(self.device)

        if y.ndim == 1:
            y = y.unsqueeze(1)

        with torch.no_grad():
            pred = self.forward(X)
            mse = torch.mean((pred - y) ** 2)
            rmse = torch.sqrt(mse)

            pred_flat = pred.flatten()
            y_flat = y.flatten()

            pred_centered = pred_flat - pred_flat.mean()
            y_centered = y_flat - y_flat.mean()

            pearson = torch.sum(pred_centered * y_centered) / (
                torch.sqrt(torch.sum(pred_centered ** 2))
                * torch.sqrt(torch.sum(y_centered ** 2))
                + 1e-8
            )

        return {
            "mse": mse.item(),
            "rmse": rmse.item(),
            "pearson": pearson.item(),
        }
