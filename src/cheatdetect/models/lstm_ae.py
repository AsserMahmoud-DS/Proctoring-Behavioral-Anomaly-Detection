"""LSTM autoencoder for sequence anomaly detection (research-only model).

Each sample is a sequence of micro-chunk feature vectors (see the paired
representation in ``cheatdetect.data``). The encoder compresses the sequence into a
latent vector which the decoder uses to reconstruct the full sequence.
Trained on normal data only; per-sample reconstruction MSE is the anomaly
score (higher = more anomalous), matching the sign convention of
:class:`~cheatdetect.models.base.AnomalyDetector`.

This module imports torch and is intentionally **not** re-exported from
``cheatdetect.models`` so the FastAPI image (built without the dev
dependencies) never imports it.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, TensorDataset

from cheatdetect.data import FeaturePreprocessor

from .base import SequenceAnomalyDetector, is_fitted_preprocessor
from .selection import sweep_candidates

LSTM_RECIPES = ("base", "log", "yj", "quantile")


class LSTMAutoencoder(nn.Module):
    """LSTM encoder-decoder for unsupervised anomaly detection.

    Encodes a sequence of micro-chunk feature vectors into a latent
    representation, then reconstructs the sequence. Trained on normal
    data only; reconstruction error serves as the anomaly score.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=1,
        latent_dropout=0.1,
        lstm_dropout=0.0,
    ):
        super().__init__()
        self.seq_len = None
        # Built-in dropout only acts between stacked LSTM layers; the study
        # keeps it at zero so depth alone varies between AE comparisons.
        builtin_dropout = lstm_dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=builtin_dropout,
        )
        self.latent_dropout = nn.Dropout(latent_dropout)
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=builtin_dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        self.seq_len = x.size(1)
        _, (hn, _) = self.encoder(x)
        latent = self.latent_dropout(hn[-1])
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32)


def _train_lstm_ae(
    model: LSTMAutoencoder,
    X_train_arr: np.ndarray,
    X_es_arr: np.ndarray | None = None,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    min_improvement: float = 1e-3,
    grad_clip: float | None = 1.0,
) -> LSTMAutoencoder:
    """Train the LSTM AE on normal-only data with the frozen study controls.

    Args:
        model: The autoencoder to train (modified in place).
        X_train_arr: Normal training sequences.
        X_es_arr: Optional held-out **normal** sequences used for early
            stopping. When ``None``, no early stopping is applied: the model
            trains for the full ``epochs`` budget and the final weights are
            kept (``best_epoch`` is set to ``None``).
        epochs, patience, lr, batch_size: Training controls.
        weight_decay: Adam L2 penalty.
        min_improvement: Relative validation-loss decrease required to reset
            the patience counter (``es_loss < best * (1 - min_improvement)``).
        grad_clip: Global gradient-norm clip; ``None`` disables clipping.

    Diagnostics are stored on the model: ``epochs_trained``, ``best_epoch``,
    ``optimizer_updates``, and ``best_es_loss``.
    """
    train_dataset = TensorDataset(_to_tensor(X_train_arr))
    train_loader = DataLoader(
        train_dataset, batch_size=min(batch_size, len(X_train_arr)), shuffle=True
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    updates = 0

    if X_es_arr is None:
        for _ in range(epochs):
            model.train()
            for (batch,) in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch), batch)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                updates += 1
        model.epochs_trained = epochs
        model.best_epoch = None
        model.optimizer_updates = updates
        model.best_es_loss = None
        return model

    es_tensor = _to_tensor(X_es_arr)
    best_es_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            updates += 1

        model.eval()
        with torch.no_grad():
            es_loss = criterion(model(es_tensor), es_tensor).item()

        improved = best_state is None or es_loss < best_es_loss * (1 - min_improvement)
        if improved:
            best_es_loss = es_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.epochs_trained = epoch + 1
    model.best_epoch = best_epoch
    model.optimizer_updates = updates
    model.best_es_loss = best_es_loss if best_state is not None else None
    return model


def _lstm_anomaly_scores(
    model: LSTMAutoencoder, X_arr: np.ndarray, batch_size: int = 64
) -> np.ndarray:
    """Compute per-sample MSE reconstruction error as anomaly scores."""
    model.eval()
    tensor = _to_tensor(X_arr)
    dataset = TensorDataset(tensor)
    loader = DataLoader(
        dataset, batch_size=min(batch_size, len(X_arr)), shuffle=False
    )
    criterion = nn.MSELoss(reduction="none")

    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            recon = model(batch)
            mse = criterion(recon, batch).mean(dim=(1, 2))
            scores.append(mse.cpu().numpy())
    return np.concatenate(scores)


class LSTMAutoencoderDetector(SequenceAnomalyDetector):
    """Sequence anomaly detector wrapping an LSTM autoencoder.

    Consumes raw 25-source-feature sequences and owns its preprocessing via a
    :class:`~cheatdetect.data.FeaturePreprocessor` (fixed-schema imputation,
    selective transforms, scaling, and direction encoding). The autoencoder
    therefore operates on the encoded feature dimension rather than the raw
    source dimension.
    """

    def __init__(
        self,
        feature_names: list[str],
        input_dim: int,
        seq_len: int,
        recipe: str = "base",
        hidden_dim: int = 16,
        num_layers: int = 1,
        latent_dropout: float = 0.1,
        lstm_dropout: float = 0.0,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
        weight_decay: float = 1e-4,
        min_improvement: float = 1e-3,
        grad_clip: float | None = 1.0,
        random_state: int = 42,
        preprocessor: FeaturePreprocessor | None = None,
    ):
        if recipe not in LSTM_RECIPES:
            raise ValueError(
                f"Unknown LSTM recipe '{recipe}'; expected one of {LSTM_RECIPES}"
            )
        self.feature_names = list(feature_names)
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.recipe = recipe
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dropout = latent_dropout
        self.lstm_dropout = lstm_dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.min_improvement = min_improvement
        self.grad_clip = grad_clip
        self.random_state = random_state
        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else FeaturePreprocessor(recipe, scale=True)
        )

        self.model: LSTMAutoencoder | None = None

    def _preprocess(self, X: np.ndarray, fit: bool) -> np.ndarray:
        n_samples, seq_len, n_features = X.shape
        if n_features != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} source features, got {n_features}"
            )
        flat = pd.DataFrame(
            X.reshape(-1, n_features), columns=self.feature_names
        )
        transformed = (
            self.preprocessor.fit_transform(flat)
            if fit
            else self.preprocessor.transform(flat)
        )
        output = np.asarray(transformed, dtype=np.float32)
        return output.reshape(n_samples, seq_len, output.shape[1])

    def fit(
        self, X: np.ndarray, X_es: np.ndarray | None = None
    ) -> "LSTMAutoencoderDetector":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # An injected preprocessor is already fitted on original training; the
        # autoencoder is trained on (possibly augmented) transformed sequences.
        frozen = is_fitted_preprocessor(self.preprocessor)
        X_scaled = self._preprocess(X, fit=not frozen)
        X_es_scaled = None if X_es is None else self._preprocess(X_es, fit=False)

        self.model = LSTMAutoencoder(
            X_scaled.shape[2],
            self.hidden_dim,
            self.num_layers,
            latent_dropout=self.latent_dropout,
            lstm_dropout=self.lstm_dropout,
        )
        _train_lstm_ae(
            self.model,
            X_scaled,
            X_es_scaled,
            epochs=self.epochs,
            patience=self.patience,
            lr=self.lr,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            min_improvement=self.min_improvement,
            grad_clip=self.grad_clip,
        )
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTMAutoencoderDetector is not fitted")
        X_scaled = self._preprocess(X, fit=False)
        return _lstm_anomaly_scores(self.model, X_scaled, self.batch_size)

    @classmethod
    def grid_search(
        cls,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_es: np.ndarray | None,
        feature_names: list[str],
        param_grid: dict,
        fixed_kwargs: dict | None = None,
        random_state: int = 42,
        preprocessor: FeaturePreprocessor | None = None,
    ) -> tuple["LSTMAutoencoderDetector | None", pd.DataFrame]:
        """Search *param_grid* and return the best detector by validation ROC-AUC.

        Mirrors ``IsolationForestDetector.grid_search``. Gridded parameters
        (``hidden_dim``, ``num_layers``, ``latent_dropout``, ``lr``,
        ``batch_size``) are swept; ``fixed_kwargs`` carries the sequence
        geometry (``input_dim``, ``seq_len``), training controls (``epochs``,
        ``patience``, ``weight_decay``, ``min_improvement``, ``grad_clip``),
        and the preprocessing ``recipe`` when non-default.

        Args:
            X_train, X_val: 3D training / validation sequences.
            y_val: Binary validation labels (1 = anomalous). Used only for
                model selection (ROC-AUC).
            X_es: Optional held-out **normal** sequences for early stopping.
            feature_names: Column names for the feature axis.
            param_grid: Dict of constructor params → list of candidates.
            fixed_kwargs: Constructor params held constant across the grid.
            random_state: Seed for reproducibility.
            preprocessor: Optional pre-fitted processor reused across candidates.

        Returns:
            ``(best_detector, results_df)`` sorted by ROC-AUC, with PR-AUC and
            per-candidate status reported.
        """
        fixed_kwargs = dict(fixed_kwargs or {})

        def fit_and_score(params: dict):
            detector = cls(
                feature_names=feature_names,
                random_state=random_state,
                preprocessor=preprocessor,
                **fixed_kwargs,
                **params,
            )
            detector.fit(X_train, X_es=X_es)
            model = detector.model
            diagnostics = {
                "epochs_trained": getattr(model, "epochs_trained", None),
                "best_epoch": getattr(model, "best_epoch", None),
                "optimizer_updates": getattr(model, "optimizer_updates", None),
                "best_es_loss": getattr(model, "best_es_loss", None),
            }
            return detector, detector.decision_function(X_val), diagnostics

        return sweep_candidates(ParameterGrid(param_grid), fit_and_score, y_val)
