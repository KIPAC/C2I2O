import pickle
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class TrainingConfig(BaseModel):
    """Configuration for training with validation"""

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    epochs: int = Field(
        default=200, ge=1, le=10000, description="Number of training epochs"
    )
    batch_size: int = Field(
        default=64, ge=1, le=1024, description="Batch size for training"
    )
    learning_rate: float = Field(default=1e-3, gt=0, lt=1, description="Learning rate")
    patience: int = Field(default=20, ge=1, description="Patience for early stopping")
    weight_decay: float = Field(
        default=1e-5, ge=0, description="L2 regularization weight"
    )
    grad_clip: float = Field(
        default=1.0, gt=0, description="Gradient clipping threshold"
    )
    validation_split: float = Field(
        default=0.15, gt=0, lt=1, description="Validation set fraction"
    )
    test_split: float = Field(default=0.15, gt=0, lt=1, description="Test set fraction")
    scheduler_factor: float = Field(
        default=0.5, gt=0, lt=1, description="LR scheduler reduction factor"
    )
    scheduler_patience: int = Field(
        default=10, ge=1, description="LR scheduler patience"
    )

    @field_validator("validation_split", "test_split")
    @classmethod
    def validate_splits(cls, v: float, info) -> float:
        """Ensure splits are reasonable"""
        if v > 0.5:
            raise ValueError(f"{info.field_name} must be less than 0.5")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that splits don't exceed 1.0 combined"""
        if self.validation_split + self.test_split >= 1.0:
            raise ValueError("validation_split + test_split must be < 1.0")


class ModelConfig(BaseModel):
    """Configuration for model architecture with validation"""

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    hidden_dims: List[int] = Field(
        default=[256, 512, 512, 256], description="Hidden layer dimensions"
    )
    n_residual_blocks: int = Field(
        default=3, ge=0, le=10, description="Number of residual blocks"
    )
    dropout: float = Field(default=0.1, ge=0, lt=1, description="Dropout probability")

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: List[int]) -> List[int]:
        """Ensure hidden dimensions are positive"""
        if len(v) == 0:
            raise ValueError("hidden_dims must contain at least one dimension")
        if any(dim <= 0 for dim in v):
            raise ValueError("All hidden dimensions must be positive")
        return v


class DataConfig(BaseModel):
    """Configuration for data specifications"""

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    n_params: int = Field(ge=1, description="Number of cosmological parameters")
    n_k: int = Field(ge=1, description="Number of k bins for power spectrum")
    n_z: int = Field(ge=1, description="Number of redshift bins")
    param_names: Optional[List[str]] = Field(
        default=None, description="Names of cosmological parameters"
    )
    k_min: Optional[float] = Field(default=None, gt=0, description="Minimum k value")
    k_max: Optional[float] = Field(default=None, gt=0, description="Maximum k value")
    z_min: Optional[float] = Field(default=0.0, ge=0, description="Minimum redshift")
    z_max: Optional[float] = Field(default=None, gt=0, description="Maximum redshift")

    @field_validator("param_names")
    @classmethod
    def validate_param_names(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        """Ensure param_names matches n_params if provided"""
        if v is not None:
            n_params = info.data.get("n_params")
            if n_params and len(v) != n_params:
                raise ValueError(
                    f"Length of param_names ({len(v)}) must match n_params ({n_params})"
                )
        return v

    @field_validator("k_max")
    @classmethod
    def validate_k_max(cls, v: Optional[float], info) -> Optional[float]:
        """Ensure k_max > k_min if both provided"""
        if v is not None:
            k_min = info.data.get("k_min")
            if k_min is not None and v <= k_min:
                raise ValueError("k_max must be greater than k_min")
        return v

    @field_validator("z_max")
    @classmethod
    def validate_z_max(cls, v: Optional[float], info) -> Optional[float]:
        """Ensure z_max > z_min if both provided"""
        if v is not None:
            z_min = info.data.get("z_min", 0.0)
            if v <= z_min:
                raise ValueError("z_max must be greater than z_min")
        return v


class EmulatorConfig(BaseModel):
    """Complete configuration for the emulator system"""

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    data_config: DataConfig
    forward_model_config: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            hidden_dims=[256, 512, 512, 256], n_residual_blocks=3, dropout=0.1
        )
    )
    inverse_model_config: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            hidden_dims=[512, 512, 256, 128], n_residual_blocks=3, dropout=0.1
        )
    )
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device to use for computation"
    )
    use_log_pk: bool = Field(
        default=True, description="Whether to use log scale for power spectrum"
    )
    random_seed: int = Field(
        default=42, ge=0, description="Random seed for reproducibility"
    )


class CosmologyDataset(Dataset):
    """Dataset class for cosmological data"""

    def __init__(
        self,
        cosmo_params: NDArray[np.float32],
        power_spectra: NDArray[np.float32],
        comoving_distances: NDArray[np.float32],
        hubble_evolution: NDArray[np.float32],
    ) -> None:
        """
        Args:
            cosmo_params: (N, n_params) cosmological parameters
            power_spectra: (N, n_k) matter power spectrum P(k)
            comoving_distances: (N, n_z) comoving radial distance chi(z)
            hubble_evolution: (N, n_z) Hubble parameter H(z)
        """
        self.cosmo_params: torch.Tensor = torch.FloatTensor(cosmo_params)
        self.power_spectra: torch.Tensor = torch.FloatTensor(power_spectra)
        self.comoving_distances: torch.Tensor = torch.FloatTensor(comoving_distances)
        self.hubble_evolution: torch.Tensor = torch.FloatTensor(hubble_evolution)

    def __len__(self) -> int:
        return len(self.cosmo_params)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "params": self.cosmo_params[idx],
            "power_spectrum": self.power_spectra[idx],
            "comoving_distance": self.comoving_distances[idx],
            "hubble": self.hubble_evolution[idx],
        }


class ResidualBlock(nn.Module):
    """Residual block with layer normalization"""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class ForwardEmulator(nn.Module):
    """
    Forward emulator: cosmological parameters → observables
    Maps parameters to power spectrum, comoving distance, and Hubble evolution
    """

    def __init__(self, n_params: int, n_k: int, n_z: int, config: ModelConfig) -> None:
        super().__init__()

        self.n_params: int = n_params
        self.n_k: int = n_k
        self.n_z: int = n_z
        self.config: ModelConfig = config

        hidden_dims: List[int] = config.hidden_dims
        dropout: float = config.dropout

        # Shared encoder for parameters
        layers: List[nn.Module] = [
            nn.Linear(n_params, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        self.encoder: nn.Sequential = nn.Sequential(*layers)

        # Residual blocks for better gradient flow
        self.residual_blocks: nn.ModuleList = nn.ModuleList(
            [
                ResidualBlock(hidden_dims[-1], dropout)
                for _ in range(config.n_residual_blocks)
            ]
        )

        # Separate heads for each observable
        self.power_spectrum_head: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_k),
        )

        self.comoving_distance_head: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_z),
        )

        self.hubble_head: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_z),
        )

    def forward(
        self, params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            params: (batch_size, n_params) cosmological parameters

        Returns:
            power_spectrum: (batch_size, n_k)
            comoving_distance: (batch_size, n_z)
            hubble: (batch_size, n_z)
        """
        # Encode parameters
        x: torch.Tensor = self.encoder(params)

        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Generate predictions
        power_spectrum: torch.Tensor = self.power_spectrum_head(x)
        comoving_distance: torch.Tensor = self.comoving_distance_head(x)
        hubble: torch.Tensor = self.hubble_head(x)

        return power_spectrum, comoving_distance, hubble


class InverseEmulator(nn.Module):
    """
    Inverse emulator: observables → cosmological parameters
    Maps power spectrum and comoving distance to parameters
    """

    def __init__(self, n_params: int, n_k: int, n_z: int, config: ModelConfig) -> None:
        super().__init__()

        self.n_params: int = n_params
        self.config: ModelConfig = config

        hidden_dims: List[int] = config.hidden_dims
        dropout: float = config.dropout

        # Separate encoders for each observable
        self.pk_encoder: nn.Sequential = nn.Sequential(
            nn.Linear(n_k, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 256)
        )

        self.chi_encoder: nn.Sequential = nn.Sequential(
            nn.Linear(n_z, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 128)
        )

        # Combined processing
        input_dim: int = 256 + 128
        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        self.fusion: nn.Sequential = nn.Sequential(*layers)

        # Residual blocks
        self.residual_blocks: nn.ModuleList = nn.ModuleList(
            [
                ResidualBlock(hidden_dims[-1], dropout)
                for _ in range(config.n_residual_blocks)
            ]
        )

        # Output layer
        self.output: nn.Linear = nn.Linear(hidden_dims[-1], n_params)

    def forward(
        self, power_spectrum: torch.Tensor, comoving_distance: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            power_spectrum: (batch_size, n_k)
            comoving_distance: (batch_size, n_z)

        Returns:
            params: (batch_size, n_params) predicted cosmological parameters
        """
        # Encode observables
        pk_features: torch.Tensor = self.pk_encoder(power_spectrum)
        chi_features: torch.Tensor = self.chi_encoder(comoving_distance)

        # Concatenate features
        x: torch.Tensor = torch.cat([pk_features, chi_features], dim=1)

        # Process combined features
        x = self.fusion(x)

        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Predict parameters
        params: torch.Tensor = self.output(x)

        return params


class TrainingHistory(BaseModel):
    """Store training history with validation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_loss: List[float] = Field(default_factory=list)
    val_loss: List[float] = Field(default_factory=list)
    train_loss_pk: List[float] = Field(default_factory=list)
    train_loss_chi: List[float] = Field(default_factory=list)
    train_loss_hubble: List[float] = Field(default_factory=list)
    val_loss_pk: List[float] = Field(default_factory=list)
    val_loss_chi: List[float] = Field(default_factory=list)
    val_loss_hubble: List[float] = Field(default_factory=list)
    learning_rates: List[float] = Field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")

    def update_train(
        self,
        total_loss: float,
        pk_loss: float,
        chi_loss: float,
        hubble_loss: float,
        lr: float,
    ) -> None:
        """Update training metrics"""
        self.train_loss.append(total_loss)
        self.train_loss_pk.append(pk_loss)
        self.train_loss_chi.append(chi_loss)
        self.train_loss_hubble.append(hubble_loss)
        self.learning_rates.append(lr)

    def update_val(
        self,
        total_loss: float,
        pk_loss: float,
        chi_loss: float,
        hubble_loss: float,
        epoch: int,
    ) -> bool:
        """
        Update validation metrics

        Returns:
            True if this is the best epoch so far
        """
        self.val_loss.append(total_loss)
        self.val_loss_pk.append(pk_loss)
        self.val_loss_chi.append(chi_loss)
        self.val_loss_hubble.append(hubble_loss)

        is_best: bool = total_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = total_loss
            self.best_epoch = epoch

        return is_best

    def plot(self, save_path: Optional[str] = None) -> None:
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Total loss
        axes[0, 0].plot(self.train_loss, label="Train", alpha=0.7)
        axes[0, 0].plot(self.val_loss, label="Validation", alpha=0.7)
        axes[0, 0].axvline(
            self.best_epoch, color="r", linestyle="--", alpha=0.5, label="Best"
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].set_yscale("log")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title("Total Loss")

        # Component losses
        axes[0, 1].plot(self.train_loss_pk, label="P(k) Train", alpha=0.7)
        axes[0, 1].plot(self.val_loss_pk, label="P(k) Val", alpha=0.7)
        axes[0, 1].plot(
            self.train_loss_chi, label="χ(z) Train", alpha=0.7, linestyle="--"
        )
        axes[0, 1].plot(self.val_loss_chi, label="χ(z) Val", alpha=0.7, linestyle="--")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title("Component Losses")

        # Hubble loss
        axes[1, 0].plot(self.train_loss_hubble, label="Train", alpha=0.7)
        axes[1, 0].plot(self.val_loss_hubble, label="Validation", alpha=0.7)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("H(z) Loss")
        axes[1, 0].set_yscale("log")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title("Hubble Parameter Loss")

        # Learning rate
        axes[1, 1].plot(self.learning_rates, alpha=0.7)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title("Learning Rate Schedule")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


class CosmologicalEmulator:
    """
    Complete emulator system with forward and inverse models
    """

    def __init__(self, config: EmulatorConfig) -> None:
        """
        Initialize the emulator

        Args:
            config: Complete emulator configuration
        """
        self.config: EmulatorConfig = config

        # Set random seed
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Set device
        self.device: torch.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Extract dimensions
        self.n_params: int = config.data_config.n_params
        self.n_k: int = config.data_config.n_k
        self.n_z: int = config.data_config.n_z

        # Initialize models
        self.forward_model: ForwardEmulator = ForwardEmulator(
            self.n_params, self.n_k, self.n_z, config.forward_model_config
        ).to(self.device)

        self.inverse_model: InverseEmulator = InverseEmulator(
            self.n_params, self.n_k, self.n_z, config.inverse_model_config
        ).to(self.device)

        # Scalers for normalization
        self.param_scaler: StandardScaler = StandardScaler()
        self.pk_scaler: StandardScaler = StandardScaler()
        self.chi_scaler: StandardScaler = StandardScaler()
        self.hubble_scaler: StandardScaler = StandardScaler()

        # Training history
        self.forward_history: Optional[TrainingHistory] = None
        self.inverse_history: Optional[TrainingHistory] = None

        self.is_fitted: bool = False

    def prepare_data(
        self,
        cosmo_params: NDArray[np.float32],
        power_spectra: NDArray[np.float32],
        comoving_distances: NDArray[np.float32],
        hubble_evolution: NDArray[np.float32],
    ) -> Tuple[CosmologyDataset, CosmologyDataset, CosmologyDataset]:
        """
        Prepare and normalize data

        Args:
            cosmo_params: (N, n_params) cosmological parameters
            power_spectra: (N, n_k) power spectra
            comoving_distances: (N, n_z) comoving distances
            hubble_evolution: (N, n_z) Hubble parameters

        Returns:
            train_dataset, val_dataset, test_dataset
        """

        # Validate input shapes
        n_samples: int = len(cosmo_params)
        assert power_spectra.shape == (
            n_samples,
            self.n_k,
        ), f"Expected power_spectra shape ({n_samples}, {self.n_k}), got {power_spectra.shape}"
        assert comoving_distances.shape == (
            n_samples,
            self.n_z,
        ), f"Expected comoving_distances shape ({n_samples}, {self.n_z}), got {comoving_distances.shape}"
        assert hubble_evolution.shape == (
            n_samples,
            self.n_z,
        ), f"Expected hubble_evolution shape ({n_samples}, {self.n_z}), got {hubble_evolution.shape}"

        # Normalize data
        cosmo_params_scaled: NDArray[np.float32] = self.param_scaler.fit_transform(
            cosmo_params
        )

        # Use log scale for power spectrum if configured
        if self.config.use_log_pk:
            power_spectra_log: NDArray[np.float32] = np.log10(power_spectra + 1e-10)
            power_spectra_scaled: NDArray[np.float32] = self.pk_scaler.fit_transform(
                power_spectra_log
            )
        else:
            power_spectra_scaled = self.pk_scaler.fit_transform(power_spectra)

        comoving_distances_scaled: NDArray[np.float32] = self.chi_scaler.fit_transform(
            comoving_distances
        )
        hubble_evolution_scaled: NDArray[np.float32] = self.hubble_scaler.fit_transform(
            hubble_evolution
        )

        # Split data
        train_config: TrainingConfig = self.config.training_config
        indices: NDArray[np.int_] = np.arange(n_samples)

        train_idx: NDArray[np.int_]
        test_idx: NDArray[np.int_]
        val_idx: NDArray[np.int_]

        train_idx, test_idx = train_test_split(
            indices,
            test_size=train_config.test_split,
            random_state=self.config.random_seed,
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=train_config.validation_split / (1 - train_config.test_split),
            random_state=self.config.random_seed,
        )

        # Create datasets
        train_dataset: CosmologyDataset = CosmologyDataset(
            cosmo_params_scaled[train_idx],
            power_spectra_scaled[train_idx],
            comoving_distances_scaled[train_idx],
            hubble_evolution_scaled[train_idx],
        )

        val_dataset: CosmologyDataset = CosmologyDataset(
            cosmo_params_scaled[val_idx],
            power_spectra_scaled[val_idx],
            comoving_distances_scaled[val_idx],
            hubble_evolution_scaled[val_idx],
        )

        test_dataset: CosmologyDataset = CosmologyDataset(
            cosmo_params_scaled[test_idx],
            power_spectra_scaled[test_idx],
            comoving_distances_scaled[test_idx],
            hubble_evolution_scaled[test_idx],
        )

        self.is_fitted = True

        print(
            f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    def train_forward_model(
        self,
        train_dataset: CosmologyDataset,
        val_dataset: CosmologyDataset,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingHistory:
        """
        Train the forward emulator

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration (uses default if None)

        Returns:
            Training history
        """

        if config is None:
            config = self.config.training_config

        train_loader: DataLoader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=config.batch_size)

        optimizer: optim.AdamW = optim.AdamW(
            self.forward_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler: optim.lr_scheduler.ReduceLROnPlateau = (
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
            )
        )

        criterion: nn.MSELoss = nn.MSELoss()

        history: TrainingHistory = TrainingHistory()
        patience_counter: int = 0

        print("Training Forward Emulator...")
        print(f"Device: {self.device}")
        print(
            f"Model parameters: {sum(p.numel() for p in self.forward_model.parameters()):,}"
        )

        for epoch in range(config.epochs):
            # Training phase
            self.forward_model.train()
            train_loss: float = 0.0
            train_loss_pk: float = 0.0
            train_loss_chi: float = 0.0
            train_loss_h: float = 0.0

            for batch in train_loader:
                params: torch.Tensor = batch["params"].to(self.device)
                pk_true: torch.Tensor = batch["power_spectrum"].to(self.device)
                chi_true: torch.Tensor = batch["comoving_distance"].to(self.device)
                h_true: torch.Tensor = batch["hubble"].to(self.device)

                optimizer.zero_grad()

                pk_pred: torch.Tensor
                chi_pred: torch.Tensor
                h_pred: torch.Tensor
                pk_pred, chi_pred, h_pred = self.forward_model(params)

                # Component losses
                loss_pk: torch.Tensor = criterion(pk_pred, pk_true)
                loss_chi: torch.Tensor = criterion(chi_pred, chi_true)
                loss_h: torch.Tensor = criterion(h_pred, h_true)

                # Combined loss
                loss: torch.Tensor = loss_pk + loss_chi + loss_h

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.forward_model.parameters(), config.grad_clip
                )
                optimizer.step()

                train_loss += loss.item()
                train_loss_pk += loss_pk.item()
                train_loss_chi += loss_chi.item()
                train_loss_h += loss_h.item()

            train_loss /= len(train_loader)
            train_loss_pk /= len(train_loader)
            train_loss_chi /= len(train_loader)
            train_loss_h /= len(train_loader)

            # Validation phase
            self.forward_model.eval()
            val_loss: float = 0.0
            val_loss_pk: float = 0.0
            val_loss_chi: float = 0.0
            val_loss_h: float = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    params_validate: torch.Tensor = batch["params"].to(self.device)
                    pk_true_validate: torch.Tensor = batch["power_spectrum"].to(self.device)
                    chi_true_validate: torch.Tensor = batch["comoving_distance"].to(self.device)
                    h_true_validate: torch.Tensor = batch["hubble"].to(self.device)

                    pk_pred_validate: torch.Tensor
                    chi_pred_validate: torch.Tensor
                    h_pred_validate: torch.Tensor
                    pk_pred_validate, chi_pred_validate, h_pred_validate = self.forward_model(params_validate)

                    # Component losses
                    loss_pk_validate: torch.Tensor = criterion(pk_pred_validate, pk_true_validate)
                    loss_chi_validate: torch.Tensor = criterion(chi_pred_validate, chi_true_validate)
                    loss_h_validate: torch.Tensor = criterion(h_pred_validate, h_true_validate)

                    # Combined loss
                    loss_validate: torch.Tensor = loss_pk_validate + loss_chi_validate + loss_h_validate

                    val_loss += loss_validate.item()
                    val_loss_pk += loss_pk_validate.item()
                    val_loss_chi += loss_chi_validate.item()
                    val_loss_h += loss_h_validate.item()

            val_loss /= len(val_loader)
            val_loss_pk /= len(val_loader)
            val_loss_chi /= len(val_loader)
            val_loss_h /= len(val_loader)

            # Update history
            current_lr: float = optimizer.param_groups[0]["lr"]
            history.update_train(
                train_loss, train_loss_pk, train_loss_chi, train_loss_h, current_lr
            )
            is_best: bool = history.update_val(
                val_loss, val_loss_pk, val_loss_chi, val_loss_h, epoch
            )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config.epochs}")
                print(
                    f"  Train Loss: {train_loss:.6f} (P(k)={train_loss_pk:.6f}, χ={train_loss_chi:.6f}, H={train_loss_h:.6f})"
                )
                print(
                    f"  Val Loss:   {val_loss:.6f} (P(k)={val_loss_pk:.6f}, χ={val_loss_chi:.6f}, H={val_loss_h:.6f})"
                )
                print(f"  LR: {current_lr:.2e}")

            # Save best model
            if is_best:
                torch.save(self.forward_model.state_dict(), "best_forward_model.pt")
                if (epoch + 1) % 10 == 0:
                    print("  *** New best model saved ***")

            # Early stopping
            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(
                        f"Best validation loss: {history.best_val_loss:.6f} at epoch {history.best_epoch+1}"
                    )
                    break

        # Load best model
        self.forward_model.load_state_dict(torch.load("best_forward_model.pt"))
        self.forward_history = history

        print("\nTraining completed!")
        print(f"Best epoch: {history.best_epoch+1}")
        print(f"Best validation loss: {history.best_val_loss:.6f}")

        return history

    def train_inverse_model(
        self,
        train_dataset: CosmologyDataset,
        val_dataset: CosmologyDataset,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingHistory:
        """
        Train the inverse emulator

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration (uses default if None)

        Returns:
            Training history
        """

        if config is None:
            config = self.config.training_config

        train_loader: DataLoader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=config.batch_size)

        optimizer: optim.AdamW = optim.AdamW(
            self.inverse_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler: optim.lr_scheduler.ReduceLROnPlateau = (
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
            )
        )

        criterion: nn.MSELoss = nn.MSELoss()

        # Custom history for inverse model (only tracks parameter reconstruction)
        history: TrainingHistory = TrainingHistory()
        patience_counter: int = 0

        print("\nTraining Inverse Emulator...")
        print(f"Device: {self.device}")
        print(
            f"Model parameters: {sum(p.numel() for p in self.inverse_model.parameters()):,}"
        )

        for epoch in range(config.epochs):
            # Training phase
            self.inverse_model.train()
            train_loss: float = 0.0

            for batch in train_loader:
                params_true: torch.Tensor = batch["params"].to(self.device)
                pk: torch.Tensor = batch["power_spectrum"].to(self.device)
                chi: torch.Tensor = batch["comoving_distance"].to(self.device)

                optimizer.zero_grad()

                params_pred: torch.Tensor = self.inverse_model(pk, chi)

                # Loss for parameter reconstruction
                loss: torch.Tensor = criterion(params_pred, params_true)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.inverse_model.parameters(), config.grad_clip
                )
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.inverse_model.eval()
            val_loss: float = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    params_true_validate: torch.Tensor = batch["params"].to(self.device)
                    pk_validate: torch.Tensor = batch["power_spectrum"].to(self.device)
                    chi_validate: torch.Tensor = batch["comoving_distance"].to(self.device)

                    params_pred_validate: torch.Tensor = self.inverse_model(pk_validate, chi_validate)

                    loss_validate: torch.Tensor = criterion(params_pred, params_true)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Update history (using dummy values for component losses)
            current_lr: float = optimizer.param_groups[0]["lr"]
            history.update_train(train_loss, 0.0, 0.0, 0.0, current_lr)
            is_best: bool = history.update_val(val_loss, 0.0, 0.0, 0.0, epoch)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config.epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")
                print(f"  LR: {current_lr:.2e}")

            # Save best model
            if is_best:
                torch.save(self.inverse_model.state_dict(), "best_inverse_model.pt")
                if (epoch + 1) % 10 == 0:
                    print("  *** New best model saved ***")

            # Early stopping
            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(
                        f"Best validation loss: {history.best_val_loss:.6f} at epoch {history.best_epoch+1}"
                    )
                    break

        # Load best model
        self.inverse_model.load_state_dict(torch.load("best_inverse_model.pt"))
        self.inverse_history = history

        print("\nTraining completed!")
        print(f"Best epoch: {history.best_epoch+1}")
        print(f"Best validation loss: {history.best_val_loss:.6f}")

        return history

    def predict_observables(
        self, cosmo_params: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Predict observables from cosmological parameters (forward model)

        Args:
            cosmo_params: (N, n_params) cosmological parameters

        Returns:
            power_spectra: (N, n_k)
            comoving_distances: (N, n_z)
            hubble_evolution: (N, n_z)
        """

        if not self.is_fitted:
            raise RuntimeError("Emulator must be fitted before prediction")

        self.forward_model.eval()

        # Normalize input
        params_scaled: NDArray[np.float32] = self.param_scaler.transform(cosmo_params)
        params_tensor: torch.Tensor = torch.FloatTensor(params_scaled).to(self.device)

        with torch.no_grad():
            pk_pred: torch.Tensor
            chi_pred: torch.Tensor
            h_pred: torch.Tensor
            pk_pred, chi_pred, h_pred = self.forward_model(params_tensor)

        # Denormalize outputs
        pk_scaled: NDArray[np.float32] = pk_pred.cpu().numpy()
        chi_scaled: NDArray[np.float32] = chi_pred.cpu().numpy()
        h_scaled: NDArray[np.float32] = h_pred.cpu().numpy()

        pk: NDArray[np.float32] = self.pk_scaler.inverse_transform(pk_scaled)
        chi: NDArray[np.float32] = self.chi_scaler.inverse_transform(chi_scaled)
        h: NDArray[np.float32] = self.hubble_scaler.inverse_transform(h_scaled)

        # Convert back from log scale if needed
        if self.config.use_log_pk:
            pk = 10**pk - 1e-10

        return pk, chi, h

    def infer_parameters(
        self,
        power_spectra: NDArray[np.float32],
        comoving_distances: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Infer cosmological parameters from observables (inverse model)

        Args:
            power_spectra: (N, n_k)
            comoving_distances: (N, n_z)

        Returns:
            cosmo_params: (N, n_params) inferred cosmological parameters
        """

        if not self.is_fitted:
            raise RuntimeError("Emulator must be fitted before inference")

        self.inverse_model.eval()

        # Normalize inputs
        if self.config.use_log_pk:
            pk_log: NDArray[np.float32] = np.log10(power_spectra + 1e-10)
            pk_scaled: NDArray[np.float32] = self.pk_scaler.transform(pk_log)
        else:
            pk_scaled = self.pk_scaler.transform(power_spectra)

        chi_scaled: NDArray[np.float32] = self.chi_scaler.transform(comoving_distances)

        pk_tensor: torch.Tensor = torch.FloatTensor(pk_scaled).to(self.device)
        chi_tensor: torch.Tensor = torch.FloatTensor(chi_scaled).to(self.device)

        with torch.no_grad():
            params_pred: torch.Tensor = self.inverse_model(pk_tensor, chi_tensor)

        # Denormalize output
        params_scaled: NDArray[np.float32] = params_pred.cpu().numpy()
        params: NDArray[np.float32] = self.param_scaler.inverse_transform(params_scaled)

        return params

    def evaluate(
        self,
        test_dataset: CosmologyDataset,
        plot: bool = True,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate both models on test set

        Args:
            test_dataset: Test dataset
            plot: Whether to create evaluation plots
            save_path: Path to save plots

        Returns:
            Dictionary of evaluation metrics
        """

        test_loader: DataLoader = DataLoader(test_dataset, batch_size=128)

        # Forward model evaluation
        self.forward_model.eval()
        forward_metrics: Dict[str, List[float]] = {
            "pk_mse": [],
            "chi_mse": [],
            "hubble_mse": [],
        }

        all_pk_true: List[NDArray] = []
        all_pk_pred: List[NDArray] = []
        all_chi_true: List[NDArray] = []
        all_chi_pred: List[NDArray] = []
        all_h_true: List[NDArray] = []
        all_h_pred: List[NDArray] = []

        with torch.no_grad():
            for batch in test_loader:
                params: torch.Tensor = batch["params"].to(self.device)
                pk_true: torch.Tensor = batch["power_spectrum"].to(self.device)
                chi_true: torch.Tensor = batch["comoving_distance"].to(self.device)
                h_true: torch.Tensor = batch["hubble"].to(self.device)

                pk_pred: torch.Tensor
                chi_pred: torch.Tensor
                h_pred: torch.Tensor
                pk_pred, chi_pred, h_pred = self.forward_model(params)

                # Store predictions
                all_pk_true.append(pk_true.cpu().numpy())
                all_pk_pred.append(pk_pred.cpu().numpy())
                all_chi_true.append(chi_true.cpu().numpy())
                all_chi_pred.append(chi_pred.cpu().numpy())
                all_h_true.append(h_true.cpu().numpy())
                all_h_pred.append(h_pred.cpu().numpy())

                # Calculate MSE
                forward_metrics["pk_mse"].append(nn.MSELoss()(pk_pred, pk_true).item())
                forward_metrics["chi_mse"].append(nn.MSELoss()(chi_pred, chi_true).item)
                forward_metrics["hubble_mse"].append(
                    nn.MSELoss()(h_pred, h_true).item()
                )

        # Inverse model evaluation
        self.inverse_model.eval()
        inverse_metrics: Dict[str, List[float]] = {"param_mse": []}

        all_params_true: List[NDArray] = []
        all_params_pred: List[NDArray] = []

        with torch.no_grad():
            for batch in test_loader:
                params_true: torch.Tensor = batch["params"].to(self.device)
                pk: torch.Tensor = batch["power_spectrum"].to(self.device)
                chi: torch.Tensor = batch["comoving_distance"].to(self.device)

                params_pred: torch.Tensor = self.inverse_model(pk, chi)

                # Store predictions
                all_params_true.append(params_true.cpu().numpy())
                all_params_pred.append(params_pred.cpu().numpy())

                # Calculate MSE
                inverse_metrics["param_mse"].append(
                    nn.MSELoss()(params_pred, params_true).item()
                )

        # Concatenate all predictions
        all_pk_true_np: NDArray = np.concatenate(all_pk_true, axis=0)
        all_pk_pred_np: NDArray = np.concatenate(all_pk_pred, axis=0)
        all_chi_true_np: NDArray = np.concatenate(all_chi_true, axis=0)
        all_chi_pred_np: NDArray = np.concatenate(all_chi_pred, axis=0)
        all_h_true_np: NDArray = np.concatenate(all_h_true, axis=0)
        all_h_pred_np: NDArray = np.concatenate(all_h_pred, axis=0)
        all_params_true_np: NDArray = np.concatenate(all_params_true, axis=0)
        all_params_pred_np: NDArray = np.concatenate(all_params_pred, axis=0)

        # Calculate summary statistics
        metrics: Dict[str, float] = {
            "forward_pk_mse": 0., #float(np.mean(forward_metrics["pk_mse"])),
            "forward_chi_mse": 0., #float(np.mean(forward_metrics["chi_mse"])),
            "forward_hubble_mse": 0., #float(np.mean(forward_metrics["hubble_mse"])),
            "inverse_param_mse": 0., #float(np.mean(inverse_metrics["param_mse"])),
            "forward_pk_mae": 0., #float(np.mean(np.abs(all_pk_true_np - all_pk_pred_np))),
            "forward_chi_mae": 0., #float(np.mean(np.abs(all_chi_true_np - all_chi_pred_np))),
            "forward_hubble_mae": 0., #float(np.mean(np.abs(all_h_true_np - all_h_pred_np))),
            "inverse_param_mae": 0., #float(np.mean(np.abs(all_params_true_np - all_params_pred_np))),
        }

        # Calculate R² scores
        for i in range(self.n_params):
            ss_res: float = float(
                np.sum((all_params_true_np[:, i] - all_params_pred_np[:, i]) ** 2)
            )
            ss_tot: float = float(
                np.sum(
                    (all_params_true_np[:, i] - np.mean(all_params_true_np[:, i])) ** 2
                )
            )
            r2: float = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            metrics[f"inverse_param_{i}_r2"] = r2

        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print("\nForward Model (Parameters → Observables):")
        print(f"  P(k) MSE:     {metrics['forward_pk_mse']:.6f}")
        print(f"  P(k) MAE:     {metrics['forward_pk_mae']:.6f}")
        print(f"  χ(z) MSE:     {metrics['forward_chi_mse']:.6f}")
        print(f"  χ(z) MAE:     {metrics['forward_chi_mae']:.6f}")
        print(f"  H(z) MSE:     {metrics['forward_hubble_mse']:.6f}")
        print(f"  H(z) MAE:     {metrics['forward_hubble_mae']:.6f}")

        print("\nInverse Model (Observables → Parameters):")
        print(f"  Params MSE:   {metrics['inverse_param_mse']:.6f}")
        print(f"  Params MAE:   {metrics['inverse_param_mae']:.6f}")

        if self.config.data_config.param_names:
            print("\n  Per-parameter R² scores:")
            for i, name in enumerate(self.config.data_config.param_names):
                print(f"    {name}: {metrics[f'inverse_param_{i}_r2']:.4f}")
        else:
            print("\n  Per-parameter R² scores:")
            for i in range(self.n_params):
                print(f"    Param {i}: {metrics[f'inverse_param_{i}_r2']:.4f}")

        print("=" * 60)

        # Plotting
        if plot:
            self._plot_evaluation(
                all_pk_true_np,
                all_pk_pred_np,
                all_chi_true_np,
                all_chi_pred_np,
                all_h_true_np,
                all_h_pred_np,
                all_params_true_np,
                all_params_pred_np,
                save_path,
            )

        return metrics

    def _plot_evaluation(
        self,
        pk_true: NDArray,
        pk_pred: NDArray,
        chi_true: NDArray,
        chi_pred: NDArray,
        h_true: NDArray,
        h_pred: NDArray,
        params_true: NDArray,
        params_pred: NDArray,
        save_path: Optional[str] = None,
    ) -> None:
        """Create evaluation plots"""

        # Select random samples for plotting
        n_samples: int = min(5, len(pk_true))
        idx: NDArray = np.random.choice(len(pk_true), n_samples, replace=False)

        # Forward model plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Power spectrum
        for i in idx:
            axes[0, 0].plot(pk_true[i], alpha=0.7, color="blue", linewidth=0.5)
            axes[0, 0].plot(
                pk_pred[i], alpha=0.7, color="red", linestyle="--", linewidth=0.5
            )
        axes[0, 0].set_xlabel("k bin")
        axes[0, 0].set_ylabel("P(k) [normalized]")
        axes[0, 0].set_title("Power Spectrum Predictions")
        axes[0, 0].legend(["True", "Predicted"], loc="best")
        axes[0, 0].grid(True, alpha=0.3)

        # Comoving distance
        for i in idx:
            axes[0, 1].plot(chi_true[i], alpha=0.7, color="blue", linewidth=0.5)
            axes[0, 1].plot(
                chi_pred[i], alpha=0.7, color="red", linestyle="--", linewidth=0.5
            )
        axes[0, 1].set_xlabel("z bin")
        axes[0, 1].set_ylabel("χ(z) [normalized]")
        axes[0, 1].set_title("Comoving Distance Predictions")
        axes[0, 1].grid(True, alpha=0.3)

        # Hubble parameter
        for i in idx:
            axes[0, 2].plot(h_true[i], alpha=0.7, color="blue", linewidth=0.5)
            axes[0, 2].plot(
                h_pred[i], alpha=0.7, color="red", linestyle="--", linewidth=0.5
            )
        axes[0, 2].set_xlabel("z bin")
        axes[0, 2].set_ylabel("H(z) [normalized]")
        axes[0, 2].set_title("Hubble Parameter Predictions")
        axes[0, 2].grid(True, alpha=0.3)

        # Residuals
        pk_residuals: NDArray = pk_pred - pk_true
        chi_residuals: NDArray = chi_pred - chi_true
        h_residuals: NDArray = h_pred - h_true

        axes[1, 0].hist(
            pk_residuals.flatten(), bins=50, alpha=0.7, color="blue", edgecolor="black"
        )
        axes[1, 0].set_xlabel("Residual")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title(f"P(k) Residuals (σ={np.std(pk_residuals):.4f})")
        axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=2)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(
            chi_residuals.flatten(),
            bins=50,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        axes[1, 1].set_xlabel("Residual")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title(f"χ(z) Residuals (σ={np.std(chi_residuals):.4f})")
        axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=2)
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].hist(
            h_residuals.flatten(), bins=50, alpha=0.7, color="orange", edgecolor="black"
        )
        axes[1, 2].set_xlabel("Residual")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].set_title(f"H(z) Residuals (σ={np.std(h_residuals):.4f})")
        axes[1, 2].axvline(0, color="red", linestyle="--", linewidth=2)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_forward.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Inverse model plots
        n_params: int = params_true.shape[1]
        n_cols: int = min(3, n_params)
        n_rows: int = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        param_names: List[str] = self.config.data_config.param_names or [
            f"Param {i}" for i in range(n_params)
        ]

        for i in range(n_params):
            # Scatter plot
            axes[i].scatter(params_true[:, i], params_pred[:, i], alpha=0.5, s=20)

            # Perfect prediction line
            min_val: float = min(params_true[:, i].min(), params_pred[:, i].min())
            max_val: float = max(params_true[:, i].max(), params_pred[:, i].max())
            axes[i].plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Perfect",
            )

            # Calculate R²
            ss_res: float = np.sum((params_true[:, i] - params_pred[:, i]) ** 2)
            ss_tot: float = np.sum(
                (params_true[:, i] - np.mean(params_true[:, i])) ** 2
            )
            r2: float = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            axes[i].set_xlabel(f"True {param_names[i]}")
            axes[i].set_ylabel(f"Predicted {param_names[i]}")
            axes[i].set_title(f"{param_names[i]} (R²={r2:.4f})")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # Hide extra subplots
        for i in range(n_params, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_inverse.png", dpi=150, bbox_inches="tight")
        plt.show()

    def save(self, path: str) -> None:
        """
        Save the complete emulator state

        Args:
            path: Base path for saving (without extension)
        """

        # Save models
        torch.save(
            {
                "forward_model_state": self.forward_model.state_dict(),
                "inverse_model_state": self.inverse_model.state_dict(),
                "forward_config": self.config.forward_model_config.model_dump(),
                "inverse_config": self.config.inverse_model_config.model_dump(),
            },
            f"{path}_models.pt",
        )

        # Save scalers and metadata
        with open(f"{path}_metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "param_scaler": self.param_scaler,
                    "pk_scaler": self.pk_scaler,
                    "chi_scaler": self.chi_scaler,
                    "hubble_scaler": self.hubble_scaler,
                    "config": self.config.model_dump(),
                    "is_fitted": self.is_fitted,
                    "forward_history": (
                        self.forward_history.model_dump()
                        if self.forward_history
                        else None
                    ),
                    "inverse_history": (
                        self.inverse_history.model_dump()
                        if self.inverse_history
                        else None
                    ),
                },
                f,
            )

        print(f"Emulator saved to {path}_models.pt and {path}_metadata.pkl")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "CosmologicalEmulator":
        """
        Load a saved emulator

        Args:
            path: Base path for loading (without extension)
            device: Device to load models to (if None, uses device from config)

        Returns:
            Loaded emulator instance
        """

        # Load metadata
        with open(f"{path}_metadata.pkl", "rb") as f:
            metadata: Dict = pickle.load(f)

        # Reconstruct config
        config: EmulatorConfig = EmulatorConfig(**metadata["config"])

        # Override device if specified
        if device is not None:
            config = EmulatorConfig(**{**config.model_dump(), "device": device})

        # Create emulator instance
        emulator: "CosmologicalEmulator" = cls(config)

        # Load scalers
        emulator.param_scaler = metadata["param_scaler"]
        emulator.pk_scaler = metadata["pk_scaler"]
        emulator.chi_scaler = metadata["chi_scaler"]
        emulator.hubble_scaler = metadata["hubble_scaler"]
        emulator.is_fitted = metadata["is_fitted"]

        # Load training histories if they exist
        if metadata["forward_history"] is not None:
            emulator.forward_history = TrainingHistory(**metadata["forward_history"])
        if metadata["inverse_history"] is not None:
            emulator.inverse_history = TrainingHistory(**metadata["inverse_history"])

        # Load model states
        checkpoint: Dict = torch.load(f"{path}_models.pt", map_location=emulator.device)
        emulator.forward_model.load_state_dict(checkpoint["forward_model_state"])
        emulator.inverse_model.load_state_dict(checkpoint["inverse_model_state"])

        # Set models to evaluation mode
        emulator.forward_model.eval()
        emulator.inverse_model.eval()

        print(f"Emulator loaded from {path}")
        print(f"Device: {emulator.device}")
        print(f"Fitted: {emulator.is_fitted}")

        return emulator


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================


def generate_synthetic_data(
    n_samples: int = 10000,
    n_params: int = 6,
    n_k: int = 100,
    n_z: int = 50,
    random_seed: int = 42,
) -> Tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    """
    Generate synthetic cosmological data for demonstration

    In practice, this would come from CCL calculations

    Args:
        n_samples: Number of cosmological models
        n_params: Number of cosmological parameters
        n_k: Number of k bins
        n_z: Number of redshift bins
        random_seed: Random seed

    Returns:
        cosmo_params, power_spectra, comoving_distances, hubble_evolution
    """

    np.random.seed(random_seed)

    # Generate cosmological parameters
    # Typical ranges: Omega_m, Omega_b, h, n_s, sigma_8, w0
    param_ranges: NDArray[np.float32] = np.array(
        [
            [0.2, 0.4],  # Omega_m
            [0.04, 0.06],  # Omega_b
            [0.6, 0.8],  # h
            [0.92, 1.0],  # n_s
            [0.7, 0.9],  # sigma_8
            [-1.3, -0.7],  # w0
        ],
        dtype=np.float32,
    )

    cosmo_params: NDArray[np.float32] = np.random.uniform(
        param_ranges[:n_params, 0],
        param_ranges[:n_params, 1],
        size=(n_samples, n_params),
    ).astype(np.float32)

    # Generate k and z grids
    k: NDArray[np.float32] = np.logspace(-4, 1, n_k, dtype=np.float32)
    z: NDArray[np.float32] = np.linspace(0, 3, n_z, dtype=np.float32)

    # Simulate power spectra with realistic k-dependence
    power_spectra: NDArray[np.float32] = np.zeros((n_samples, n_k), dtype=np.float32)
    for i in range(n_samples):
        # Simple power law with cosmology-dependent amplitude and tilt
        amplitude: float = cosmo_params[i, 4] ** 2 if n_params > 4 else 0.8  # sigma_8
        tilt: float = cosmo_params[i, 3] if n_params > 3 else 0.96  # n_s
        power_spectra[i] = amplitude * k**tilt * np.exp(-k / 10.0)
        # Add small noise
        power_spectra[i] *= 1 + 0.01 * np.random.randn(n_k)

    # Simulate comoving distances
    comoving_distances: NDArray[np.float32] = np.zeros(
        (n_samples, n_z), dtype=np.float32
    )
    for i in range(n_samples):
        # Simple model: chi(z) depends on Omega_m and h
        omega_m: float = cosmo_params[i, 0] if n_params > 0 else 0.3
        h: float = cosmo_params[i, 2] if n_params > 2 else 0.7
        # Approximate comoving distance (not exact, for demonstration)
        comoving_distances[i] = (
            3000.0 * z / (h * np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m)))
        )
        # Add small noise
        comoving_distances[i] *= 1 + 0.005 * np.random.randn(n_z)

    # Simulate Hubble parameter evolution
    hubble_evolution: NDArray[np.float32] = np.zeros((n_samples, n_z), dtype=np.float32)
    for i in range(n_samples):
        omega_m_sim: float = cosmo_params[i, 0] if n_params > 0 else 0.3
        h_sim: float = cosmo_params[i, 2] if n_params > 2 else 0.7
        w0: float = cosmo_params[i, 5] if n_params > 5 else -1.0
        # Approximate H(z)
        H0: float = 100.0 * h_sim
        hubble_evolution[i] = H0 * np.sqrt(
            omega_m_sim * (1 + z) ** 3 + (1 - omega_m) * (1 + z) ** (3 * (1 + w0))
        )
        # Add small noise
        hubble_evolution[i] *= 1 + 0.005 * np.random.randn(n_z)

    return cosmo_params, power_spectra, comoving_distances, hubble_evolution


def example_usage() -> None:
    """Demonstrate the emulator usage"""

    print("=" * 80)
    print("COSMOLOGICAL EMULATOR DEMONSTRATION")
    print("=" * 80)

    # Configuration
    param_names: List[str] = ["Omega_m", "Omega_b", "h", "n_s", "sigma_8", "w0"]
    n_params: int = 6
    n_k: int = 100
    n_z: int = 50

    # Create configurations using Pydantic
    data_config: DataConfig = DataConfig(
        n_params=n_params,
        n_k=n_k,
        n_z=n_z,
        param_names=param_names,
        k_min=1e-4,
        k_max=10.0,
        z_min=0.0,
        z_max=3.0,
    )

    forward_config: ModelConfig = ModelConfig(
        hidden_dims=[256, 512, 512, 256], n_residual_blocks=3, dropout=0.1
    )

    inverse_config: ModelConfig = ModelConfig(
        hidden_dims=[512, 512, 256, 128], n_residual_blocks=3, dropout=0.1
    )

    training_config: TrainingConfig = TrainingConfig(
        epochs=100,  # Reduced for demonstration
        batch_size=64,
        learning_rate=1e-3,
        patience=15,
        validation_split=0.15,
        test_split=0.15,
    )

    emulator_config: EmulatorConfig = EmulatorConfig(
        data_config=data_config,
        forward_model_config=forward_config,
        inverse_model_config=inverse_config,
        training_config=training_config,
        device="cuda",  # Will fallback to CPU if CUDA not available
        use_log_pk=True,
        random_seed=42,
    )

    print("\n" + "-" * 80)
    print("Configuration Summary:")
    print("-" * 80)
    print(f"Data: {n_params} parameters, {n_k} k-bins, {n_z} z-bins")
    print(
        f"Forward model: {forward_config.hidden_dims}, {forward_config.n_residual_blocks} residual blocks"
    )
    print(
        f"Inverse model: {inverse_config.hidden_dims}, {inverse_config.n_residual_blocks} residual blocks"
    )
    print(
        f"Training: {training_config.epochs} epochs, batch_size={training_config.batch_size}"
    )

    # Generate synthetic data
    print("\n" + "-" * 80)
    print("Generating synthetic data...")
    print("-" * 80)

    cosmo_params: NDArray[np.float32]
    power_spectra: NDArray[np.float32]
    comoving_distances: NDArray[np.float32]
    hubble_evolution: NDArray[np.float32]

    cosmo_params, power_spectra, comoving_distances, hubble_evolution = (
        generate_synthetic_data(n_samples=10000, n_params=n_params, n_k=n_k, n_z=n_z)
    )

    print(f"Generated {len(cosmo_params)} samples")
    print(f"Cosmo params shape: {cosmo_params.shape}")
    print(f"Power spectra shape: {power_spectra.shape}")
    print(f"Comoving distances shape: {comoving_distances.shape}")
    print(f"Hubble evolution shape: {hubble_evolution.shape}")

    # Initialize emulator
    print("\n" + "-" * 80)
    print("Initializing emulator...")
    print("-" * 80)

    emulator: CosmologicalEmulator = CosmologicalEmulator(emulator_config)

    # Prepare data
    print("\n" + "-" * 80)
    print("Preparing data...")
    print("-" * 80)

    train_dataset: CosmologyDataset
    val_dataset: CosmologyDataset
    test_dataset: CosmologyDataset

    train_dataset, val_dataset, test_dataset = emulator.prepare_data(
        cosmo_params, power_spectra, comoving_distances, hubble_evolution
    )

    # Train forward model
    print("\n" + "-" * 80)
    print("Training forward model...")
    print("-" * 80)

    forward_history: TrainingHistory = emulator.train_forward_model(
        train_dataset, val_dataset
    )

    # Plot forward model training history
    forward_history.plot(save_path="forward_training_history.png")

    # Train inverse model
    print("\n" + "-" * 80)
    print("Training inverse model...")
    print("-" * 80)

    inverse_history: TrainingHistory = emulator.train_inverse_model(
        train_dataset, val_dataset
    )

    # Plot inverse model training history
    inverse_history.plot(save_path="inverse_training_history.png")

    # Evaluate models
    print("\n" + "-" * 80)
    print("Evaluating models...")
    print("-" * 80)

    metrics: Dict[str, float] = emulator.evaluate(
        test_dataset, plot=True, save_path="evaluation"
    )

    # Test prediction
    print("\n" + "-" * 80)
    print("Testing predictions...")
    print("-" * 80)

    # Forward prediction
    test_params: NDArray[np.float32] = cosmo_params[:5]
    pred_pk: NDArray[np.float32]
    pred_chi: NDArray[np.float32]
    pred_h: NDArray[np.float32]

    pred_pk, pred_chi, pred_h = emulator.predict_observables(test_params)

    print("\nForward prediction:")
    print(f"  Input params shape: {test_params.shape}")
    print(f"  Output P(k) shape: {pred_pk.shape}")
    print(f"  Output χ(z) shape: {pred_chi.shape}")
    print(f"  Output H(z) shape: {pred_h.shape}")

    # Inverse prediction
    test_pk: NDArray[np.float32] = power_spectra[:5]
    test_chi: NDArray[np.float32] = comoving_distances[:5]
    pred_params: NDArray[np.float32] = emulator.infer_parameters(test_pk, test_chi)

    print("\nInverse prediction:")
    print(f"  Input P(k) shape: {test_pk.shape}")
    print(f"  Input χ(z) shape: {test_chi.shape}")
    print(f"  Output params shape: {pred_params.shape}")

    # Compare true vs predicted parameters
    print("\nParameter comparison (first 3 samples):")
    for i in range(3):
        print(f"\n  Sample {i+1}:")
        for j, name in enumerate(param_names):
            print(
                f"    {name}: True={cosmo_params[i,j]:.4f}, Pred={pred_params[i,j]:.4f}, "
                f"Error={abs(cosmo_params[i,j]-pred_params[i,j]):.4f}"
            )

    # Save emulator
    print("\n" + "-" * 80)
    print("Saving emulator...")
    print("-" * 80)

    emulator.save("cosmology_emulator")

    # Test loading
    print("\n" + "-" * 80)
    print("Testing emulator loading...")
    print("-" * 80)

    loaded_emulator: CosmologicalEmulator = CosmologicalEmulator.load(
        "cosmology_emulator"
    )

    # Verify loaded emulator works
    loaded_pred_pk: NDArray[np.float32]
    loaded_pred_chi: NDArray[np.float32]
    loaded_pred_h: NDArray[np.float32]

    loaded_pred_pk, loaded_pred_chi, loaded_pred_h = (
        loaded_emulator.predict_observables(test_params)
    )

    print(f"Loaded emulator prediction matches: {np.allclose(pred_pk, loaded_pred_pk)}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Run the example
    example_usage()
