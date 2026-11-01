"""
GraphSentry Configuration Module
Defines hyperparameter specifications, filesystem paths, and environment settings
using Pydantic BaseSettings.
"""

from pathlib import Path
from typing import List, Tuple
from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _USE_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as BaseSettings
    _USE_PYDANTIC_SETTINGS = False


class Settings(BaseSettings):
    """Global configuration settings for GraphSentry engine."""

    # Application Metadata
    APP_NAME: str = "GraphSentry AML GNN Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Base Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    MODEL_DIR: Path = ARTIFACTS_DIR / "models"
    VISUALIZATION_DIR: Path = ARTIFACTS_DIR / "visualizations"

    # Processed Data Artifacts
    HETERO_GRAPH_FILE: str = "hetero_graph.pt"
    METADATA_FILE: str = "graph_metadata.json"
    BEST_MODEL_FILE: str = "best_hetero_gnn.pt"

    # Heterogeneous Graph Schema Definitions
    ACCOUNT_NODE: str = "account"
    CUSTOMER_NODE: str = "customer"
    DEVICE_NODE: str = "device"

    TRANSFER_RELATION: Tuple[str, str, str] = ("account", "transfers_to", "account")
    REV_TRANSFER_RELATION: Tuple[str, str, str] = ("account", "rev_transfers_to", "account")
    OWNS_RELATION: Tuple[str, str, str] = ("customer", "owns", "account")
    REV_OWNS_RELATION: Tuple[str, str, str] = ("account", "rev_owns", "customer")
    LOGIN_RELATION: Tuple[str, str, str] = ("account", "logged_in_from", "device")
    REV_LOGIN_RELATION: Tuple[str, str, str] = ("device", "rev_logged_in_from", "account")

    # GNN Architecture Hyperparameters
    HIDDEN_CHANNELS: int = Field(default=64, description="Hidden embedding dimension across GNN layers")
    NUM_LAYERS: int = Field(default=2, description="Number of message-passing layers")
    DROPOUT: float = Field(default=0.2, description="Dropout rate for regularization")
    GNN_AGGR: str = Field(default="mean", description="Neighborhood aggregation operator ('mean', 'sum', 'max')")

    # Mini-batch Neighborhood Sampling
    NUM_NEIGHBORS: List[int] = Field(default=[15, 10], description="k-hop sampling fan-out per layer")
    TRAIN_BATCH_SIZE: int = Field(default=512, description="Mini-batch size for training loader")
    EVAL_BATCH_SIZE: int = Field(default=1024, description="Mini-batch size for evaluation loader")

    # Training Optimization Hyperparameters
    LEARNING_RATE: float = Field(default=1e-3, description="AdamW learning rate")
    WEIGHT_DECAY: float = Field(default=1e-4, description="Weight decay / L2 regularization")
    EPOCHS: int = Field(default=25, description="Maximum training epochs")
    EARLY_STOPPING_PATIENCE: int = Field(default=6, description="Patience epochs before early stopping")

    # Class Imbalance Mitigation (Weighted Focal Loss)
    FOCAL_GAMMA: float = Field(default=2.0, description="Focal Loss focusing parameter gamma")
    FOCAL_ALPHA: float = Field(default=0.85, description="Focal Loss class-weighting factor alpha")

    # Inference & Decision Boundaries
    DECISION_THRESHOLD: float = Field(default=0.45, description="Probability cutoff for mule/fraud classification")
    CRITICAL_RISK_THRESHOLD: float = Field(default=0.80, description="Cutoff for immediate account freezing")
    HIGH_RISK_THRESHOLD: float = Field(default=0.60, description="Cutoff for SAR filing & Enhanced Due Diligence")
    MEDIUM_RISK_THRESHOLD: float = Field(default=0.40, description="Cutoff for analyst review")

    # API Server Deployment Settings
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    WORKERS: int = 1

    # Reproducibility
    RANDOM_SEED: int = 42

    if _USE_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not already exist."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.ARTIFACTS_DIR,
            self.MODEL_DIR,
            self.VISUALIZATION_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Instantiate singleton config
settings = Settings()
settings.ensure_directories()
