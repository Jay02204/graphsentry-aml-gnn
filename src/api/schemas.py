"""
Pydantic v2 Request & Response Data Models for GraphSentry API.

Provides strict data validation for:
- /health diagnostics
- /api/v1/investigate single-account forensic queries
- /api/v1/batch-predict bulk inference
- Forensic explanations and subgraph topology structures
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """System health check and graph registry status."""
    status: str = Field(default="healthy", description="Service health state")
    version: str = Field(..., description="API version")
    service: str = Field(..., description="Service name")
    compute_device: str = Field(..., description="Active compute device (cuda/cpu)")
    model_loaded: bool = Field(..., description="Whether GNN model weights are loaded")
    graph_loaded: bool = Field(..., description="Whether HeteroData graph cache is loaded")
    total_accounts: int = Field(..., description="Total accounts in active graph")
    total_transactions: int = Field(..., description="Total transactions in active graph")


class InvestigateRequest(BaseModel):
    """Request payload to trigger localized GNN inference and forensic SAR audit."""
    account_id: str = Field(
        ...,
        description="Target account identifier (e.g. 'ACC_000042')",
        examples=["ACC_000001"],
    )
    k_hops: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Subnetwork neighborhood expansion depth",
    )
    include_subgraph: bool = Field(
        default=True,
        description="Whether to return full localized node and edge structures",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom decision boundary threshold for flagging",
    )

    @field_validator("account_id")
    @classmethod
    def validate_account_id(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("account_id cannot be blank.")
        return cleaned


class EdgeExplanation(BaseModel):
    """Attribution and metadata for a relation in the ego-subgraph."""
    source: str = Field(..., description="Source node identifier")
    target: str = Field(..., description="Target node identifier")
    relation: str = Field(..., description="Edge relation type ('transfers_to', 'owns', 'logged_in_from')")
    importance_score: float = Field(..., ge=0.0, le=1.0, description="Model attribution importance")
    amount: Optional[float] = Field(default=None, description="Transaction monetary amount")
    timestamp: Optional[str] = Field(default=None, description="Transaction timestamp")


class SubnetworkNode(BaseModel):
    """Node entity in the localized forensic ego-subgraph."""
    id: str = Field(..., description="Entity identifier (account, customer, or device ID)")
    type: str = Field(..., description="Node category: 'account', 'customer', or 'device'")
    risk_score: Optional[float] = Field(default=None, description="Predicted risk score")
    is_target: bool = Field(default=False, description="Whether this is the inquiry focal node")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary node attributes")


class SubnetworkTopology(BaseModel):
    """Aggregate statistics of the localized computational subnetwork."""
    num_accounts: int = Field(..., description="Total accounts in k-hop neighborhood")
    num_customers: int = Field(..., description="Total customer entities connected")
    num_devices: int = Field(..., description="Total hardware devices connected")
    num_transactions: int = Field(..., description="Total transfer edges in subnetwork")


class InvestigateResponse(BaseModel):
    """Comprehensive forensic report and regulatory audit narrative."""
    account_id: str = Field(..., description="Target account identifier")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="GNN predicted fraud probability")
    is_flagged: bool = Field(..., description="True if risk_score exceeds decision threshold")
    risk_tier: str = Field(..., description="Assigned risk tier: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'")
    detected_typologies: List[str] = Field(..., description="Identified AML crime patterns")
    regulatory_summary: str = Field(..., description="Audit narrative for FinCEN SAR compliance")
    recommended_action: str = Field(..., description="Recommended regulatory remediation mandate")
    evidence_notes: List[str] = Field(..., description="Itemized forensic topological findings")
    top_contributing_edges: List[EdgeExplanation] = Field(..., description="Key influential graph relations")
    subgraph_topology: SubnetworkTopology = Field(..., description="Graph neighborhood volume")
    subgraph_nodes: Optional[List[SubnetworkNode]] = Field(default=None, description="Localized subnetwork nodes")
    subgraph_edges: Optional[List[EdgeExplanation]] = Field(default=None, description="Localized subnetwork edges")
    inference_latency_ms: float = Field(..., description="End-to-end inference execution latency in ms")


class BatchPredictRequest(BaseModel):
    """Request payload for multi-account bulk scoring."""
    account_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of account identifiers to evaluate",
        examples=[["ACC_000001", "ACC_000002", "ACC_000003"]],
    )


class BatchAccountRisk(BaseModel):
    """Risk summary for an individual account in batch evaluation."""
    account_id: str
    risk_score: float
    is_flagged: bool
    risk_tier: str


class BatchPredictResponse(BaseModel):
    """Bulk evaluation results."""
    results: List[BatchAccountRisk]
    total_evaluated: int
    flagged_count: int
    batch_latency_ms: float
