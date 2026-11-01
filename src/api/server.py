"""
FastAPI Production REST API Server for GraphSentry.

Exposes high-throughput, low-latency AML inference endpoints:
- GET  /health: System diagnostics, memory footprint, and graph stats
- POST /api/v1/investigate: Single-account GNN scoring & forensic explainability audit
- POST /api/v1/batch-predict: High-volume bulk account scoring
- GET  /api/v1/account/{account_id}/visualize: Standalone interactive Vis.js dashboard HTML
- GET  /api/v1/graph/stats: High-level graph topology and entity distribution
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.api.schemas import (
    BatchAccountRisk,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    InvestigateRequest,
    InvestigateResponse,
    SubnetworkNode,
    SubnetworkTopology,
)
from src.config import settings
from src.graph.dataset import AMLHeteroDataset
from src.models.explainer import ForensicExplainer
from dashboard.visualizer import generate_subgraph_html

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Lifespan Context Manager for Model and Graph Warmup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Bootstrapping GraphSentry AML Inference Engine...")
    try:
        dataset = AMLHeteroDataset()
        explainer = ForensicExplainer(dataset=dataset)
        app.state.dataset = dataset
        app.state.explainer = explainer
        app.state.model_loaded = True
        app.state.graph_loaded = True
        print(f"[+] Engine successfully loaded graph with {dataset.num_accounts:,} accounts.")
    except Exception as exc:
        print(f"[!] Warning during warmup: {exc}")
        app.state.dataset = None
        app.state.explainer = None
        app.state.model_loaded = False
        app.state.graph_loaded = False
    yield
    print("[*] Shutting down GraphSentry Inference Engine.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Heterogeneous Graph Neural Network Engine for Anti-Money Laundering & Mule Ring Detection",
    lifespan=lifespan,
)

# Enable CORS for enterprise compliance dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """Service landing page and endpoint catalog."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "investigate": "/api/v1/investigate",
            "batch_predict": "/api/v1/batch-predict",
            "graph_stats": "/api/v1/graph/stats",
            "visualize_template": "/api/v1/account/{account_id}/visualize",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Health check endpoint reporting engine status and graph statistics."""
    dataset = getattr(app.state, "dataset", None)
    total_acc = dataset.num_accounts if dataset else 0
    total_tx = dataset.metadata.get("edge_counts", {}).get("transfers_to", 0) if dataset else 0
    device_str = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"

    return HealthResponse(
        status="healthy" if dataset is not None else "degraded",
        version=settings.APP_VERSION,
        service=settings.APP_NAME,
        compute_device=device_str,
        model_loaded=getattr(app.state, "model_loaded", False),
        graph_loaded=getattr(app.state, "graph_loaded", False),
        total_accounts=total_acc,
        total_transactions=total_tx,
    )


@app.get("/api/v1/graph/stats", tags=["Graph Metadata"])
def graph_stats():
    """Return topological and entity distribution metrics of active graph."""
    dataset = getattr(app.state, "dataset", None)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph dataset is not yet initialized.",
        )
    return {
        "entity_counts": dataset.metadata.get("node_counts", {}),
        "edge_counts": dataset.metadata.get("edge_counts", {}),
        "mule_prevalence_rate_pct": round(dataset.metadata.get("imbalance_ratio", 0.0) * 100.0, 3),
        "account_feature_names": dataset.metadata.get("account_feature_names", []),
    }


def _resolve_account_id(raw_id: str, id_to_idx: dict) -> str:
    """Normalize input account identifiers (e.g. ACCT_00001 -> ACC_000001, or 1 -> ACC_000001)."""
    clean = raw_id.strip().upper()
    if clean in id_to_idx:
        return clean
    if clean.startswith("ACCT_") or clean.isdigit():
        digits = "".join(c for c in clean if c.isdigit())
        if digits:
            candidate = f"ACC_{int(digits):06d}"
            if candidate in id_to_idx:
                return candidate
    return clean


@app.post("/api/v1/investigate", response_model=InvestigateResponse, tags=["Forensic AML Inference"])
def investigate_account(payload: InvestigateRequest):
    """
    Execute localized GNN inference and generate regulatory compliance audit report.
    """
    dataset = getattr(app.state, "dataset", None)
    explainer = getattr(app.state, "explainer", None)

    if not dataset or not explainer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph inference engine is currently uninitialized.",
        )

    start_time = time.perf_counter()
    account_id = _resolve_account_id(payload.account_id, dataset.account_id_to_idx)

    if account_id not in dataset.account_id_to_idx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account '{account_id}' was not found in the transaction graph.",
        )

    # Execute forensic explainability & topological attribution
    explanation = explainer.explain_account(
        account_id=account_id,
        k_hops=payload.k_hops,
    )

    threshold = payload.threshold if payload.threshold is not None else settings.DECISION_THRESHOLD
    risk_score = explanation["risk_score"]
    is_flagged = risk_score >= threshold

    # Optional subnetwork node payload
    subgraph_nodes = None
    if payload.include_subgraph:
        sub_raw = dataset.extract_ego_subgraph(account_id, k_hops=payload.k_hops)
        idx_to_aid = dataset.idx_to_account_id
        idx_to_cid = {int(v): k for k, v in dataset.metadata.get("customer_id_to_idx", {}).items()}
        idx_to_did = {int(v): k for k, v in dataset.metadata.get("device_id_to_idx", {}).items()}

        subgraph_nodes = []
        for aid_idx in sub_raw["account_indices"][:50]:
            aid_str = idx_to_aid.get(aid_idx, f"ACC_{aid_idx}")
            subgraph_nodes.append(
                SubnetworkNode(
                    id=aid_str,
                    type="account",
                    is_target=(aid_str == account_id),
                )
            )
        for cid_idx in sub_raw["customer_indices"][:20]:
            cid_str = idx_to_cid.get(cid_idx, f"CUST_{cid_idx}")
            subgraph_nodes.append(SubnetworkNode(id=cid_str, type="customer"))

        for did_idx in sub_raw["device_indices"][:20]:
            did_str = idx_to_did.get(did_idx, f"DEV_{did_idx}")
            subgraph_nodes.append(SubnetworkNode(id=did_str, type="device"))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return InvestigateResponse(
        account_id=account_id,
        risk_score=risk_score,
        is_flagged=is_flagged,
        risk_tier=explanation["risk_tier"],
        detected_typologies=explanation["detected_typologies"],
        regulatory_summary=explanation["regulatory_summary"],
        recommended_action=explanation["recommended_action"],
        evidence_notes=explanation["evidence_notes"],
        top_contributing_edges=explanation["top_contributing_edges"],
        subgraph_topology=SubnetworkTopology(**explanation["subgraph_topology"]),
        subgraph_nodes=subgraph_nodes,
        inference_latency_ms=round(elapsed_ms, 2),
    )


@app.get("/api/v1/account/{account_id}/visualize", response_class=HTMLResponse, tags=["Visualization"])
@app.get("/visualize/{account_id}", response_class=HTMLResponse, tags=["Visualization"], include_in_schema=False)
def visualize_account_network(account_id: str, k_hops: int = Query(default=2, ge=1, le=3)):
    """
    Renders an interactive Vis.js forensic graph visualization in HTML.
    """
    dataset = getattr(app.state, "dataset", None)
    explainer = getattr(app.state, "explainer", None)

    if not dataset or not explainer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine uninitialized.",
        )

    clean_id = _resolve_account_id(account_id, dataset.account_id_to_idx)
    if clean_id not in dataset.account_id_to_idx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account '{clean_id}' does not exist in graph registry.",
        )

    explanation = explainer.explain_account(clean_id, k_hops=k_hops)
    html_content = generate_subgraph_html(explanation)
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/api/v1/batch-predict", response_model=BatchPredictResponse, tags=["Batch Scoring"])
def batch_predict(payload: BatchPredictRequest):
    """
    Bulk evaluation of account fraud risk scores.
    """
    dataset = getattr(app.state, "dataset", None)
    explainer = getattr(app.state, "explainer", None)

    if not dataset or not explainer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine is uninitialized.",
        )

    start_time = time.perf_counter()
    results: List[BatchAccountRisk] = []
    flagged_count = 0

    for aid in payload.account_ids:
        clean_id = _resolve_account_id(aid, dataset.account_id_to_idx)
        if clean_id in dataset.account_id_to_idx:
            exp = explainer.explain_account(clean_id, k_hops=1)
            score = exp["risk_score"]
            flagged = score >= settings.DECISION_THRESHOLD
            if flagged:
                flagged_count += 1
            results.append(
                BatchAccountRisk(
                    account_id=clean_id,
                    risk_score=score,
                    is_flagged=flagged,
                    risk_tier=exp["risk_tier"],
                )
            )
        else:
            results.append(
                BatchAccountRisk(
                    account_id=clean_id,
                    risk_score=0.0,
                    is_flagged=False,
                    risk_tier="UNKNOWN",
                )
            )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return BatchPredictResponse(
        results=results,
        total_evaluated=len(results),
        flagged_count=flagged_count,
        batch_latency_ms=round(elapsed_ms, 2),
    )
