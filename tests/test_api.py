"""
Integration Tests for GraphSentry FastAPI Serving Endpoints.

Validates:
- Health check and runtime diagnostics (/health)
- High-level graph topology metadata (/api/v1/graph/stats)
- Single-account localized GNN investigation (/api/v1/investigate)
- Non-existent entity 404 error handling
- Multi-account batch scoring (/api/v1/batch-predict)
- Standalone interactive HTML visualizer generation (/api/v1/account/{id}/visualize)
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.server import (
    app,
    health_check,
    graph_stats,
    investigate_account,
    visualize_account_network,
    batch_predict,
    root,
)
from src.api.schemas import InvestigateRequest, BatchPredictRequest
from src.graph.dataset import AMLHeteroDataset
from src.models.explainer import ForensicExplainer


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load dataset & explainer into app state once
        dataset = AMLHeteroDataset()
        explainer = ForensicExplainer(dataset=dataset)
        app.state.dataset = dataset
        app.state.explainer = explainer
        app.state.model_loaded = True
        app.state.graph_loaded = True

    def test_root_endpoint(self):
        """Test landing page endpoint."""
        data = root()
        self.assertEqual(data["status"], "operational")
        self.assertIn("endpoints", data)

    def test_health_endpoint(self):
        """Test system health diagnostics."""
        res = health_check()
        self.assertEqual(res.status, "healthy")
        self.assertTrue(res.graph_loaded)
        self.assertGreater(res.total_accounts, 0)
        self.assertGreater(res.total_transactions, 0)

    def test_graph_stats_endpoint(self):
        """Test graph statistics and entity counts."""
        data = graph_stats()
        self.assertIn("entity_counts", data)
        self.assertIn("edge_counts", data)
        self.assertIn("account", data["entity_counts"])
        self.assertIn("transfers_to", data["edge_counts"])

    def test_investigate_valid_account(self):
        """Test single-account forensic AML investigation."""
        req = InvestigateRequest(account_id="ACC_000001", k_hops=2, include_subgraph=True)
        res = investigate_account(req)
        self.assertEqual(res.account_id, "ACC_000001")
        self.assertTrue(0.0 <= res.risk_score <= 1.0)
        self.assertIn(res.risk_tier, ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
        self.assertGreater(len(res.detected_typologies), 0)
        self.assertIn("FINANCIAL CRIME COMPLIANCE AUDIT TRAIL", res.regulatory_summary)
        self.assertGreater(res.subgraph_topology.num_accounts, 0)
        self.assertGreater(res.inference_latency_ms, 0.0)

    def test_investigate_non_existent_account(self):
        """Test 404 error handling for invalid account identifier."""
        from fastapi import HTTPException
        req = InvestigateRequest(account_id="ACC_NON_EXISTENT_999999", k_hops=2)
        with self.assertRaises(HTTPException) as ctx:
            investigate_account(req)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_batch_predict_endpoint(self):
        """Test bulk scoring endpoint."""
        req = BatchPredictRequest(
            account_ids=["ACC_000001", "ACC_000002", "ACC_000003", "ACC_INVALID_999"]
        )
        res = batch_predict(req)
        self.assertEqual(res.total_evaluated, 4)
        self.assertEqual(len(res.results), 4)
        self.assertEqual(res.results[0].account_id, "ACC_000001")
        self.assertEqual(res.results[3].risk_tier, "UNKNOWN")

    def test_visualization_endpoint(self):
        """Test interactive Vis.js HTML rendering endpoint."""
        res = visualize_account_network("ACC_000001", k_hops=2)
        html_text = res.body.decode("utf-8")
        self.assertIn("<!DOCTYPE html>", html_text)
        self.assertIn("GraphSentry AML Forensic Subgraph", html_text)
        self.assertIn("ACC_000001", html_text)


if __name__ == "__main__":
    unittest.main()
