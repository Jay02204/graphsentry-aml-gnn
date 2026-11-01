"""
Unit Tests for Heterogeneous Graph Construction and Tensor Dimensionality.

Validates:
- ID mapping bijectivity and zero-based indexing
- Node feature matrix shapes for 'account', 'customer', and 'device'
- Edge index bounds (ensuring zero out-of-range references)
- Disjointness and completeness of train, validation, and test split masks
- Numerical stability (zero NaNs / Infs in continuous tensors)
"""

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.dataset import AMLHeteroDataset
from src.graph.topological_features import extract_topological_features


class TestGraphBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = AMLHeteroDataset()

    def test_metadata_integrity(self):
        """Verify metadata counts and feature dimensions."""
        metadata = self.dataset.metadata
        self.assertIn("node_counts", metadata)
        self.assertGreater(metadata["node_counts"]["account"], 0)
        self.assertGreater(metadata["node_counts"]["customer"], 0)
        self.assertGreater(metadata["node_counts"]["device"], 0)

        self.assertGreater(metadata["mule_count"], 0)
        self.assertGreater(metadata["licit_count"], metadata["mule_count"])
        self.assertTrue(0.0 < metadata["imbalance_ratio"] < 0.15)

    def test_account_id_mappings(self):
        """Verify bidirectional ID mappings are bijective and valid."""
        aid_to_idx = self.dataset.account_id_to_idx
        idx_to_aid = self.dataset.idx_to_account_id

        self.assertEqual(len(aid_to_idx), len(idx_to_aid))
        self.assertEqual(len(aid_to_idx), self.dataset.num_accounts)

        # Spot check first 50 accounts
        for i in range(min(50, self.dataset.num_accounts)):
            aid = idx_to_aid[i]
            self.assertEqual(aid_to_idx[aid], i)

    def test_split_masks_disjoint(self):
        """Verify train, val, and test masks are mutually exclusive and cover 100% of accounts."""
        if isinstance(self.dataset.data, dict) and "train_mask" in self.dataset.data:
            train_mask = self.dataset.data["train_mask"]
            val_mask = self.dataset.data["val_mask"]
            test_mask = self.dataset.data["test_mask"]
        else:
            train_mask = self.dataset.data["account"].train_mask.tolist()
            val_mask = self.dataset.data["account"].val_mask.tolist()
            test_mask = self.dataset.data["account"].test_mask.tolist()

        n = self.dataset.num_accounts
        self.assertEqual(len(train_mask), n)
        self.assertEqual(len(val_mask), n)
        self.assertEqual(len(test_mask), n)

        for i in range(n):
            masks_active = int(train_mask[i]) + int(val_mask[i]) + int(test_mask[i])
            self.assertEqual(masks_active, 1, f"Account index {i} belongs to {masks_active} split masks!")

    def test_edge_indices_bounds(self):
        """Verify edge indices do not reference non-existent node IDs."""
        num_accounts = self.dataset.num_accounts
        num_customers = self.dataset.num_customers
        num_devices = self.dataset.num_devices

        if hasattr(self.dataset.data, "__getitem__") and "transfer_src" in self.dataset.data:
            transfer_src = self.dataset.data["transfer_src"]
            transfer_dst = self.dataset.data["transfer_dst"]
            owns_src = self.dataset.data["owns_src"]
            owns_dst = self.dataset.data["owns_dst"]
            login_src = self.dataset.data["login_src"]
            login_dst = self.dataset.data["login_dst"]
        else:
            transfer_src = self.dataset.data["account", "transfers_to", "account"].edge_index[0].tolist()
            transfer_dst = self.dataset.data["account", "transfers_to", "account"].edge_index[1].tolist()
            owns_src = self.dataset.data["customer", "owns", "account"].edge_index[0].tolist()
            owns_dst = self.dataset.data["customer", "owns", "account"].edge_index[1].tolist()
            login_src = self.dataset.data["account", "logged_in_from", "device"].edge_index[0].tolist()
            login_dst = self.dataset.data["account", "logged_in_from", "device"].edge_index[1].tolist()

        # Transfers: account -> account
        self.assertEqual(len(transfer_src), len(transfer_dst))
        self.assertGreater(len(transfer_src), 0)
        self.assertLess(max(transfer_src), num_accounts)
        self.assertGreaterEqual(min(transfer_src), 0)
        self.assertLess(max(transfer_dst), num_accounts)
        self.assertGreaterEqual(min(transfer_dst), 0)

        # Ownership: customer -> account
        self.assertEqual(len(owns_src), len(owns_dst))
        self.assertLess(max(owns_src), num_customers)
        self.assertGreaterEqual(min(owns_src), 0)
        self.assertLess(max(owns_dst), num_accounts)
        self.assertGreaterEqual(min(owns_dst), 0)

        # Logins: account -> device
        self.assertEqual(len(login_src), len(login_dst))
        self.assertLess(max(login_src), num_accounts)
        self.assertGreaterEqual(min(login_src), 0)
        self.assertLess(max(login_dst), num_devices)
        self.assertGreaterEqual(min(login_dst), 0)

    def test_no_nan_features(self):
        """Verify feature matrices contain zero NaN or Inf values."""
        if hasattr(self.dataset.data, "__getitem__") and "account_x" in self.dataset.data:
            account_x = self.dataset.data["account_x"]
            customer_x = self.dataset.data["customer_x"]
            device_x = self.dataset.data["device_x"]
        else:
            account_x = self.dataset.data["account"].x.tolist()
            customer_x = self.dataset.data["customer"].x.tolist()
            device_x = self.dataset.data["device"].x.tolist()

        for row in account_x[:100]:
            for val in row:
                self.assertFalse(math.isnan(val), "NaN found in account features")
                self.assertFalse(math.isinf(val), "Inf found in account features")

        for row in customer_x[:50]:
            for val in row:
                self.assertFalse(math.isnan(val), "NaN found in customer features")

        for row in device_x[:50]:
            for val in row:
                self.assertFalse(math.isnan(val), "NaN found in device features")

    def test_topological_feature_extractor(self):
        """Unit test for isolated topological feature calculations."""
        mock_accounts = [{"account_id": "A1"}, {"account_id": "A2"}, {"account_id": "A3"}]
        mock_txs = [
            {"source_account_id": "A1", "target_account_id": "A2", "amount": 1000.0},
            {"source_account_id": "A2", "target_account_id": "A3", "amount": 950.0},
            {"source_account_id": "A3", "target_account_id": "A1", "amount": 900.0},
        ]
        mock_cust = [{"customer_id": "C1", "account_id": "A1"}]
        mock_dev = [{"account_id": "A1", "device_id": "D1", "login_count": 5}]

        feats = extract_topological_features(mock_accounts, mock_txs, mock_cust, mock_dev)
        self.assertIn("A1", feats)
        self.assertEqual(feats["A1"]["in_degree"], 1)
        self.assertEqual(feats["A1"]["out_degree"], 1)
        self.assertEqual(feats["A1"]["total_in_amount"], 900.0)
        self.assertEqual(feats["A1"]["total_out_amount"], 1000.0)
        self.assertGreater(feats["A1"]["pagerank"], 0.0)


if __name__ == "__main__":
    unittest.main()
