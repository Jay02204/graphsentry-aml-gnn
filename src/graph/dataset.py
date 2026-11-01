"""
PyTorch Geometric Dataset and Dynamic Neighborhood Sampling Loader.

Provides:
- AMLHeteroDataset: High-level dataset abstraction over PyG HeteroData.
- Scalable k-hop NeighborLoader builders for GPU/CPU mini-batching.
- Ego-subgraph extraction for localized real-time inference and forensic audit.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config import settings

try:
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader
    _HAS_TORCH_PYG = True
except ImportError:
    _HAS_TORCH_PYG = False


class AMLHeteroDataset:
    """
    Manages heterogeneous AML graph data, mini-batch loaders, and ego-subgraphs.
    """

    def __init__(
        self,
        processed_dir: Path = settings.PROCESSED_DATA_DIR,
        device: Optional[str] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.metadata_path = self.processed_dir / settings.METADATA_FILE
        self.graph_pt_path = self.processed_dir / settings.HETERO_GRAPH_FILE
        self.fallback_json_path = self.processed_dir / "graph_tensors.json"

        if device is None and _HAS_TORCH_PYG:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device or "cpu"

        self.metadata = self._load_metadata()
        self.data = self._load_graph()
        self._build_subgraph_indices()

    def _build_subgraph_indices(self) -> None:
        """Pre-index adjacency lists for sub-millisecond ego-subgraph extraction."""
        from collections import defaultdict
        self._tx_neighbors = defaultdict(set)
        self._tx_pairs = set()

        if _HAS_TORCH_PYG and isinstance(self.data, HeteroData):
            tx_edges = self.data["account", "transfers_to", "account"].edge_index.cpu().numpy()
            owns_edges = self.data["customer", "owns", "account"].edge_index.cpu().numpy()
            login_edges = self.data["account", "logged_in_from", "device"].edge_index.cpu().numpy()
        else:
            tx_edges = [self.data["transfer_src"], self.data["transfer_dst"]]
            owns_edges = [self.data["owns_src"], self.data["owns_dst"]]
            login_edges = [self.data["login_src"], self.data["login_dst"]]

        num_tx = len(tx_edges[0])
        for i in range(num_tx):
            u = int(tx_edges[0][i])
            v = int(tx_edges[1][i])
            self._tx_neighbors[u].add(v)
            self._tx_neighbors[v].add(u)
            self._tx_pairs.add((u, v))

        self._acct_to_customers = defaultdict(list)
        num_owns = len(owns_edges[0])
        for i in range(num_owns):
            cid = int(owns_edges[0][i])
            aid = int(owns_edges[1][i])
            self._acct_to_customers[aid].append(cid)

        self._acct_to_devices = defaultdict(list)
        num_logins = len(login_edges[0])
        for i in range(num_logins):
            aid = int(login_edges[0][i])
            did = int(login_edges[1][i])
            self._acct_to_devices[aid].append(did)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load graph metadata and node id mapping tables."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {self.metadata_path}. "
                "Run 'python data/process_graph.py' first."
            )
        with open(self.metadata_path, mode="r", encoding="utf-8") as f:
            return json.load(f)

    def _load_graph(self) -> Any:
        """Load PyG HeteroData or tensor dictionary fallback."""
        if _HAS_TORCH_PYG and self.graph_pt_path.exists():
            print(f"[*] Loading PyG HeteroData from {self.graph_pt_path}...")
            return torch.load(self.graph_pt_path, weights_only=False)
        elif self.fallback_json_path.exists():
            print(f"[*] Loading fallback graph tensors from {self.fallback_json_path}...")
            with open(self.fallback_json_path, mode="r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise FileNotFoundError(
                f"No processed graph artifacts found in {self.processed_dir}. "
                "Execute data preprocessing pipeline before loading dataset."
            )

    @property
    def num_accounts(self) -> int:
        return self.metadata["node_counts"]["account"]

    @property
    def num_customers(self) -> int:
        return self.metadata["node_counts"]["customer"]

    @property
    def num_devices(self) -> int:
        return self.metadata["node_counts"]["device"]

    @property
    def account_id_to_idx(self) -> Dict[str, int]:
        return self.metadata["account_id_to_idx"]

    @property
    def idx_to_account_id(self) -> Dict[str, str]:
        # JSON keys are strings, convert to int-keyed dictionary
        return {int(k): v for k, v in self.metadata["idx_to_account_id"].items()}

    def get_loaders(
        self,
        batch_size: int = settings.TRAIN_BATCH_SIZE,
        eval_batch_size: int = settings.EVAL_BATCH_SIZE,
        num_neighbors: Optional[List[int]] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Instantiate NeighborLoader objects for train, val, and test splits.
        Mini-batches k-hop computation trees to prevent GPU out-of-memory.
        """
        if num_neighbors is None:
            num_neighbors = settings.NUM_NEIGHBORS

        if not _HAS_TORCH_PYG or not isinstance(self.data, HeteroData):
            return self._get_fallback_loaders(batch_size, eval_batch_size)

        # Mapping of edge types to sample neighbor counts
        # e.g. 15 neighbors for hop 1, 10 for hop 2
        neighbor_dict = {
            ("account", "transfers_to", "account"): num_neighbors,
            ("account", "rev_transfers_to", "account"): num_neighbors,
            ("customer", "owns", "account"): [num_neighbors[0]] * len(num_neighbors),
            ("account", "rev_owns", "customer"): [num_neighbors[0]] * len(num_neighbors),
            ("account", "logged_in_from", "device"): [num_neighbors[0]] * len(num_neighbors),
            ("device", "rev_logged_in_from", "account"): [num_neighbors[0]] * len(num_neighbors),
        }

        try:
            train_loader = NeighborLoader(
                self.data,
                num_neighbors=neighbor_dict,
                batch_size=batch_size,
                input_nodes=("account", self.data["account"].train_mask),
                shuffle=True,
            )
            # Verify if C++ sampler backend is functional
            _ = next(iter(train_loader))

            val_loader = NeighborLoader(
                self.data,
                num_neighbors=neighbor_dict,
                batch_size=eval_batch_size,
                input_nodes=("account", self.data["account"].val_mask),
                shuffle=False,
            )

            test_loader = NeighborLoader(
                self.data,
                num_neighbors=neighbor_dict,
                batch_size=eval_batch_size,
                input_nodes=("account", self.data["account"].test_mask),
                shuffle=False,
            )

            return train_loader, val_loader, test_loader
        except Exception:
            return self._get_hetero_batch_loaders(batch_size, eval_batch_size)

    def _get_hetero_batch_loaders(self, batch_size: int, eval_batch_size: int):
        """HeteroData batch loader when C++ pyg-lib/torch-sparse is unavailable on host."""
        class AccountTargetView:
            def __init__(self, full_account, indices):
                self.indices = indices
                self.batch_size = len(indices)
                self.y = full_account.y[indices]
                self.x = full_account.x[indices]

        class HeteroMiniBatch:
            def __init__(self, data: HeteroData, target_indices: Any):
                self.data = data
                self.target_indices = target_indices

            @property
            def x_dict(self):
                return self.data.x_dict

            @property
            def edge_index_dict(self):
                return self.data.edge_index_dict

            def to(self, device):
                self.data = self.data.to(device)
                self.target_indices = self.target_indices.to(device)
                return self

            def __getitem__(self, key):
                if key == "account":
                    return AccountTargetView(self.data["account"], self.target_indices)
                return self.data[key]

        class HeteroBatchLoader:
            def __init__(self, data: HeteroData, mask: Any, batch_sz: int, shuffle: bool = False):
                self.data = data
                self.batch_sz = batch_sz
                self.shuffle = shuffle
                if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
                    self.indices = torch.where(mask)[0]
                elif isinstance(mask, torch.Tensor):
                    self.indices = mask
                else:
                    self.indices = torch.tensor(mask, dtype=torch.long)

            def __iter__(self):
                indices = self.indices
                if self.shuffle:
                    perm = torch.randperm(len(indices))
                    indices = indices[perm]
                for i in range(0, len(indices), self.batch_sz):
                    chunk = indices[i : i + self.batch_sz]
                    yield HeteroMiniBatch(self.data, chunk)

            def __len__(self):
                return (len(self.indices) + self.batch_sz - 1) // self.batch_sz

        train_loader = HeteroBatchLoader(self.data, self.data["account"].train_mask, batch_size, shuffle=True)
        val_loader = HeteroBatchLoader(self.data, self.data["account"].val_mask, eval_batch_size, shuffle=False)
        test_loader = HeteroBatchLoader(self.data, self.data["account"].test_mask, eval_batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def _get_fallback_loaders(self, batch_size: int, eval_batch_size: int):
        """Fallback lightweight batch generator for CPU/test runner without PyG."""
        class SimpleBatchLoader:
            def __init__(self, indices: List[int], batch_sz: int, data_dict: Dict):
                self.indices = indices
                self.batch_sz = batch_sz
                self.data_dict = data_dict

            def __iter__(self):
                for i in range(0, len(self.indices), self.batch_sz):
                    batch_idx = self.indices[i : i + self.batch_sz]
                    yield {
                        "batch_indices": batch_idx,
                        "batch_size": len(batch_idx),
                    }

            def __len__(self):
                return (len(self.indices) + self.batch_sz - 1) // self.batch_sz

        if isinstance(self.data, dict):
            train_mask = self.data.get("train_mask", [])
            val_mask = self.data.get("val_mask", [])
            test_mask = self.data.get("test_mask", [])
        else:
            train_mask = [True] * self.num_accounts
            val_mask = [False] * self.num_accounts
            test_mask = [False] * self.num_accounts

        train_indices = [i for i, m in enumerate(train_mask) if m]
        val_indices = [i for i, m in enumerate(val_mask) if m]
        test_indices = [i for i, m in enumerate(test_mask) if m]

        return (
            SimpleBatchLoader(train_indices, batch_size, self.data),
            SimpleBatchLoader(val_indices, eval_batch_size, self.data),
            SimpleBatchLoader(test_indices, eval_batch_size, self.data),
        )

    def extract_ego_subgraph(
        self,
        account_id_or_idx: Union[str, int],
        k_hops: int = 2,
        max_neighbors: int = 15,
        max_total_nodes: int = 60,
    ) -> Dict[str, Any]:
        """
        Extract the k-hop ego-neighborhood around a target account.
        Caps fan-out sampling per node (similar to NeighborLoader) to prevent
        exponential neighborhood explosion on high-degree merchant nodes.
        """
        if isinstance(account_id_or_idx, str):
            if account_id_or_idx not in self.account_id_to_idx:
                raise KeyError(f"Account ID '{account_id_or_idx}' not found in graph.")
            target_idx = self.account_id_to_idx[account_id_or_idx]
            target_aid = account_id_or_idx
        else:
            target_idx = account_id_or_idx
            target_aid = self.idx_to_account_id.get(target_idx, f"ACC_{target_idx:06d}")

        current_accounts = {target_idx}
        frontier = {target_idx}

        for _ in range(k_hops):
            next_frontier = set()
            for a in frontier:
                all_nbrs = list(self._tx_neighbors.get(a, set()))
                # Sample top neighbors (capped at max_neighbors)
                sampled_nbrs = all_nbrs[:max_neighbors]
                for nbr in sampled_nbrs:
                    if len(current_accounts) < max_total_nodes:
                        current_accounts.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier
            if len(current_accounts) >= max_total_nodes:
                break

        # Collect internal transaction edges
        discovered_txs = []
        for u in current_accounts:
            for v in self._tx_neighbors.get(u, set()):
                if v in current_accounts and (u, v) in self._tx_pairs:
                    discovered_txs.append((u, v))

        # Collect connected customers and ownerships (capped)
        discovered_customers = set()
        discovered_ownerships = []
        for aid in list(current_accounts)[:30]:
            for cid in self._acct_to_customers.get(aid, [])[:3]:
                discovered_customers.add(cid)
                discovered_ownerships.append((cid, aid))

        # Collect connected devices and logins (capped)
        discovered_devices = set()
        discovered_logins = []
        for aid in list(current_accounts)[:30]:
            for did in self._acct_to_devices.get(aid, [])[:3]:
                discovered_devices.add(did)
                discovered_logins.append((aid, did))

        return {
            "target_account_id": target_aid,
            "target_account_idx": target_idx,
            "account_indices": list(current_accounts),
            "customer_indices": list(discovered_customers),
            "device_indices": list(discovered_devices),
            "transactions": discovered_txs,
            "ownerships": discovered_ownerships,
            "logins": discovered_logins,
        }
