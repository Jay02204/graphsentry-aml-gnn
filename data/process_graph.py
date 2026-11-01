"""
Graph Construction and Preprocessing Pipeline.

Transforms raw multi-relational CSV records into PyTorch Geometric HeteroData:
- Extracts topological graph features (centrality, degree ratio, flow kinetics)
- One-hot encodes categorical metadata & normalizes continuous tensors
- Constructs canonical and reverse edge index matrices for bidirectional message passing
- Generates stratified train, validation, and test split masks
- Persists HeteroData tensor and JSON metadata for reproducible serving
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.topological_features import extract_topological_features
from src.config import settings

try:
    import torch
    from torch_geometric.data import HeteroData
    _HAS_TORCH_PYG = True
except ImportError:
    _HAS_TORCH_PYG = False


def _read_csv(filepath: Path) -> List[Dict[str, str]]:
    """Helper to read CSV into a list of row dictionaries."""
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _normalize_features(features: List[List[float]]) -> List[List[float]]:
    """Standardize feature matrix (mean=0, std=1) with epsilon guard."""
    if not features or not features[0]:
        return features

    num_samples = len(features)
    num_features = len(features[0])

    normalized = [[0.0] * num_features for _ in range(num_samples)]

    for j in range(num_features):
        col_vals = [features[i][j] for i in range(num_samples)]
        mean_val = sum(col_vals) / num_samples
        variance = sum((x - mean_val) ** 2 for x in col_vals) / num_samples
        std_val = math.sqrt(variance)
        if std_val < 1e-8:
            std_val = 1.0

        for i in range(num_samples):
            normalized[i][j] = (features[i][j] - mean_val) / std_val

    return normalized


def build_hetero_graph(
    raw_dir: Path = settings.RAW_DATA_DIR,
    output_dir: Path = settings.PROCESSED_DATA_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Any:
    """
    Constructs HeteroData from raw CSV tables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Processing raw AML CSV data from: {raw_dir}")

    # 1. Load CSV Tables
    accounts_rows = _read_csv(raw_dir / "accounts.csv")
    customers_rows = _read_csv(raw_dir / "customers.csv")
    devices_rows = _read_csv(raw_dir / "devices.csv")
    transactions_rows = _read_csv(raw_dir / "transactions.csv")
    customer_accounts_rows = _read_csv(raw_dir / "customer_accounts.csv")
    account_devices_rows = _read_csv(raw_dir / "account_devices.csv")

    print(f"[*] Ingested {len(accounts_rows):,} accounts, {len(customers_rows):,} customers, "
          f"{len(devices_rows):,} devices, {len(transactions_rows):,} transactions.")

    # 2. Extract Topological Graph Features for Accounts
    print("[*] Extracting graph-topological kinetics and centrality features...")
    topo_features = extract_topological_features(
        accounts=accounts_rows,
        transactions=transactions_rows,
        customer_accounts=customer_accounts_rows,
        account_devices=account_devices_rows,
    )

    # 3. Create Contiguous ID Mappings (str -> int)
    account_id_to_idx = {row["account_id"]: i for i, row in enumerate(accounts_rows)}
    idx_to_account_id = {i: row["account_id"] for i, row in enumerate(accounts_rows)}

    customer_id_to_idx = {row["customer_id"]: i for i, row in enumerate(customers_rows)}
    idx_to_customer_id = {i: row["customer_id"] for i, row in enumerate(customers_rows)}

    device_id_to_idx = {row["device_id"]: i for i, row in enumerate(devices_rows)}
    idx_to_device_id = {i: row["device_id"] for i, row in enumerate(devices_rows)}

    # 4. Construct Account Node Feature Matrix
    account_types = ["checking", "savings", "business", "crypto_onramp"]
    topo_keys = [
        "in_degree", "out_degree", "total_degree", "degree_ratio",
        "total_in_amount", "total_out_amount", "net_flow", "flow_ratio",
        "avg_in_amount", "avg_out_amount", "pagerank", "clustering_coefficient",
        "reciprocal_count", "customer_total_accounts", "max_device_shared_accounts",
        "device_login_intensity"
    ]

    raw_account_x = []
    account_y = []

    for row in accounts_rows:
        aid = row["account_id"]
        # One-hot account type
        atype = row.get("account_type", "checking")
        one_hot_type = [1.0 if atype == t else 0.0 for t in account_types]

        # Log-transformed balance
        init_bal = float(row.get("initial_balance", 0.0))
        log_bal = math.log1p(max(0.0, init_bal))

        # Topological attributes
        aid_topo = topo_features.get(aid, {})
        topo_vals = [float(aid_topo.get(k, 0.0)) for k in topo_keys]

        full_feats = one_hot_type + [log_bal] + topo_vals
        raw_account_x.append(full_feats)
        account_y.append(int(row.get("is_mule", 0)))

    norm_account_x = _normalize_features(raw_account_x)

    # 5. Construct Customer Node Feature Matrix
    raw_customer_x = []
    for row in customers_rows:
        kyc = float(row.get("kyc_level", 1)) / 3.0
        risk = float(row.get("risk_score", 0.0))
        country_risk = float(row.get("country_risk", 0.0))
        raw_customer_x.append([kyc, risk, country_risk])

    norm_customer_x = _normalize_features(raw_customer_x)

    # 6. Construct Device Node Feature Matrix
    device_types = ["mobile_ios", "mobile_android", "desktop_windows", "desktop_mac", "server_linux"]
    raw_device_x = []
    for row in devices_rows:
        dtype = row.get("device_type", "mobile_android")
        one_hot_dev = [1.0 if dtype == dt else 0.0 for dt in device_types]
        is_proxy = float(row.get("is_proxy_or_vpn", 0.0))
        is_emu = float(row.get("is_emulator", 0.0))
        raw_device_x.append(one_hot_dev + [is_proxy, is_emu])

    norm_device_x = _normalize_features(raw_device_x)

    # 7. Construct Edge Indices and Attributes
    # ('account', 'transfers_to', 'account')
    transfer_src = []
    transfer_dst = []
    transfer_attr = []
    min_time = 0.0
    max_time = 1.0

    # Parse timestamps for temporal edge attributes
    parsed_txs = []
    for tx in transactions_rows:
        src = tx["source_account_id"]
        dst = tx["target_account_id"]
        if src in account_id_to_idx and dst in account_id_to_idx:
            amt = float(tx.get("amount", 0.0))
            log_amt = math.log1p(amt)
            parsed_txs.append((account_id_to_idx[src], account_id_to_idx[dst], amt, log_amt))

    for u_idx, v_idx, amt, log_amt in parsed_txs:
        transfer_src.append(u_idx)
        transfer_dst.append(v_idx)
        transfer_attr.append([amt, log_amt])

    transfer_attr_norm = _normalize_features(transfer_attr) if transfer_attr else []

    # ('customer', 'owns', 'account')
    owns_src = []
    owns_dst = []
    for ca in customer_accounts_rows:
        cid = ca["customer_id"]
        aid = ca["account_id"]
        if cid in customer_id_to_idx and aid in account_id_to_idx:
            owns_src.append(customer_id_to_idx[cid])
            owns_dst.append(account_id_to_idx[aid])

    # ('account', 'logged_in_from', 'device')
    login_src = []
    login_dst = []
    for ad in account_devices_rows:
        aid = ad["account_id"]
        did = ad["device_id"]
        if aid in account_id_to_idx and did in device_id_to_idx:
            login_src.append(account_id_to_idx[aid])
            login_dst.append(device_id_to_idx[did])

    # 8. Create Stratified Train / Val / Test Masks
    num_accounts = len(accounts_rows)
    mule_indices = [i for i, y in enumerate(account_y) if y == 1]
    licit_indices = [i for i, y in enumerate(account_y) if y == 0]

    import random
    rng = random.Random(seed)
    rng.shuffle(mule_indices)
    rng.shuffle(licit_indices)

    def split_indices(indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]

    train_mules, val_mules, test_mules = split_indices(mule_indices)
    train_licits, val_licits, test_licits = split_indices(licit_indices)

    train_idx_set = set(train_mules + train_licits)
    val_idx_set = set(val_mules + val_licits)
    test_idx_set = set(test_mules + test_licits)

    train_mask = [i in train_idx_set for i in range(num_accounts)]
    val_mask = [i in val_idx_set for i in range(num_accounts)]
    test_mask = [i in test_idx_set for i in range(num_accounts)]

    # 9. Assemble Native PyG HeteroData or Fallback Data Dictionary
    print("[*] Assembling HeteroData structures with bidirectional relations...")

    metadata = {
        "node_counts": {
            "account": num_accounts,
            "customer": len(customers_rows),
            "device": len(devices_rows),
        },
        "mule_count": sum(account_y),
        "licit_count": num_accounts - sum(account_y),
        "imbalance_ratio": sum(account_y) / num_accounts,
        "feature_dimensions": {
            "account": len(norm_account_x[0]),
            "customer": len(norm_customer_x[0]),
            "device": len(norm_device_x[0]),
        },
        "edge_counts": {
            "transfers_to": len(transfer_src),
            "owns": len(owns_src),
            "logged_in_from": len(login_src),
        },
        "splits": {
            "train_count": sum(train_mask),
            "val_count": sum(val_mask),
            "test_count": sum(test_mask),
        },
        "account_id_to_idx": account_id_to_idx,
        "idx_to_account_id": idx_to_account_id,
        "customer_id_to_idx": customer_id_to_idx,
        "device_id_to_idx": device_id_to_idx,
        "account_feature_names": account_types + ["log_initial_balance"] + topo_keys,
    }

    # Save metadata JSON
    metadata_path = output_dir / settings.METADATA_FILE
    with open(metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"    [+] Saved Graph Metadata -> {metadata_path}")

    # Build HeteroData if PyTorch & PyG are available
    if _HAS_TORCH_PYG:
        data = HeteroData()

        # Nodes
        data["account"].x = torch.tensor(norm_account_x, dtype=torch.float)
        data["account"].y = torch.tensor(account_y, dtype=torch.float)
        data["account"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data["account"].val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data["account"].test_mask = torch.tensor(test_mask, dtype=torch.bool)

        data["customer"].x = torch.tensor(norm_customer_x, dtype=torch.float)
        data["device"].x = torch.tensor(norm_device_x, dtype=torch.float)

        # Edges & Reverse Edges (for symmetric HeteroGNN message passing)
        data["account", "transfers_to", "account"].edge_index = torch.tensor(
            [transfer_src, transfer_dst], dtype=torch.long
        )
        if transfer_attr_norm:
            data["account", "transfers_to", "account"].edge_attr = torch.tensor(
                transfer_attr_norm, dtype=torch.float
            )

        data["account", "rev_transfers_to", "account"].edge_index = torch.tensor(
            [transfer_dst, transfer_src], dtype=torch.long
        )

        data["customer", "owns", "account"].edge_index = torch.tensor(
            [owns_src, owns_dst], dtype=torch.long
        )
        data["account", "rev_owns", "customer"].edge_index = torch.tensor(
            [owns_dst, owns_src], dtype=torch.long
        )

        data["account", "logged_in_from", "device"].edge_index = torch.tensor(
            [login_src, login_dst], dtype=torch.long
        )
        data["device", "rev_logged_in_from", "account"].edge_index = torch.tensor(
            [login_dst, login_src], dtype=torch.long
        )

        # Save PyG HeteroData
        graph_pt_path = output_dir / settings.HETERO_GRAPH_FILE
        torch.save(data, graph_pt_path)
        print(f"    [+] Saved PyG HeteroData -> {graph_pt_path}")
        return data
    else:
        # Fallback pure python dictionary structure
        raw_graph_data = {
            "account_x": norm_account_x,
            "account_y": account_y,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "customer_x": norm_customer_x,
            "device_x": norm_device_x,
            "transfer_src": transfer_src,
            "transfer_dst": transfer_dst,
            "owns_src": owns_src,
            "owns_dst": owns_dst,
            "login_src": login_src,
            "login_dst": login_dst,
        }
        json_graph_path = output_dir / "graph_tensors.json"
        with open(json_graph_path, mode="w", encoding="utf-8") as f:
            json.dump(raw_graph_data, f)
        print(f"    [+] Saved graph tensors fallback JSON -> {json_graph_path}")
        return raw_graph_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct HeteroData from AML CSV records.")
    parser.add_argument("--raw-dir", type=str, default=str(settings.RAW_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(settings.PROCESSED_DATA_DIR))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    build_hetero_graph(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
