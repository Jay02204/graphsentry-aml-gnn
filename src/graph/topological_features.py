"""
Topological Feature Extraction Engine for Financial Transaction Graphs.

Extracts graph-theoretic and money-laundering structural indicators from directed
transaction topologies and bipartite entity relationships:
- Degree centralities (in-degree, out-degree, total degree, in/out ratio)
- Flow kinetics (total in/out amount, net flow, flow ratio, average transaction sizes)
- Directed PageRank (authority & sink node identification)
- Local Clustering Coefficient (clique & circular ring detection)
- Bilateral Reciprocity (ping-pong laundering flows)
- Entity co-tenancy (shared device degree, customer portfolio fan-out)
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False


def _compute_pagerank_power_iteration(
    nodes: List[str],
    adj_out: Dict[str, List[str]],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute PageRank via power iteration algorithm as a standalone
    fallback if NetworkX is not installed.
    """
    n = len(nodes)
    if n == 0:
        return {}

    rank = {node: 1.0 / n for node in nodes}
    node_set = set(nodes)

    for _ in range(max_iter):
        next_rank = {node: (1.0 - alpha) / n for node in nodes}
        dangling_sum = sum(rank[node] for node in nodes if len(adj_out.get(node, [])) == 0)
        dangling_contrib = alpha * (dangling_sum / n)

        for u in nodes:
            out_neighbors = adj_out.get(u, [])
            if out_neighbors:
                share = (alpha * rank[u]) / len(out_neighbors)
                for v in out_neighbors:
                    if v in node_set:
                        next_rank[v] += share

        for node in nodes:
            next_rank[node] += dangling_contrib

        err = sum(abs(next_rank[node] - rank[node]) for node in nodes)
        rank = next_rank
        if err < tol:
            break

    return rank


def _compute_clustering_coefficient(
    nodes: List[str],
    undirected_adj: Dict[str, set],
) -> Dict[str, float]:
    """
    Compute local clustering coefficient on undirected projection of the graph.
    C_v = (2 * triangles_v) / (k_v * (k_v - 1))
    """
    clustering = {}
    for v in nodes:
        neighbors = list(undirected_adj.get(v, set()))
        k = len(neighbors)
        if k < 2:
            clustering[v] = 0.0
            continue

        triangles = 0
        for i in range(k):
            u1 = neighbors[i]
            u1_nbrs = undirected_adj.get(u1, set())
            for j in range(i + 1, k):
                u2 = neighbors[j]
                if u2 in u1_nbrs:
                    triangles += 1

        clustering[v] = (2.0 * triangles) / (k * (k - 1))
    return clustering


def extract_topological_features(
    accounts: List[Dict[str, Any]],
    transactions: List[Dict[str, Any]],
    customer_accounts: List[Dict[str, Any]],
    account_devices: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Extract comprehensive topological and multi-relational features for all accounts.

    Args:
        accounts: List of account dictionaries containing 'account_id'.
        transactions: List of transaction dictionaries with 'source_account_id',
                      'target_account_id', and 'amount'.
        customer_accounts: List of customer-account linkage records.
        account_devices: List of account-device login records.

    Returns:
        Dictionary mapping account_id -> Dict of extracted topological features.
    """
    account_ids = [str(a["account_id"]) for a in accounts]
    account_set = set(account_ids)

    # 1. Build Adjacency and Flow structures
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    in_amount = defaultdict(float)
    out_amount = defaultdict(float)
    adj_out = defaultdict(list)
    adj_in = defaultdict(list)
    undirected_adj = defaultdict(set)
    direct_edges = set()

    for tx in transactions:
        src = str(tx["source_account_id"])
        dst = str(tx["target_account_id"])
        try:
            amt = float(tx.get("amount", 0.0))
        except (ValueError, TypeError):
            amt = 0.0

        if src in account_set:
            out_degree[src] += 1
            out_amount[src] += amt
            adj_out[src].append(dst)
            undirected_adj[src].add(dst)

        if dst in account_set:
            in_degree[dst] += 1
            in_amount[dst] += amt
            adj_in[dst].append(src)
            undirected_adj[dst].add(src)

        direct_edges.add((src, dst))

    # 2. Compute PageRank
    if _HAS_NETWORKX:
        G = nx.DiGraph()
        G.add_nodes_from(account_ids)
        for tx in transactions:
            src = str(tx["source_account_id"])
            dst = str(tx["target_account_id"])
            amt = float(tx.get("amount", 1.0))
            if src in account_set and dst in account_set:
                G.add_edge(src, dst, weight=amt)
        try:
            pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            pagerank_scores = _compute_pagerank_power_iteration(account_ids, adj_out)

        # Clustering coefficient on undirected graph
        G_undir = G.to_undirected()
        clustering_scores = nx.clustering(G_undir)
    else:
        pagerank_scores = _compute_pagerank_power_iteration(account_ids, adj_out)
        clustering_scores = _compute_clustering_coefficient(account_ids, undirected_adj)

    # 3. Calculate Multi-Relational Fan-outs:
    # Customer ownership: how many accounts does the owner customer have?
    cust_to_accounts = defaultdict(set)
    acct_to_cust = {}
    for ca in customer_accounts:
        cid = str(ca["customer_id"])
        aid = str(ca["account_id"])
        cust_to_accounts[cid].add(aid)
        acct_to_cust[aid] = cid

    # Device co-tenancy: how many accounts use the devices linked to this account?
    dev_to_accounts = defaultdict(set)
    dev_login_counts = defaultdict(int)
    acct_to_devs = defaultdict(set)

    for ad in account_devices:
        aid = str(ad["account_id"])
        did = str(ad["device_id"])
        try:
            logins = int(ad.get("login_count", 1))
        except (ValueError, TypeError):
            logins = 1

        dev_to_accounts[did].add(aid)
        dev_login_counts[did] += logins
        acct_to_devs[aid].add(did)

    # 4. Assemble Topological Feature Dictionary
    features: Dict[str, Dict[str, float]] = {}

    for aid in account_ids:
        k_in = in_degree[aid]
        k_out = out_degree[aid]
        k_total = k_in + k_out
        deg_ratio = (k_in + 1.0) / (k_out + 1.0)

        tot_in_amt = in_amount[aid]
        tot_out_amt = out_amount[aid]
        net_flow = tot_in_amt - tot_out_amt
        flow_ratio = (tot_in_amt + 1.0) / (tot_out_amt + 1.0)
        avg_in_amt = tot_in_amt / max(k_in, 1)
        avg_out_amt = tot_out_amt / max(k_out, 1)

        # Bilateral reciprocity (counterparty ping-pong flows)
        reciprocal_count = 0
        for out_nbr in set(adj_out.get(aid, [])):
            if (out_nbr, aid) in direct_edges:
                reciprocal_count += 1

        # Customer entity metrics
        cust_id = acct_to_cust.get(aid)
        cust_total_accounts = len(cust_to_accounts[cust_id]) if cust_id else 1

        # Device syndicate metrics
        linked_devs = acct_to_devs.get(aid, set())
        max_shared_accounts = 1
        total_syndicate_logins = 0
        for did in linked_devs:
            dev_accts = len(dev_to_accounts[did])
            if dev_accts > max_shared_accounts:
                max_shared_accounts = dev_accts
            total_syndicate_logins += dev_login_counts[did]

        features[aid] = {
            "in_degree": float(k_in),
            "out_degree": float(k_out),
            "total_degree": float(k_total),
            "degree_ratio": float(deg_ratio),
            "total_in_amount": float(tot_in_amt),
            "total_out_amount": float(tot_out_amt),
            "net_flow": float(net_flow),
            "flow_ratio": float(flow_ratio),
            "avg_in_amount": float(avg_in_amt),
            "avg_out_amount": float(avg_out_amt),
            "pagerank": float(pagerank_scores.get(aid, 0.0)),
            "clustering_coefficient": float(clustering_scores.get(aid, 0.0)),
            "reciprocal_count": float(reciprocal_count),
            "customer_total_accounts": float(cust_total_accounts),
            "max_device_shared_accounts": float(max_shared_accounts),
            "device_login_intensity": float(total_syndicate_logins),
        }

    return features
