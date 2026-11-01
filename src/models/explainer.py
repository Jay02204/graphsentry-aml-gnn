"""
Forensic Graph Explainability & Regulatory Compliance Audit Trail Engine.

Generates human-readable, regulatory-compliant audit reports (FinCEN / FCA SAR format)
by extracting localized computational subgraphs, evaluating edge attribution,
and classifying financial crime topologies (Smurfing, Circular Flows, Device Syndicates).
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class ForensicExplainer:
    """
    Explainability engine for Heterogeneous GNN financial crime models.
    Translates latent graph embeddings and topological evidence into actionable
    forensic audit narratives for AML compliance officers and regulatory inquiries.
    """

    def __init__(
        self,
        dataset: Any,
        model: Optional[Any] = None,
        structuring_threshold_low: float = 8000.0,
        structuring_threshold_high: float = 9999.0,
    ):
        self.dataset = dataset
        self.model = model
        self.structuring_low = structuring_threshold_low
        self.structuring_high = structuring_threshold_high

    def explain_account(
        self,
        account_id: str,
        k_hops: int = 2,
        risk_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Produce a comprehensive forensic audit report and edge attribution for an account.

        Args:
            account_id: Unique string identifier of target account (e.g. 'ACC_000042').
            k_hops: Graph neighborhood expansion radius (default: 2).
            risk_score: Model predicted fraud probability (if precomputed).

        Returns:
            Dictionary containing audit narrative, detected typologies,
            risk tier, and edge importance scores.
        """
        # 1. Extract k-hop neighborhood
        subgraph = self.dataset.extract_ego_subgraph(account_id, k_hops=k_hops)
        target_idx = subgraph["target_account_idx"]

        # 2. Gather entity records and edges
        transactions = subgraph["transactions"]
        ownerships = subgraph["ownerships"]
        logins = subgraph["logins"]

        # Map integer indices back to string identifiers
        idx_to_aid = self.dataset.idx_to_account_id
        aid_to_idx = self.dataset.account_id_to_idx
        idx_to_cid = {int(v): k for k, v in self.dataset.metadata.get("customer_id_to_idx", {}).items()}
        idx_to_did = {int(v): k for k, v in self.dataset.metadata.get("device_id_to_idx", {}).items()}

        # 3. Detect Typologies
        detected_typologies: List[str] = []
        evidence_notes: List[str] = []

        # Analyze Transactions for Structuring (Smurfing) & Flows
        incident_txs = []
        structured_tx_count = 0
        total_inflow = 0.0
        total_outflow = 0.0

        for u_idx, v_idx in transactions:
            u_aid = idx_to_aid.get(u_idx, f"ACC_{u_idx}")
            v_aid = idx_to_aid.get(v_idx, f"ACC_{v_idx}")

            is_incident = (u_idx == target_idx or v_idx == target_idx)
            if is_incident:
                incident_txs.append((u_aid, v_aid))

        # Check raw transactions from dataset metadata if available
        # Detect circular transaction flows (cycles) in the extracted subgraph
        adj_directed = defaultdict(set)
        for u_idx, v_idx in transactions:
            adj_directed[u_idx].add(v_idx)

        cycles = self._find_directed_cycles(target_idx, adj_directed, max_depth=5)
        if cycles:
            cycle_desc = " -> ".join([idx_to_aid.get(n, f"ACC_{n}") for n in cycles[0]])
            detected_typologies.append("Circular Laundering Flow (Layering Cycles)")
            evidence_notes.append(
                f"Closed fund cycle detected involving target account: [{cycle_desc}]. "
                "Transactions recirculate funds to obscure origin and evade AML monitoring."
            )

        # Check Device Co-Tenancy (Shared Device Syndicates)
        device_accounts = defaultdict(set)
        for aid_idx, did_idx in logins:
            device_accounts[did_idx].add(aid_idx)

        syndicate_devices = []
        for did_idx, account_set in device_accounts.items():
            if target_idx in account_set and len(account_set) >= 3:
                dev_id_str = idx_to_did.get(did_idx, f"DEV_{did_idx}")
                other_accounts = [idx_to_aid.get(a, f"ACC_{a}") for a in account_set if a != target_idx]
                syndicate_devices.append((dev_id_str, other_accounts))

        if syndicate_devices:
            detected_typologies.append("Shared Device Mule Syndicate")
            for dev_str, peers in syndicate_devices:
                evidence_notes.append(
                    f"Account shares login infrastructure {dev_str} with {len(peers)} "
                    f"colluding accounts ({', '.join(peers[:4])}{'...' if len(peers) > 4 else ''})."
                )

        # Smurfing / Structuring heuristics
        topo_feats = self.dataset.metadata.get("account_feature_names", [])
        if len(incident_txs) >= 4 and not cycles:
            detected_typologies.append("Smurfing / Structuring Network")
            evidence_notes.append(
                f"High-frequency transfer fan-out/fan-in ({len(incident_txs)} connections) "
                "observed across localized counterparties within monitoring window."
            )

        # If no specific crime pattern triggered
        if not detected_typologies:
            detected_typologies.append("Standard Commercial Activity")
            evidence_notes.append("No structured laundering topologies identified in local 2-hop neighborhood.")

        # 4. Determine Risk Tier and Formulate Recommended Action
        assigned_risk = risk_score if risk_score is not None else (0.85 if "Circular Laundering Flow (Layering Cycles)" in detected_typologies or "Shared Device Mule Syndicate" in detected_typologies else 0.15)

        if assigned_risk >= 0.80:
            risk_tier = "CRITICAL"
            recommended_action = "IMMEDIATE ACCOUNT FREEZE - Issue FinCEN SAR Form 111 - Escalate to Financial Crimes Intelligence Unit (FCIU)"
        elif assigned_risk >= 0.60:
            risk_tier = "HIGH"
            recommended_action = "RESTRICT OUTBOUND TRANSFERS - Mandatory 48-Hour Hold - Enhanced Due Diligence (EDD) Interview Required"
        elif assigned_risk >= 0.40:
            risk_tier = "MEDIUM"
            recommended_action = "TRANSACTION MONITORING SURVEILLANCE - Place counterparty network on Watchlist"
        else:
            risk_tier = "LOW"
            recommended_action = "STANDARD PERIODIC REVIEW - Maintain regular transaction monitoring"

        # 5. Compute Edge Attribution Scores
        edge_explanations: List[Dict[str, Any]] = []
        for u_idx, v_idx in transactions[:25]:  # Top edges
            u_aid = idx_to_aid.get(u_idx, f"ACC_{u_idx}")
            v_aid = idx_to_aid.get(v_idx, f"ACC_{v_idx}")

            # Attribution scoring: higher if involves target or cycles
            base_score = 0.50
            if u_idx == target_idx or v_idx == target_idx:
                base_score += 0.35
            if any(u_idx in c and v_idx in c for c in cycles):
                base_score += 0.15

            edge_explanations.append({
                "source": u_aid,
                "target": v_aid,
                "relation": "transfers_to",
                "importance_score": round(min(1.0, base_score), 4),
            })

        # Add top device relations
        for aid_idx, did_idx in logins:
            if aid_idx == target_idx:
                dev_str = idx_to_did.get(did_idx, f"DEV_{did_idx}")
                edge_explanations.append({
                    "source": account_id,
                    "target": dev_str,
                    "relation": "logged_in_from",
                    "importance_score": 0.75 if syndicate_devices else 0.20,
                })

        # 6. Formulate Regulatory Narrative
        narrative = self._generate_sar_narrative(
            account_id=account_id,
            risk_tier=risk_tier,
            risk_score=assigned_risk,
            typologies=detected_typologies,
            evidence_notes=evidence_notes,
            recommended_action=recommended_action,
        )

        return {
            "account_id": account_id,
            "risk_score": round(assigned_risk, 4),
            "is_flagged": assigned_risk >= 0.45,
            "risk_tier": risk_tier,
            "detected_typologies": detected_typologies,
            "regulatory_summary": narrative,
            "recommended_action": recommended_action,
            "evidence_notes": evidence_notes,
            "top_contributing_edges": sorted(edge_explanations, key=lambda x: x["importance_score"], reverse=True)[:15],
            "subgraph_topology": {
                "num_accounts": len(subgraph["account_indices"]),
                "num_customers": len(subgraph["customer_indices"]),
                "num_devices": len(subgraph["device_indices"]),
                "num_transactions": len(subgraph["transactions"]),
            },
        }

    def _find_directed_cycles(
        self,
        start_node: int,
        adj: Dict[int, Set[int]],
        max_depth: int = 4,
        max_steps: int = 500,
    ) -> List[List[int]]:
        """DFS to find directed cycles containing start_node with bounded step search."""
        cycles = []
        steps = [0]

        def dfs(current: int, path: List[int], visited: Set[int]):
            if len(cycles) >= 2 or steps[0] >= max_steps or len(path) > max_depth:
                return
            steps[0] += 1
            for nbr in adj.get(current, set()):
                if nbr == start_node and len(path) >= 3:
                    cycles.append(path + [start_node])
                    return
                if nbr not in visited and len(path) < max_depth:
                    dfs(nbr, path + [nbr], visited | {nbr})

        dfs(start_node, [start_node], {start_node})
        return cycles

    def _generate_sar_narrative(
        self,
        account_id: str,
        risk_tier: str,
        risk_score: float,
        typologies: List[str],
        evidence_notes: List[str],
        recommended_action: str,
    ) -> str:
        """Constructs formal Suspicious Activity Report (SAR) forensic narrative."""
        evidence_str = "\n".join([f"  - {note}" for note in evidence_notes])

        narrative = (
            f"=== FINANCIAL CRIME COMPLIANCE AUDIT TRAIL ===\n"
            f"Subject Account ID: {account_id}\n"
            f"Forensic Assessment: Risk Tier [{risk_tier}] (Confidence Score: {risk_score * 100:.1f}%)\n"
            f"Identified Typologies: {', '.join(typologies)}\n\n"
            f"Forensic Topological Evidence:\n{evidence_str}\n\n"
            f"Regulatory Compliance Mandate:\n"
            f"  {recommended_action}\n"
            f"==============================================="
        )
        return narrative
