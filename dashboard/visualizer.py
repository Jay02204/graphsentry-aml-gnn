"""
Interactive Forensic Graph Visualizer.

Generates standalone, interactive HTML/JS network visualizers for AML investigator dashboards:
- Color-coded node topologies: Target Node, Mules, Licit Accounts, Customers, Devices
- Weighted directed transaction flow edges with tooltip inspection
- Force-directed physics layout powered by Vis.js / Pyvis
- Integrated forensic investigation banner with risk tiers and detected typologies
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except ImportError:
    _HAS_PYVIS = False


def generate_subgraph_html(
    explanation_result: Dict[str, Any],
    output_path: Optional[Path] = None,
    height: str = "750px",
    width: str = "100%",
) -> str:
    """
    Renders an interactive forensic network visualization HTML file.

    Args:
        explanation_result: Output dictionary from ForensicExplainer.explain_account.
        output_path: Filepath where HTML will be saved.
        height: Canvas height.
        width: Canvas width.

    Returns:
        HTML string representation of the interactive graph.
    """
    target_aid = explanation_result["account_id"]
    risk_score = explanation_result.get("risk_score", 0.5)
    risk_tier = explanation_result.get("risk_tier", "MEDIUM")
    typologies = explanation_result.get("detected_typologies", ["Standard Activity"])
    top_edges = explanation_result.get("top_contributing_edges", [])
    evidence_notes = explanation_result.get("evidence_notes", [])

    # Color mapping for risk tiers
    tier_colors = {
        "CRITICAL": "#E63946",
        "HIGH": "#F4A261",
        "MEDIUM": "#E9C46A",
        "LOW": "#2A9D8F",
    }
    banner_color = tier_colors.get(risk_tier, "#E63946")

    # Generate vis.js interactive payload
    nodes_data = []
    edges_data = []
    seen_nodes = set()

    # Target Node
    nodes_data.append({
        "id": target_aid,
        "label": f"TARGET: {target_aid}",
        "title": f"<b>Target Account:</b> {target_aid}<br><b>Risk Tier:</b> {risk_tier}<br><b>Risk Score:</b> {risk_score * 100:.1f}%",
        "color": {"background": "#FF0055", "border": "#FFFFFF", "highlight": {"background": "#FF3377"}},
        "shape": "star",
        "size": 35,
        "borderWidth": 3,
        "font": {"color": "#FFFFFF", "size": 14, "bold": True},
    })
    seen_nodes.add(target_aid)

    # Process edges and collect peripheral nodes
    for edge in top_edges:
        src = edge["source"]
        dst = edge["target"]
        rel = edge.get("relation", "transfers_to")
        importance = edge.get("importance_score", 0.5)

        for n_id in [src, dst]:
            if n_id not in seen_nodes:
                seen_nodes.add(n_id)
                if n_id.startswith("ACC_"):
                    nodes_data.append({
                        "id": n_id,
                        "label": n_id,
                        "title": f"<b>Counterparty Account:</b> {n_id}",
                        "color": {"background": "#457B9D", "border": "#1D3557"},
                        "shape": "dot",
                        "size": 20,
                        "font": {"color": "#E0E0E0", "size": 11},
                    })
                elif n_id.startswith("CUST_"):
                    nodes_data.append({
                        "id": n_id,
                        "label": n_id,
                        "title": f"<b>Customer Entity:</b> {n_id}",
                        "color": {"background": "#9D4EDD", "border": "#5A189A"},
                        "shape": "diamond",
                        "size": 22,
                        "font": {"color": "#E0E0E0", "size": 11},
                    })
                elif n_id.startswith("DEV_"):
                    nodes_data.append({
                        "id": n_id,
                        "label": n_id,
                        "title": f"<b>Device Hardware:</b> {n_id}",
                        "color": {"background": "#2A9D8F", "border": "#264653"},
                        "shape": "square",
                        "size": 22,
                        "font": {"color": "#E0E0E0", "size": 11},
                    })
                else:
                    nodes_data.append({
                        "id": n_id,
                        "label": n_id,
                        "color": "#6C757D",
                        "shape": "ellipse",
                        "size": 15,
                    })

        # Edge Styling
        if rel == "transfers_to":
            edge_color = "#E63946" if importance > 0.6 else "#A8DADC"
            edges_data.append({
                "from": src,
                "to": dst,
                "arrows": "to",
                "label": f"imp: {importance:.2f}",
                "title": f"<b>Transfer:</b> {src} &rarr; {dst}<br><b>Attribution:</b> {importance:.3f}",
                "color": {"color": edge_color, "highlight": "#FFD166"},
                "width": max(1, int(importance * 5)),
            })
        elif rel == "owns":
            edges_data.append({
                "from": src,
                "to": dst,
                "arrows": "to",
                "label": "owns",
                "color": {"color": "#C77DFF"},
                "dashes": True,
                "width": 2,
            })
        elif rel == "logged_in_from":
            edges_data.append({
                "from": src,
                "to": dst,
                "arrows": "to",
                "label": "login",
                "color": {"color": "#2A9D8F"},
                "dashes": [4, 4],
                "width": 2,
            })

    # Build standalone HTML template with Vis.js CDN
    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)
    typologies_str = ", ".join(typologies)
    evidence_items_html = "".join([f"<li>{note}</li>" for note in evidence_notes])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GraphSentry Forensic Investigation: {target_aid}</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 0;
      background-color: #0F172A;
      color: #F8FAFC;
    }}
    .header {{
      background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
      padding: 20px 28px;
      border-bottom: 2px solid {banner_color};
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
    .badge {{
      display: inline-block;
      padding: 5px 12px;
      font-size: 13px;
      font-weight: 700;
      border-radius: 6px;
      background-color: {banner_color};
      color: #FFFFFF;
      margin-left: 10px;
    }}
    .metrics-bar {{
      display: flex;
      gap: 20px;
      margin-top: 10px;
      font-size: 14px;
      color: #94A3B8;
    }}
    .metrics-bar span strong {{
      color: #F8FAFC;
    }}
    .container {{
      display: flex;
      height: calc(100vh - 120px);
    }}
    #network-canvas {{
      flex: 1;
      height: 100%;
      background-color: #090D16;
    }}
    .sidebar {{
      width: 380px;
      background-color: #1E293B;
      padding: 24px;
      overflow-y: auto;
      border-left: 1px solid #334155;
    }}
    h3 {{
      margin-top: 0;
      font-size: 16px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #38BDF8;
      border-bottom: 1px solid #334155;
      padding-bottom: 8px;
    }}
    ul.evidence-list {{
      padding-left: 20px;
      font-size: 13px;
      line-height: 1.6;
      color: #CBD5E1;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .legend-color {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
    }}
    .sar-mandate {{
      background-color: rgba(230, 57, 70, 0.15);
      border-left: 4px solid {banner_color};
      padding: 12px;
      border-radius: 4px;
      font-size: 13px;
      margin-top: 15px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h2 style="margin: 0; display: flex; align-items: center;">
      GraphSentry AML Forensic Subgraph
      <span class="badge">{risk_tier} RISK ({risk_score*100:.1f}%)</span>
    </h2>
    <div class="metrics-bar">
      <span>Target: <strong>{target_aid}</strong></span>
      <span>Typologies: <strong>{typologies_str}</strong></span>
      <span>Nodes: <strong>{len(nodes_data)}</strong></span>
      <span>Edges: <strong>{len(edges_data)}</strong></span>
    </div>
  </div>

  <div class="container">
    <div id="network-canvas"></div>
    <div class="sidebar">
      <h3>Regulatory Audit Findings</h3>
      <ul class="evidence-list">
        {evidence_items_html}
      </ul>

      <div class="sar-mandate">
        <strong>Compliance Mandate:</strong><br>
        {explanation_result.get("recommended_action", "Proceed with standard inquiry.")}
      </div>

      <h3 style="margin-top: 24px;">Graph Schema Legend</h3>
      <div class="legend-item">
        <div class="legend-color" style="background-color: #FF0055;"></div>
        <span>Target Account (Focal Point)</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background-color: #457B9D;"></div>
        <span>Counterparty Account</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background-color: #9D4EDD;"></div>
        <span>Customer Entity (owns)</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background-color: #2A9D8F;"></div>
        <span>Hardware Device (logged_in_from)</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background-color: #E63946;"></div>
        <span>High-Salience Transfer Edge</span>
      </div>
    </div>
  </div>

  <script type="text/javascript">
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});

    const container = document.getElementById("network-canvas");
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      physics: {{
        solver: "forceAtlas2Based",
        forceAtlas2Based: {{
          gravitationalConstant: -70,
          centralGravity: 0.015,
          springLength: 120,
          springConstant: 0.08
        }},
        maxVelocity: 50,
        minVelocity: 0.1,
        stabilization: {{ iterations: 150 }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 100,
        zoomView: true,
        dragView: true
      }}
    }};

    const network = new vis.Network(container, data, options);
  </script>
</body>
</html>
"""

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
