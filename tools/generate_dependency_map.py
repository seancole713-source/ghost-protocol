#!/usr/bin/env python3
"""
🗺️ GHOST DEPENDENCY MAP GENERATOR

Scans Ghost codebase and generates live dependency graph (DAG).
Outputs: graph.json, graph.svg (visual map)

Usage:
    python3 tools/generate_dependency_map.py
"""

import json
import os
import re

# Node types
NODE_TYPES = {
    "endpoint": "🌐",
    "provider": "📡",
    "database": "💾",
    "cache": "⚡",
    "ai": "🤖",
    "broker": "💰",
    "metric": "📊",
    "telegram": "💬",
}


def extract_endpoints(filepath: str) -> list[tuple[str, str, str]]:
    """Extract FastAPI endpoints from wolf_app.py"""
    endpoints = []
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Find all @APP.{method} decorators
    pattern = r'@APP\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\)'
    matches = re.findall(pattern, content)

    for method, path in matches:
        endpoints.append((method.upper(), path, "endpoint"))

    return endpoints


def extract_providers(filepath: str) -> list[tuple[str, str]]:
    """Extract price/data providers"""
    providers = []
    provider_patterns = [
        r"alphavantage",
        r"polygon",
        r"yahoo",
        r"coingecko",
        r"binance",
        r"coinbase",
    ]

    with open(filepath, encoding="utf-8") as f:
        content = f.read().lower()

    for pattern in provider_patterns:
        if pattern in content:
            providers.append((pattern, "provider"))

    return providers


def extract_dependencies(wolf_app_path: str) -> dict:
    """Build dependency graph from wolf_app.py"""

    graph = {
        "nodes": [],
        "edges": [],
        "layers": {
            "ui": [],
            "api": [],
            "orchestration": [],
            "providers": [],
            "storage": [],
            "external": [],
        },
    }

    # Extract endpoints
    endpoints = extract_endpoints(wolf_app_path)
    for method, path, node_type in endpoints:
        node_id = f"{method}_{path.replace('/', '_')}"
        graph["nodes"].append(
            {"id": node_id, "label": f"{method} {path}", "type": node_type, "layer": "api"}
        )
        graph["layers"]["api"].append(node_id)

    # Extract providers
    providers = extract_providers(wolf_app_path)
    for name, node_type in providers:
        node_id = f"provider_{name}"
        graph["nodes"].append(
            {"id": node_id, "label": name.title(), "type": node_type, "layer": "providers"}
        )
        graph["layers"]["providers"].append(node_id)

    # Add known storage nodes
    storage_nodes = [
        ("redis_cache", "Redis Cache", "cache"),
        ("sqlite_wolf", "SQLite wolf.db", "database"),
        ("sqlite_predictions", "SQLite predictions.db", "database"),
        ("sqlite_risk", "SQLite risk.db", "database"),
    ]

    for node_id, label, node_type in storage_nodes:
        graph["nodes"].append(
            {"id": node_id, "label": label, "type": node_type, "layer": "storage"}
        )
        graph["layers"]["storage"].append(node_id)

    # Add AI nodes
    ai_nodes = [
        ("openai", "OpenAI GPT", "ai"),
        ("anthropic", "Claude", "ai"),
        ("ollama", "Ollama", "ai"),
    ]

    for node_id, label, node_type in ai_nodes:
        graph["nodes"].append(
            {"id": node_id, "label": label, "type": node_type, "layer": "external"}
        )
        graph["layers"]["external"].append(node_id)

    # Add broker node
    graph["nodes"].append(
        {"id": "alpaca_broker", "label": "Alpaca Broker", "type": "broker", "layer": "external"}
    )
    graph["layers"]["external"].append("alpaca_broker")

    # Add Telegram node
    graph["nodes"].append(
        {"id": "telegram_webhook", "label": "Telegram Bot", "type": "telegram", "layer": "external"}
    )
    graph["layers"]["external"].append("telegram_webhook")

    # Define edges (dependencies)
    # API → Providers
    graph["edges"].extend(
        [
            {"from": "GET_/api/quotes", "to": "provider_alphavantage", "type": "price_fetch"},
            {"from": "GET_/api/quotes", "to": "provider_polygon", "type": "price_fetch"},
            {"from": "GET_/api/quotes", "to": "provider_yahoo", "type": "price_fetch"},
            {
                "from": "GET_/api/crypto_price_{symbol}",
                "to": "provider_coingecko",
                "type": "price_fetch",
            },
            {
                "from": "GET_/api/crypto_price_{symbol}",
                "to": "provider_binance",
                "type": "price_fetch",
            },
        ]
    )

    # API → Storage
    graph["edges"].extend(
        [
            {"from": "GET_/api/quotes", "to": "redis_cache", "type": "cache_read"},
            {"from": "POST_/api/trade_submit", "to": "sqlite_wolf", "type": "db_write"},
            {"from": "GET_/api/risk_status", "to": "sqlite_risk", "type": "db_read"},
        ]
    )

    # API → AI
    graph["edges"].extend(
        [
            {"from": "POST_/telegram_webhook", "to": "openai", "type": "ai_chat"},
            {"from": "POST_/ai_chat", "to": "openai", "type": "ai_chat"},
        ]
    )

    # API → Broker
    graph["edges"].extend(
        [
            {"from": "POST_/api/trade_submit", "to": "alpaca_broker", "type": "order_submit"},
            {"from": "GET_/api/broker_positions", "to": "alpaca_broker", "type": "api_call"},
        ]
    )

    # Telegram → API
    graph["edges"].append(
        {"from": "telegram_webhook", "to": "POST_/telegram_webhook", "type": "webhook_call"}
    )

    return graph


def generate_svg(graph: dict, output_path: str):
    """Generate SVG visualization of dependency graph"""

    svg_header = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
<style>
  .node { fill: #4A90E2; stroke: #2C5AA0; stroke-width: 2; }
  .node-text { fill: white; font-family: Arial; font-size: 12px; font-weight: bold; }
  .edge { stroke: #999; stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }
  .layer-label { fill: #333; font-family: Arial; font-size: 14px; font-weight: bold; }
</style>
<defs>
  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#999" />
  </marker>
</defs>
"""

    svg_footer = "</svg>"

    # Layer positions (y-coordinate)
    layer_y = {
        "ui": 50,
        "api": 150,
        "orchestration": 300,
        "providers": 450,
        "storage": 600,
        "external": 750,
    }

    # Calculate node positions
    node_positions = {}
    for layer_name, nodes in graph["layers"].items():
        y = layer_y.get(layer_name, 400)
        x_step = 1200 / (len(nodes) + 1) if nodes else 200
        for i, node_id in enumerate(nodes):
            x = x_step * (i + 1)
            node_positions[node_id] = (x, y)

    svg_content = svg_header

    # Draw edges
    for edge in graph["edges"]:
        from_node = edge["from"]
        to_node = edge["to"]
        if from_node in node_positions and to_node in node_positions:
            x1, y1 = node_positions[from_node]
            x2, y2 = node_positions[to_node]
            svg_content += (
                f'  <line class="edge" x1="{x1}" y1="{y1 + 20}" x2="{x2}" y2="{y2 - 20}" />\n'
            )

    # Draw nodes
    for node in graph["nodes"]:
        node_id = node["id"]
        if node_id in node_positions:
            x, y = node_positions[node_id]
            label = node["label"]
            # Truncate long labels
            if len(label) > 20:
                label = label[:17] + "..."

            svg_content += f'  <rect class="node" x="{x - 50}" y="{y - 10}" width="100" height="40" rx="5" />\n'
            svg_content += f'  <text class="node-text" x="{x}" y="{y + 15}" text-anchor="middle">{label}</text>\n'

    # Draw layer labels
    for layer_name, y in layer_y.items():
        svg_content += (
            f'  <text class="layer-label" x="10" y="{y + 5}">{layer_name.upper()}</text>\n'
        )

    svg_content += svg_footer

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)


def main():
    """Main execution"""
    print("🗺️  Generating Ghost dependency map...")

    wolf_app_path = "/workspaces/GHOST/wolf_app.py"

    if not os.path.exists(wolf_app_path):
        print(f"❌ wolf_app.py not found at {wolf_app_path}")
        return

    # Extract dependencies
    graph = extract_dependencies(wolf_app_path)

    # Save JSON
    json_path = "/workspaces/GHOST/docs/dependency_graph.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"✅ Saved dependency graph: {json_path}")
    print(f"   - {len(graph['nodes'])} nodes")
    print(f"   - {len(graph['edges'])} edges")

    # Generate SVG
    svg_path = "/workspaces/GHOST/docs/dependency_graph.svg"
    generate_svg(graph, svg_path)
    print(f"✅ Saved SVG visualization: {svg_path}")

    # Print summary
    print("\n📊 Layer Summary:")
    for layer, nodes in graph["layers"].items():
        print(f"   {layer:15} {len(nodes):3} nodes")

    print("\n🔗 Critical Paths:")
    print("   1. Stock Prices: API → Providers → Cache → UI")
    print("   2. Trading: UI → API → Risk → Broker → DB")
    print("   3. Telegram: Webhook → API → AI → Response")
    print("   4. Crypto: API → CoinGecko → Cache → UI")

    print("\n✅ Dependency map generation complete!")


if __name__ == "__main__":
    main()
