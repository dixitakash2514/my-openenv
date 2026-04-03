---
title: Supply Chain Retail Environment
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Supply Chain Retail Environment

Supply chain decisions cost the retail industry $1.1 trillion annually in stockouts and overstock. Yet there's no standardized benchmark for evaluating whether AI agents can reason about inventory, logistics, and disruptions. This environment fills that gap — 3 tasks that test the core reasoning capabilities needed for supply chain AI, from simple prioritization to multi-node planning under uncertainty.

## Tasks

### 1. Shelf Restock Priority (Easy)
A store manager has limited time before opening and can only restock 4 out of 10 products. The agent must prioritize based on stockout risk (current stock vs daily sales rate) and revenue impact.

**Action**: Select 4 product IDs ordered by urgency
**Grading**: Selection quality (50%), priority order (30%), feasibility (20%)

### 2. Delivery Route Assignment (Medium)
A distribution center dispatcher must assign 6 delivery orders to 3 drivers, balancing vehicle capacity, delivery deadlines, shift hours, and workload distribution.

**Action**: Assign each order to a driver
**Grading**: On-time delivery (35%), capacity compliance (25%), efficiency (25%), driver balance (15%)

### 3. Demand Surge Planning with Disruption (Hard)
A festival approaches in 5 days. The agent must plan procurement across 4 suppliers (one is OFFLINE) and redistribute inventory across 3 warehouses, staying within budget while maximizing demand fulfillment.

**Action**: Procurement orders + inventory redistribution plan
**Grading**: Demand fulfillment (30%), budget compliance (20%), disruption handling (20%), inventory balance (15%), waste prevention (15%)

## Action Space

```python
class SupplyChainAction(Action):
    decision: Dict[str, Any]  # Task-specific JSON decision
    reasoning: str = ""       # Agent's explanation
```

Task-specific decision formats:
- **shelf_restock**: `{"restock_products": ["P001", "P003", "P007", "P005"]}`
- **delivery_routing**: `{"assignments": [{"order_id": "ORD001", "driver_id": "D1"}, ...]}`
- **demand_surge**: `{"procurement_orders": [...], "redistribution": [...]}`

## Observation Space

```python
class SupplyChainObservation(Observation):
    task_name: str            # Current task identifier
    scenario_text: str        # Human-readable scenario for LLM
    scenario_data: Dict       # Structured data
    score_breakdown: Dict     # Per-criterion scores (after grading)
    feedback: str             # Grader feedback (after grading)
```

## Setup

```bash
pip install openenv-core
cd my_env
uv sync
```

## Usage

```bash
# Run server locally
uv run server

# Run inference
HF_TOKEN=your_token uv run inference.py

# Docker
docker build -t my_env-env:latest .
docker run -p 8000:8000 my_env-env:latest
```

## API

```bash
# Reset with task selection
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "shelf_restock", "seed": 42}'

# Submit decision
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"decision": {"restock_products": ["P003", "P007", "P001", "P005"]}}}'
```

## Baseline Scores

Baseline agent (Qwen2.5-72B-Instruct, temperature=0.2):
- shelf_restock: ~0.70-0.90
- delivery_routing: ~0.50-0.70
- demand_surge: ~0.30-0.60

## Deployment

```bash
openenv push --repo-id your-username/supply-chain-retail
```
