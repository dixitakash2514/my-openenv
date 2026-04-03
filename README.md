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

Supply chain decisions cost the retail industry $1.1 trillion annually in stockouts and overstock. This environment benchmarks whether AI agents can reason about inventory, logistics, and disruptions across multi-step episodes with dynamic events — not just answer static quiz questions.

Each task is a **multi-step episode** where the situation evolves between steps: demand spikes, suppliers go offline, vehicles break down, and warehouses hit capacity limits. The agent must adapt its strategy in response.

## Tasks

### 1. Shelf Restock Priority (Easy — 3 steps)
A store manager has limited time before opening. The agent selects products to restock across 3 rounds as the situation changes.

- **Step 1**: Pick 2 most urgent products from 10 (based on stockout risk and revenue)
- **Step 2**: Dynamic event — demand spike on one product + surprise delivery for another. Pick 1 more.
- **Step 3**: Pick 1 final product with fully updated data.

**Grading per step**: Selection quality (60%) + feasibility (40%)

### 2. Delivery Route Assignment (Medium — 4 steps)
A distribution center dispatcher assigns orders to drivers as new complications arise.

- **Step 1**: Assign 4 initial orders to 3 drivers
- **Step 2**: 2 new urgent express orders arrive + traffic delay on one route
- **Step 3**: A driver reports vehicle issues (capacity reduced 40%)
- **Step 4**: Final adjustments with all constraints active

**Grading per step**: On-time (30%) + capacity compliance (25%) + coverage (25%) + balance (20%)

### 3. Demand Surge Planning (Hard — 5 steps)
A festival approaches in 5 days. The agent procures inventory and redistributes stock as the situation deteriorates.

- **Step 1**: Initial procurement with all 4 suppliers active
- **Step 2**: One supplier goes OFFLINE — adjust procurement
- **Step 3**: Demand forecasts change (some categories up, some down)
- **Step 4**: Warehouse capacity alert — section closed for maintenance
- **Step 5**: Final review and adjustments

**Grading per step**: Fulfillment (30%) + budget (20%) + disruption handling (20%) + balance (15%) + waste prevention (15%)

**Final score** = mean of all per-step rewards.

## Action Space

```python
class SupplyChainAction(Action):
    decision: Dict[str, Any]  # Task-specific JSON decision
    reasoning: str = ""       # Agent's explanation
```

Task-specific decision formats:
- **shelf_restock**: `{"restock_products": ["P001", "P003"]}`
- **delivery_routing**: `{"assignments": [{"order_id": "ORD001", "driver_id": "D1"}, ...]}`
- **demand_surge**: `{"procurement_orders": [...], "redistribution": [...]}`

## Observation Space

```python
class SupplyChainObservation(Observation):
    task_name: str            # Current task identifier
    step_number: int          # Current step (0 = reset, 1+ = after step)
    total_steps: int          # Total steps in this task (3, 4, or 5)
    scenario_text: str        # Human-readable scenario for LLM (updated each step)
    scenario_data: Dict       # Structured data
    score_breakdown: Dict     # Per-step scores (final step only)
    feedback: str             # Brief feedback per step, detailed on final step
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

# Submit step 1 decision
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"decision": {"restock_products": ["P003", "P007"]}}}'

# Submit step 2 decision (scenario updates automatically)
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"decision": {"restock_products": ["P001"]}}}'
```

## Baseline Scores

Baseline agent (Qwen2.5-72B-Instruct, temperature=0.2):
- shelf_restock (3 steps): ~0.60-0.85
- delivery_routing (4 steps): ~0.40-0.65
- demand_surge (5 steps): ~0.25-0.55

## Deployment

```bash
openenv push --repo-id your-username/supply-chain-retail
```
