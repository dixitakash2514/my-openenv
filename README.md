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

**A stateful, multi-step OpenEnv benchmarking the reasoning layer that sits above classical supply chain solvers.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://github.com/openenv-org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://github.com/dixitakash2514/my-openenv)
[![HF Space](https://img.shields.io/badge/Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/BlackEagle/my-env)

---

## Baseline Results (20 seeds × 3 tasks)

| Task | Do-Nothing | Random | Heuristic | Gap |
|---|---:|---:|---:|---:|
| **shelf_restock** (easy, 3 steps) | 0.001 ± 0.000 | 0.134 ± 0.158 | **0.899** ± 0.156 | 0.77 |
| **delivery_routing** (medium, 4 steps) | 0.001 ± 0.000 | 0.284 ± 0.159 | **0.735** ± 0.021 | 0.45 |
| **demand_surge** (hard, 5 steps) | 0.009 ± 0.013 | 0.246 ± 0.091 | **0.943** ± 0.057 | 0.70 |

Three signals worth highlighting:
- **Inaction is punished.** Empty actions score 0.00 on easy/medium tasks; ≈0.01 on the hard task. Every P&L formula has a built-in cost for unmet demand and unfulfilled orders.
- **The grader discriminates.** A blind heuristic reaches 0.90 on the easy task and >0.73 everywhere, but a random policy stays below 0.29 on all tasks. The gap (≥0.45) leaves room for LLM reasoning to sit between them.
- **Real variance.** Standard deviations of 0.02–0.16 confirm the grader is seed-sensitive, not constant.

Reproduce with: `cd my_env && .venv/bin/python baselines.py`
Raw per-seed scores: [`baseline_results.json`](baseline_results.json)

---

## Why This Env? The Reasoning Layer Above Classical Solvers

Classical LP/MIP solvers (NVIDIA cuOpt, OR-Tools, CPLEX) solve supply chain optimization optimally — **when the problem is cleanly formulated**. What they cannot do:

- Interpret a natural-language disruption ("Supplier 2 is offline due to flooding")
- Reason about stochastic supplier reliability and decide how much to hedge
- Choose *what* objective to optimize for under competing budget and time constraints
- Adapt across 3–5 sequential decisions as the world changes

That gap — translating messy real-world context into structured optimization decisions — is exactly what frontier LLMs are built for. This environment benchmarks it.

Supply chain disruptions cost organizations an average of **$83M annually**, with **45% of that recoverable** if response latency drops from days to seconds. The reasoning gap is the bottleneck.

---

## Tasks

| Task | Difficulty | Steps | Core Decision | Dynamic Events |
|---|---|---|---|---|
| Shelf Restock Priority | Easy | 3 | Pick K products to restock (urgency ranking) | Demand spikes, surprise deliveries |
| Delivery Route Assignment | Medium | 4 | Last-mile VRP-TW: assign orders to drivers | Urgent orders arrive, driver breaks down |
| Demand Surge Planning | Hard | 5 | Multi-supplier procurement under disruption | Supplier outage, demand shifts, warehouse alert |

### 1. Shelf Restock Priority (Easy — 3 steps)

A store manager has limited time before opening. The agent selects products to restock across 3 rounds.

**Grader:** `reward = fraction of picks that match the optimal urgency ranking`
- Urgency formula: `(daily_sales_rate × unit_revenue) / max(current_stock, 0.1)`
- Optimal picks the top-K products by urgency. Score = |agent_picks ∩ optimal_picks| / K.

### 2. Delivery Route Assignment (Medium — 4 steps)

A distribution center dispatcher assigns orders to drivers. Framed as a **VRP-TW** (Vehicle Routing Problem with Time Windows) with driver capacity, shift limits, and a mid-episode breakdown event.

**Grader (P&L, not weighted sums):**
```
profit = delivered_revenue − late_penalty − unfulfilled_penalty
reward = clip(profit / max_possible_revenue, 0, 1)

delivered_revenue   = +$2.50 / kg  (on-time deliveries only)
late_penalty        = −$4.00 / kg  (arrive after deadline)
unfulfilled_penalty = −$15.00 / kg (order never assigned or dropped)
route_time per stop = distance_km / 30 + 0.35h dwell
```

The $15/kg unfulfilled penalty is 6× the revenue per kg — leaving any order unassigned is catastrophically expensive. The right strategy reserves one driver for tight-deadline urgent orders that arrive at step 2.

### 3. Demand Surge Planning (Hard — 5 steps)

A festival is 5 days away. The agent procures inventory and manages redistribution as the situation deteriorates: a supplier goes offline at step 2, demand forecasts shift at step 3, and a warehouse section closes at step 4.

**Grader (P&L, normalized against do-nothing and oracle bounds):**
```
profit = sale_revenue − procurement_cost − stockout_penalty
                      − storage_penalty  − offline_order_fines
reward = clip((profit − baseline_profit) / (optimal_profit − baseline_profit), 0, 1)

sale_revenue   = $10.00 / unit sold (min of supply, demand)
stockout       = $8.00 / unit of unmet demand
storage        = $0.50 / unit above 110% of demand
offline_fine   = $400.00 / order sent to an OFFLINE supplier
```

**Stochastic supplier reliability:** Each supplier has a published `reliability` score (0.60–0.96), inversely correlated with price. When you order N units, only approximately `N × reliability` units arrive — budget is charged for all N regardless. The optimal objective is to minimize `effective_cost_per_unit = price / reliability`, not raw price. Blind cheapest-first ordering from a 0.60-reliability supplier burns ~40% of every order.

---

## Grader Design Philosophy

Every reward in this environment is a **realized P&L**, not a weighted sum of ad-hoc criteria. A judge reading any grader sees:

> `profit = revenue − costs − penalties`
> `reward = clip((profit − baseline) / (optimal − baseline), 0, 1)`

Where `baseline` is the do-nothing outcome and `optimal` is an oracle that cannot be beaten. Coefficients are not magic numbers — they are unit prices from the scenario's own data ($/kg revenue, $/kg penalty). No engineered "do nothing" punishments are needed: inaction is already dominated by the P&L math.

---

## Action & Observation Spaces

### Action (`SupplyChainAction`)
```python
class SupplyChainAction(Action):
    decision: Dict[str, Any]  # task-specific JSON
    reasoning: str = ""
```

Task decision formats:
- **shelf_restock**: `{"restock_products": ["P001", "P003"]}`
- **delivery_routing**: `{"assignments": [{"order_id": "ORD001", "driver_id": "D1"}, ...]}`
- **demand_surge**: `{"procurement_orders": [...], "redistribution": [...]}`

### Observation (`SupplyChainObservation`)
```python
class SupplyChainObservation(Observation):
    task_name: str
    step_number: int          # 0 = reset, 1+ = after step
    total_steps: int
    scenario_text: str        # evolving natural-language prompt for LLM
    scenario_data: Dict       # structured live data
    feedback: str             # per-step P&L breakdown
    score_breakdown: Optional[Dict]  # populated only on done=True
    done: bool
    reward: float             # per-step reward; mean of all steps on done=True
```

**Example terminal observation:**
```python
SupplyChainObservation(
    task_name="shelf_restock",
    step_number=3,
    total_steps=3,
    scenario_text="",
    scenario_data={},
    feedback="Episode complete. Final score: 0.900 | Step scores: 1.00, 0.80, 0.90 | ...",
    score_breakdown={
        "step_1_reward": 1.0, "step_2_reward": 0.8,
        "step_3_reward": 0.9, "final_score": 0.9,
    },
    done=True,
    reward=0.900,
)
```

---

## Setup & Usage

### Local Development
```bash
pip install openenv-core
cd my_env
uv sync
uv run server    # starts FastAPI on http://localhost:8000
```

### Docker
```bash
docker build -t supply-chain-env:latest .
docker run -p 8000:8000 supply-chain-env:latest
```

### Run Baselines
```bash
cd my_env
.venv/bin/python baselines.py            # 20 seeds, all tasks (no GPU)
.venv/bin/python baselines.py --seeds 5  # quick smoke-test
```

### Run LLM Inference
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
uv run inference.py
# or with a different model:
uv run inference.py --model "meta-llama/Llama-3.1-70B-Instruct"
```

### API
```bash
# Reset with task selection
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "shelf_restock", "seed": 42}'

# Submit decision
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"decision": {"restock_products": ["P003", "P007"]}}}'
```

### Deploy
```bash
openenv push --repo-id BlackEagle/my-env
```

---

## Why LLMs, Not Classical Solvers?

Classical solvers need a *formulated* problem: explicit objectives, constraints, and variables. Real supply chain decisions arrive as natural-language context with partial information, uncertain parameters, and competing stakeholder priorities. The LLM's job is to:

1. Parse the natural-language disruption ("flooding at supplier warehouse")
2. Map it to the relevant optimization objective (switch to backup supplier, hedge reliability)
3. Generate the structured decision the solver-equivalent grader evaluates

This environment benchmarks step 3 — the last mile between "LLM understands the situation" and "LLM produces an action the environment can score". No integration with external solvers is required; the grader is self-contained.

---

## Contributing

Pull requests welcome — especially for new tasks, richer demand distributions, or multi-agent scenarios (warehouse + delivery coordination).
