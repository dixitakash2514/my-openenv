"""
Heuristic Baseline Eval — Supply Chain Retail
=============================================

A model-free, deterministic baseline that demonstrates the environment is
solvable and produces meaningful score variance across seeds.

This script:
  - Imports the env directly (no Docker, no LLM, no HTTP).
  - Runs three baselines per task across 15 seeds:
      1. random   — picks a random valid action each step.
      2. do_nothing — submits an empty action (sanity check that the
                      patched graders no longer reward inaction).
      3. heuristic — applies a simple, hand-coded strategy that uses
                     the same scoring formulas the env grades against.
  - Dumps a JSON results file at training/results/heuristic_scores.json.

Usage:
    cd my_env && python -m training.eval_heuristic
    # OR from inside training/ with the venv:
    cd training && ../.venv/bin/python eval_heuristic.py
"""

import json
import os
import random
import statistics
import sys
from typing import Any, Dict, List, Tuple

# Make `my_env` importable when invoked from inside training/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT)

from server.supply_chain_environment import SupplyChainEnvironment  # noqa: E402
from models import SupplyChainAction  # noqa: E402

TASKS = ["shelf_restock", "delivery_routing", "demand_surge"]
SEEDS = [42, 123, 456, 789, 1000, 7, 31, 65, 200, 555, 888, 1234, 1729, 2024, 9999]


# ---------------------------------------------------------------------------
# Heuristic policies — one per task. These mirror the grading formulas.
# ---------------------------------------------------------------------------


def heuristic_shelf_restock(observation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the highest-urgency products. Urgency = sales_rate * revenue / stock."""
    products = observation_data.get("products", [])
    restocked = set(observation_data.get("restocked", []))
    slots_per_step = observation_data.get("slots_per_step", [2, 1, 1])
    # Determine current step's slot count from how many we've already restocked.
    already = len(restocked)
    cumulative = 0
    slots = 1
    for s in slots_per_step:
        cumulative += s
        if already < cumulative:
            slots = s
            break

    candidates = []
    for p in products:
        if p["product_id"] in restocked:
            continue
        stock = max(p["current_stock"], 0.1)
        urgency = (p["daily_sales_rate"] * p["unit_revenue"]) / stock
        candidates.append((urgency, p["product_id"]))
    candidates.sort(reverse=True)
    return {"restock_products": [pid for _, pid in candidates[:slots]]}


def heuristic_delivery_routing(observation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assign each pending order to the driver with the most remaining capacity."""
    orders = observation_data.get("orders", [])
    drivers = observation_data.get("drivers", [])

    # Snapshot remaining capacity per driver (we update locally as we plan).
    remaining = {
        d["driver_id"]: d["vehicle_capacity_kg"] - d["used_capacity_kg"]
        for d in drivers
    }

    pending = [o for o in orders if o.get("status") == "pending"]
    # Sort by tightest deadline first.
    pending.sort(key=lambda o: o.get("deadline_hours", 99))

    assignments = []
    for order in pending:
        weight = order.get("weight_kg", 0)
        # Pick driver with most remaining capacity that can fit this order.
        best_driver = None
        best_cap = -1
        for did, cap in remaining.items():
            if cap >= weight and cap > best_cap:
                best_cap = cap
                best_driver = did
        if best_driver is not None:
            assignments.append({"order_id": order["order_id"], "driver_id": best_driver})
            remaining[best_driver] -= weight
    return {"assignments": assignments}


def heuristic_demand_surge(observation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Order from active suppliers to close the demand gap, cheapest supplier first."""
    suppliers = observation_data.get("suppliers", [])
    warehouses = observation_data.get("warehouses", [])
    total_demand = observation_data.get("total_demand", {})
    budget_remaining = observation_data.get("budget_remaining", 0)

    # Compute total available per category across all warehouses.
    total_avail: Dict[str, int] = {}
    for wh in warehouses:
        for cat, qty in wh.get("inventory", {}).items():
            total_avail[cat] = total_avail.get(cat, 0) + qty

    # Build per-category gap (need - have).
    gaps = {}
    for cat, demand in total_demand.items():
        gap = max(0, demand - total_avail.get(cat, 0))
        if gap > 0:
            gaps[cat] = gap

    # Active suppliers sorted by price (cheapest first).
    active = [s for s in suppliers if s.get("status") != "OFFLINE"]
    active.sort(key=lambda s: s.get("price_per_unit", 1e9))

    # Pick the warehouse with the most free capacity as the destination.
    if warehouses:
        free_cap = sorted(
            warehouses,
            key=lambda w: (w.get("max_capacity", 0) - w.get("current_total", 0)),
            reverse=True,
        )
        dest = free_cap[0]["warehouse_id"]
    else:
        dest = "WH1"

    # Order ~60% of the remaining gap per step. Spreads procurement across steps
    # so the agent isn't punished for overstock when later steps reveal more demand
    # or supplier disruptions.
    procurement = []
    remaining_budget = budget_remaining
    for cat, gap in gaps.items():
        target = max(1, int(gap * 0.6))
        for sup in active:
            price = sup.get("price_per_unit", 1)
            max_qty = min(target, sup.get("max_order_qty", target))
            affordable = int(remaining_budget // price) if price > 0 else max_qty
            qty = min(max_qty, affordable)
            if qty <= 0:
                continue
            procurement.append({
                "supplier_id": sup["supplier_id"],
                "product": cat,
                "quantity": qty,
                "destination_warehouse": dest,
            })
            remaining_budget -= qty * price
            target -= qty
            if target <= 0:
                break

    return {"procurement_orders": procurement, "redistribution": []}


HEURISTICS = {
    "shelf_restock": heuristic_shelf_restock,
    "delivery_routing": heuristic_delivery_routing,
    "demand_surge": heuristic_demand_surge,
}


# ---------------------------------------------------------------------------
# Random and do-nothing policies (sanity checks)
# ---------------------------------------------------------------------------


def random_shelf_restock(obs: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    products = [p["product_id"] for p in obs.get("products", []) if p["product_id"] not in obs.get("restocked", [])]
    rng.shuffle(products)
    return {"restock_products": products[:2]}


def random_delivery_routing(obs: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    pending = [o["order_id"] for o in obs.get("orders", []) if o.get("status") == "pending"]
    drivers = [d["driver_id"] for d in obs.get("drivers", [])]
    if not drivers:
        return {"assignments": []}
    return {"assignments": [{"order_id": oid, "driver_id": rng.choice(drivers)} for oid in pending]}


def random_demand_surge(obs: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    suppliers = [s["supplier_id"] for s in obs.get("suppliers", []) if s.get("status") != "OFFLINE"]
    warehouses = [w["warehouse_id"] for w in obs.get("warehouses", [])]
    cats = list(obs.get("total_demand", {}).keys())
    if not (suppliers and warehouses and cats):
        return {"procurement_orders": [], "redistribution": []}
    return {
        "procurement_orders": [
            {
                "supplier_id": rng.choice(suppliers),
                "product": rng.choice(cats),
                "quantity": rng.randint(20, 80),
                "destination_warehouse": rng.choice(warehouses),
            }
            for _ in range(3)
        ],
        "redistribution": [],
    }


RANDOM_POLICIES = {
    "shelf_restock": random_shelf_restock,
    "delivery_routing": random_delivery_routing,
    "demand_surge": random_demand_surge,
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(env: SupplyChainEnvironment, task: str, seed: int, policy: str) -> Tuple[float, List[float]]:
    """Run one full episode under a given policy and return (final_reward, per_step_rewards)."""
    obs = env.reset(task_name=task, seed=seed)
    done = obs.done
    rng = random.Random(seed)
    step_rewards: List[float] = []
    final_reward = 0.0

    while not done:
        scenario_data = obs.scenario_data
        if policy == "do_nothing":
            decision: Dict[str, Any] = {}
        elif policy == "random":
            decision = RANDOM_POLICIES[task](scenario_data, rng)
        elif policy == "heuristic":
            decision = HEURISTICS[task](scenario_data)
        else:
            decision = {}

        result = env.step(SupplyChainAction(decision=decision))
        step_rewards.append(float(result.reward or 0.0))
        final_reward = float(result.reward or 0.0)
        done = bool(result.done)
        obs = result

    return final_reward, step_rewards


def evaluate(policy: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for task in TASKS:
        scores: List[float] = []
        for seed in SEEDS:
            env = SupplyChainEnvironment()
            final_reward, _ = run_episode(env, task, seed, policy)
            scores.append(round(final_reward, 4))
        avg = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        out[task] = {
            "scores": scores,
            "avg": round(avg, 4),
            "std": round(std, 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
    return out


def main() -> None:
    print(f"Evaluating across {len(SEEDS)} seeds × {len(TASKS)} tasks per policy.\n")
    results: Dict[str, Any] = {}
    for policy in ["do_nothing", "random", "heuristic"]:
        print(f"--- Policy: {policy} ---")
        results[policy] = evaluate(policy)
        for task in TASKS:
            stats = results[policy][task]
            print(f"  {task:<20} avg={stats['avg']:.3f}  std={stats['std']:.3f}  min={stats['min']:.3f}  max={stats['max']:.3f}")
        print()

    out_dir = os.path.join(THIS_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "heuristic_scores.json")
    with open(out_path, "w") as f:
        json.dump({"seeds": SEEDS, "results": results}, f, indent=2)
    print(f"Saved → {out_path}")

    # Also dump a compact markdown table for easy README copy.
    md_path = os.path.join(out_dir, "heuristic_table.md")
    with open(md_path, "w") as f:
        f.write("| Task | Do-Nothing | Random | Heuristic |\n")
        f.write("|---|---:|---:|---:|\n")
        for task in TASKS:
            row = (
                f"| {task} "
                f"| {results['do_nothing'][task]['avg']:.3f} ± {results['do_nothing'][task]['std']:.3f} "
                f"| {results['random'][task]['avg']:.3f} ± {results['random'][task]['std']:.3f} "
                f"| {results['heuristic'][task]['avg']:.3f} ± {results['heuristic'][task]['std']:.3f} |\n"
            )
            f.write(row)
    print(f"Markdown table → {md_path}")


if __name__ == "__main__":
    main()
