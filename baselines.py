"""
baselines.py — Multi-seed baseline runner for the Supply Chain Retail Environment.

Evaluates three reference policies (do-nothing, random, heuristic) across all
three tasks and writes results to baseline_results.json.  No GPU, no training
stack, no TRL dependency — only the env itself and stdlib.

Usage:
    cd my_env
    .venv/bin/python baselines.py            # 20 seeds, all tasks
    .venv/bin/python baselines.py --seeds 5  # quick smoke-test
"""

import argparse
import json
import math
import os
import random as _random
import sys
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SupplyChainAction
from server.supply_chain_environment import SupplyChainEnvironment


TASKS = ["shelf_restock", "delivery_routing", "demand_surge"]
DEFAULT_SEEDS = [
    42, 123, 456, 789, 1000,
    7, 31, 65, 200, 555,
    888, 1234, 1729, 2024, 9999,
    11, 77, 314, 2718, 4242,
]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def do_nothing_action(obs, task):
    if task == "shelf_restock":
        return SupplyChainAction(decision={"restock_products": []})
    if task == "delivery_routing":
        return SupplyChainAction(decision={"assignments": []})
    return SupplyChainAction(decision={"procurement_orders": [], "redistribution": []})


def random_action(obs, task, rng):
    data = obs.scenario_data
    if task == "shelf_restock":
        ids = [p["product_id"] for p in data.get("products", [])]
        restocked = set(data.get("restocked", []))
        available = [p for p in ids if p not in restocked]
        picks = rng.sample(available, min(2, len(available)))
        return SupplyChainAction(decision={"restock_products": picks})

    if task == "delivery_routing":
        pending = [
            o["order_id"] for o in data.get("orders", []) if o["status"] == "pending"
        ]
        driver_ids = [d["driver_id"] for d in data.get("drivers", [])]
        if not driver_ids:
            return SupplyChainAction(decision={"assignments": []})
        assignments = [
            {"order_id": oid, "driver_id": rng.choice(driver_ids)} for oid in pending
        ]
        return SupplyChainAction(decision={"assignments": assignments})

    # demand_surge
    suppliers = [s for s in data.get("suppliers", []) if s.get("status") == "ACTIVE"]
    warehouses = data.get("warehouses", [])
    cats = ["rice", "cooking_oil", "flour", "sugar", "pulses", "spices"]
    orders = []
    for _ in range(rng.randint(1, 4)):
        if not suppliers or not warehouses:
            break
        sup = rng.choice(suppliers)
        wh = rng.choice(warehouses)
        orders.append({
            "supplier_id": sup["supplier_id"],
            "product": rng.choice(cats),
            "quantity": rng.randint(20, 100),
            "destination_warehouse": wh["warehouse_id"],
        })
    return SupplyChainAction(decision={"procurement_orders": orders, "redistribution": []})


def heuristic_action(obs, task):
    """Competent closed-form policy mirroring the grading formulas."""
    data = obs.scenario_data

    if task == "shelf_restock":
        products = data.get("products", [])
        restocked = set(data.get("restocked", []))
        slots_per_step = data.get("slots_per_step", [2, 1, 1])
        step = obs.step_number  # 0 on reset, 1 after first step, etc.
        slots = slots_per_step[step] if step < len(slots_per_step) else 1
        ranked = sorted(
            (p for p in products if p["product_id"] not in restocked),
            key=lambda p: (
                (p["daily_sales_rate"] * p["unit_revenue"]) / max(p["current_stock"], 0.1)
            ),
            reverse=True,
        )
        picks = [p["product_id"] for p in ranked[:slots]]
        return SupplyChainAction(decision={"restock_products": picks})

    if task == "delivery_routing":
        all_orders = data.get("orders", [])
        order_lookup = {o["order_id"]: o for o in all_orders}
        pending = [o for o in all_orders if o["status"] == "pending"]
        drivers = data.get("drivers", [])
        distances = data.get("distances_from_dc", {})
        speed = data.get("speed_kmh", 30)
        dwell = data.get("dwell_hours_per_stop", 0.35)

        def committed_route_time(d):
            t = 0.0
            for oid in d.get("assigned_orders", []):
                o = order_lookup.get(oid)
                if not o:
                    continue
                t += distances.get(o["destination"], 20.0) / speed
                t += dwell
            return t

        drv_state = [
            {
                "driver_id": d["driver_id"],
                "cap_left": d["vehicle_capacity_kg"] - d["used_capacity_kg"],
                "route_time": committed_route_time(d),
                "shift": d.get("remaining_shift_hours", 99.0),
            }
            for d in drivers
        ]

        # Best-Fit Decreasing: pack heaviest orders first into the most-filled
        # drivers, leaving one driver empty as a reserve for tight-deadline
        # urgent orders that arrive at step 2.
        pending.sort(key=lambda o: (-o["weight_kg"], o["deadline_hours"]))
        assignments = []
        for o in pending:
            on_time_best = None
            any_best = None
            for d in drv_state:
                if d["cap_left"] < o["weight_kg"]:
                    continue
                leg = distances.get(o["destination"], 20.0) / speed
                arrive = d["route_time"] + leg
                completed = arrive + dwell
                if completed > d["shift"]:
                    continue
                projected_remaining = d["cap_left"] - o["weight_kg"]
                cand = (d, leg, arrive, completed, projected_remaining)
                if arrive <= o["deadline_hours"]:
                    if on_time_best is None or projected_remaining < on_time_best[4]:
                        on_time_best = cand
                else:
                    if any_best is None or projected_remaining < any_best[4]:
                        any_best = cand
            best = on_time_best or any_best
            if best is not None:
                drv, leg, arrive, completed, _ = best
                drv["cap_left"] -= o["weight_kg"]
                drv["route_time"] = completed
                assignments.append({"order_id": o["order_id"], "driver_id": drv["driver_id"]})
        return SupplyChainAction(decision={"assignments": assignments})

    # demand_surge: buy from supplier with best effective cost (price/reliability)
    warehouses = data.get("warehouses", [])
    suppliers = [s for s in data.get("suppliers", []) if s.get("status") == "ACTIVE"]
    if not suppliers or not warehouses:
        return SupplyChainAction(decision={"procurement_orders": [], "redistribution": []})
    total_demand = data.get("total_demand", {})
    cats = list(total_demand.keys())
    supply = {c: sum(w["inventory"].get(c, 0) for w in warehouses) for c in cats}
    shortfall = {c: max(0, total_demand[c] - supply[c]) for c in cats}
    suppliers.sort(
        key=lambda s: s["price_per_unit"] / max(s.get("reliability", 1.0), 0.01)
    )
    budget = data.get("budget_remaining", data.get("budget", 0))
    orders = []
    for c in cats:
        if shortfall[c] <= 0:
            continue
        for sup in suppliers:
            if budget <= 0:
                break
            price = sup["price_per_unit"]
            reliability = max(sup.get("reliability", 1.0), 0.01)
            target_units = shortfall[c] / reliability
            max_units = min(target_units, sup["max_order_qty"], budget // price)
            if max_units <= 0:
                continue
            wh = min(warehouses, key=lambda w: w["current_total"])
            qty = int(max_units)
            orders.append({
                "supplier_id": sup["supplier_id"],
                "product": c,
                "quantity": qty,
                "destination_warehouse": wh["warehouse_id"],
            })
            budget -= qty * price
            shortfall[c] -= qty * reliability
            if shortfall[c] <= 0:
                break
        if budget <= 0:
            break
    return SupplyChainAction(decision={"procurement_orders": orders, "redistribution": []})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_episode(env, task, seed, policy):
    obs = env.reset(seed=seed, task_name=task)
    rng = _random.Random(seed + 9999)
    while not obs.done:
        if policy == "do_nothing":
            action = do_nothing_action(obs, task)
        elif policy == "random":
            action = random_action(obs, task, rng)
        else:
            action = heuristic_action(obs, task)
        obs = env.step(action)
    return float(obs.reward)


def run_baselines(seeds):
    env = SupplyChainEnvironment()
    results = {}
    for task in TASKS:
        results[task] = {}
        for policy in ("do_nothing", "random", "heuristic"):
            scores = [run_episode(env, task, s, policy) for s in seeds]
            results[task][policy] = {
                "scores": [round(s, 4) for s in scores],
                "avg": round(mean(scores), 4),
                "std": round(stdev(scores) if len(scores) > 1 else 0.0, 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }
    return results


def print_table(results, seeds):
    print(f"\n{'='*72}")
    print("Supply Chain Retail — Baseline Scores")
    print(f"Seeds: {len(seeds)}  Tasks: {', '.join(TASKS)}")
    print(f"{'='*72}")
    print(f"{'Task':<20} {'do_nothing':>16} {'random':>16} {'heuristic':>16}")
    print(f"{'-'*72}")
    for task in TASKS:
        row = []
        for p in ("do_nothing", "random", "heuristic"):
            r = results[task][p]
            row.append(f"{r['avg']:.3f} ± {r['std']:.3f}")
        print(f"{task:<20} {row[0]:>16} {row[1]:>16} {row[2]:>16}")

    print("\nAcceptance criteria (do_nothing≤0.15, random≤0.30, heuristic≥0.65, gap≥0.35):")
    all_pass = True
    for task in TASKS:
        dn = results[task]["do_nothing"]["avg"]
        rn = results[task]["random"]["avg"]
        hu = results[task]["heuristic"]["avg"]
        gap = hu - rn
        ok = dn <= 0.15 and rn <= 0.30 and hu >= 0.65 and gap >= 0.35
        all_pass = all_pass and ok
        mark = "PASS" if ok else "FAIL"
        print(
            f"  [{mark}] {task}: do_nothing={dn:.2f}  random={rn:.2f}  "
            f"heuristic={hu:.2f}  gap={gap:.2f}"
        )
    if all_pass:
        print("\n>>> All targets met. Grader discriminates policies.")
    else:
        print("\n>>> Some targets failed — see above.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to run")
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS[: args.seeds]
    print(f"Running baselines on {len(seeds)} seeds...", flush=True)

    results = run_baselines(seeds)
    print_table(results, seeds)

    out = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "results": results,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
