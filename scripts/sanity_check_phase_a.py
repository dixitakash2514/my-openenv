"""Sanity check for Phase A — verify the new P&L grader discriminates.

Runs do-nothing, random, and a simple heuristic on all three tasks across
10 seeds and prints the mean score per policy. Expected outcome:

    shelf_restock     : do-nothing ≈ 0.00, random ≈ 0.15, heuristic ≈ 0.95
    delivery_routing  : do-nothing ≈ 0.00, random ≈ 0.30, heuristic ≈ 0.75
    demand_surge      : do-nothing ≈ 0.00, random ≤ 0.25, heuristic ≥ 0.65

If random is too close to heuristic on any task, the grader still doesn't
discriminate — back to the drawing board.
"""

import random as _random
import sys
import os
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SupplyChainAction
from server.supply_chain_environment import SupplyChainEnvironment


SEEDS = [42, 123, 456, 789, 1000, 7, 31, 65, 200, 555,
         888, 1234, 1729, 2024, 9999, 11, 77, 314, 2718, 4242]
TASKS = ["shelf_restock", "delivery_routing", "demand_surge"]


# ---------- policies ----------


def do_nothing_action(obs, task):
    if task == "shelf_restock":
        return SupplyChainAction(decision={"restock_products": []})
    if task == "delivery_routing":
        return SupplyChainAction(decision={"assignments": []})
    return SupplyChainAction(
        decision={"procurement_orders": [], "redistribution": []}
    )


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
    suppliers = [
        s for s in data.get("suppliers", []) if s.get("status") == "ACTIVE"
    ]
    warehouses = data.get("warehouses", [])
    cats = ["rice", "cooking_oil", "flour", "sugar", "pulses", "spices"]
    orders = []
    for _ in range(rng.randint(1, 4)):
        if not suppliers or not warehouses:
            break
        sup = rng.choice(suppliers)
        wh = rng.choice(warehouses)
        orders.append(
            {
                "supplier_id": sup["supplier_id"],
                "product": rng.choice(cats),
                "quantity": rng.randint(20, 100),
                "destination_warehouse": wh["warehouse_id"],
            }
        )
    return SupplyChainAction(
        decision={"procurement_orders": orders, "redistribution": []}
    )


def heuristic_action(obs, task, step):
    """A competent closed-form policy that mirrors the grading formulas."""
    data = obs.scenario_data
    if task == "shelf_restock":
        products = data.get("products", [])
        restocked = set(data.get("restocked", []))
        slots_per_step = data.get("slots_per_step", [2, 1, 1])
        slots = slots_per_step[step] if step < len(slots_per_step) else 1
        ranked = sorted(
            (p for p in products if p["product_id"] not in restocked),
            key=lambda p: (
                (p["daily_sales_rate"] * p["unit_revenue"])
                / max(p["current_stock"], 0.1)
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

        # Best-Fit Decreasing by weight: place the heaviest order first on
        # the driver it fills most tightly (smallest remaining cap after
        # placement). This packs drivers compactly so one or more drivers
        # stay empty — leaving slack for the tight-deadline urgents that
        # arrive at step 2. A loose "first-fit on min-route-time" heuristic
        # spreads every order across drivers and leaves no empty driver,
        # which dooms step-2 urgents to "late" and collapses P&L.
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
                # Best-fit: prefer the driver with the LEAST remaining cap
                # (after hypothetical placement) so the next order has
                # more room on a different driver.
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
                assignments.append(
                    {"order_id": o["order_id"], "driver_id": drv["driver_id"]}
                )
        return SupplyChainAction(decision={"assignments": assignments})

    # demand_surge: buy the shortfall from the supplier with the best
    # expected cost-per-delivered-unit (price / reliability). Cheapest-by-
    # sticker-price loses money to failed deliveries.
    warehouses = data.get("warehouses", [])
    suppliers = [
        s for s in data.get("suppliers", []) if s.get("status") == "ACTIVE"
    ]
    if not suppliers or not warehouses:
        return SupplyChainAction(
            decision={"procurement_orders": [], "redistribution": []}
        )
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
            # Over-order to compensate for expected failure rate: to deliver
            # `shortfall` units in expectation from a reliability=r supplier,
            # order shortfall/r.
            target_units = shortfall[c] / reliability
            max_units = min(target_units, sup["max_order_qty"], budget // price)
            if max_units <= 0:
                continue
            wh = min(warehouses, key=lambda w: w["current_total"])
            qty = int(max_units)
            orders.append(
                {
                    "supplier_id": sup["supplier_id"],
                    "product": c,
                    "quantity": qty,
                    "destination_warehouse": wh["warehouse_id"],
                }
            )
            budget -= qty * price
            # Decrement shortfall by EXPECTED delivery (qty * reliability).
            shortfall[c] -= qty * reliability
            if shortfall[c] <= 0:
                break
        if budget <= 0:
            break
    return SupplyChainAction(
        decision={"procurement_orders": orders, "redistribution": []}
    )


# ---------- runner ----------


def run_episode(env, task, seed, policy_name):
    obs = env.reset(seed=seed, task_name=task)
    rng = _random.Random(seed + 9999)
    rewards = []
    step = 0
    while not obs.done:
        if policy_name == "do_nothing":
            action = do_nothing_action(obs, task)
        elif policy_name == "random":
            action = random_action(obs, task, rng)
        else:
            action = heuristic_action(obs, task, step)
        obs = env.step(action)
        rewards.append(obs.reward)
        step += 1
    return rewards[-1] if rewards else 0.0


def main():
    env = SupplyChainEnvironment()
    results = {}
    for task in TASKS:
        results[task] = {}
        for policy in ("do_nothing", "random", "heuristic"):
            scores = [run_episode(env, task, s, policy) for s in SEEDS]
            results[task][policy] = (mean(scores), stdev(scores) if len(scores) > 1 else 0.0, scores)

    print("\n" + "=" * 70)
    print("Phase A sanity check — unified P&L grader")
    print("=" * 70)
    print(f"{'Task':<20} {'do_nothing':>14} {'random':>14} {'heuristic':>14}")
    print("-" * 70)
    for task in TASKS:
        rows = []
        for p in ("do_nothing", "random", "heuristic"):
            m, sd, _ = results[task][p]
            rows.append(f"{m:.3f} ± {sd:.3f}")
        print(f"{task:<20} {rows[0]:>14} {rows[1]:>14} {rows[2]:>14}")

    print("\nAcceptance criteria:")
    oks = []
    for task in TASKS:
        dn = results[task]["do_nothing"][0]
        rn = results[task]["random"][0]
        hu = results[task]["heuristic"][0]
        gap = hu - rn
        ok = dn <= 0.15 and rn <= 0.30 and hu >= 0.65 and gap >= 0.35
        oks.append(ok)
        mark = "PASS" if ok else "FAIL"
        print(
            f"  [{mark}] {task}: do_nothing={dn:.2f} random={rn:.2f} "
            f"heuristic={hu:.2f} gap={gap:.2f}"
        )

    if all(oks):
        print("\n>>> Phase A grader passes all targets. Ready for Phase B.")
    else:
        print("\n>>> Phase A grader needs more tuning. See failures above.")


if __name__ == "__main__":
    main()
