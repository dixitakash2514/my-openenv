# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Supply Chain Retail Environment — Multi-Step with Dynamic Events.

Tasks:
  shelf_restock    (easy,   3 steps) — prioritize restocking with evolving demand
  delivery_routing (medium, 4 steps) — assign deliveries with live disruptions
  demand_surge     (hard,   5 steps) — plan procurement as situation evolves
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import random as random_module

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupplyChainAction, SupplyChainObservation
except (ImportError, SystemError):
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import SupplyChainAction, SupplyChainObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRODUCT_CATALOG = [
    ("P001", "Whole Milk 1L", 4.50),
    ("P002", "Organic Eggs 12pk", 6.20),
    ("P003", "Sourdough Bread", 5.80),
    ("P004", "Chicken Breast 500g", 8.90),
    ("P005", "Basmati Rice 2kg", 7.50),
    ("P006", "Olive Oil 500ml", 9.30),
    ("P007", "Greek Yogurt 500g", 3.80),
    ("P008", "Fresh Spinach 250g", 3.20),
    ("P009", "Cheddar Cheese 200g", 4.60),
    ("P010", "Orange Juice 1L", 4.10),
    ("P011", "Butter 250g", 3.90),
    ("P012", "Pasta 500g", 2.80),
]

STORE_LOCATIONS = {
    "DC": (0, 0),
    "Metro Market": (5, 12),
    "Green Grocer": (15, 3),
    "FreshMart": (8, 18),
    "QuickStop": (20, 10),
    "ValueStore": (3, 22),
    "PrimeMart": (18, 15),
}

SUPPLIER_NAMES = ["AgriCorp", "FreshSource", "BulkTrade", "QuickSupply"]
PRODUCT_CATEGORIES = ["rice", "cooking_oil", "flour", "sugar", "pulses", "spices"]

TASK_STEPS = {"shelf_restock": 3, "delivery_routing": 4, "demand_surge": 5}
VALID_TASKS = set(TASK_STEPS.keys())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SupplyChainEnvironment(Environment):
    """
    Supply Chain Retail environment with 3 multi-step tasks.

    Each task has dynamic events that change the scenario mid-episode,
    requiring the agent to adapt its strategy.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name: str = ""
        self._rng: random_module.Random = random_module.Random(42)
        self._step_num: int = 0
        self._max_steps: int = 1
        self._step_rewards: List[float] = []
        # Task-specific mutable state
        self._env_state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "shelf_restock",
        **kwargs,
    ) -> SupplyChainObservation:
        if task_name not in VALID_TASKS:
            task_name = "shelf_restock"

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._task_name = task_name
        self._rng = random_module.Random(seed if seed is not None else 42)
        self._step_num = 0
        self._max_steps = TASK_STEPS[task_name]
        self._step_rewards = []

        generators = {
            "shelf_restock": self._init_shelf_restock,
            "delivery_routing": self._init_delivery_routing,
            "demand_surge": self._init_demand_surge,
        }
        generators[task_name]()
        scenario_text = self._build_scenario_text()

        return SupplyChainObservation(
            task_name=task_name,
            step_number=0,
            total_steps=self._max_steps,
            scenario_text=scenario_text,
            scenario_data=self._get_public_state(),
            done=False,
            reward=0.0,
        )

    def step(self, action: SupplyChainAction, **kwargs) -> SupplyChainObservation:
        if not self._task_name:
            self.reset(task_name="shelf_restock", seed=42)

        self._step_num += 1
        self._state.step_count = self._step_num

        step_handlers = {
            "shelf_restock": self._step_shelf_restock,
            "delivery_routing": self._step_delivery_routing,
            "demand_surge": self._step_demand_surge,
        }

        reward, feedback = step_handlers[self._task_name](action.decision)
        reward = max(0.0, min(1.0, reward))
        self._step_rewards.append(reward)

        is_done = self._step_num >= self._max_steps

        # Build next observation
        if is_done:
            final_score = sum(self._step_rewards) / len(self._step_rewards)
            breakdown = self._get_final_breakdown()
            final_feedback = (
                f"Episode complete. Final score: {final_score:.3f} | "
                f"Step scores: {', '.join(f'{r:.2f}' for r in self._step_rewards)} | "
                f"{feedback}"
            )
            return SupplyChainObservation(
                task_name=self._task_name,
                step_number=self._step_num,
                total_steps=self._max_steps,
                scenario_text="",
                scenario_data={},
                score_breakdown=breakdown,
                feedback=final_feedback,
                done=True,
                reward=final_score,
            )
        else:
            scenario_text = self._build_scenario_text()
            return SupplyChainObservation(
                task_name=self._task_name,
                step_number=self._step_num,
                total_steps=self._max_steps,
                scenario_text=scenario_text,
                scenario_data=self._get_public_state(),
                feedback=f"Step {self._step_num} reward: {reward:.2f} | {feedback}",
                done=False,
                reward=reward,
            )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_public_state(self) -> Dict[str, Any]:
        """Return the public portion of state (no internal grading data)."""
        return {
            k: v
            for k, v in self._env_state.items()
            if not k.startswith("_")
        }

    def _get_final_breakdown(self) -> Dict[str, float]:
        """Per-step reward breakdown."""
        breakdown = {}
        for i, r in enumerate(self._step_rewards):
            breakdown[f"step_{i+1}_reward"] = round(r, 3)
        breakdown["final_score"] = round(
            sum(self._step_rewards) / len(self._step_rewards), 3
        )
        return breakdown

    # ==================================================================
    # TASK 1: Shelf Restock (3 steps)
    #
    # Step 1: Pick 2 most urgent products to restock
    # Step 2: Dynamic event (demand spike + surprise delivery). Pick 1 more.
    # Step 3: Final pick (1 more). Episode done.
    # ==================================================================

    def _init_shelf_restock(self):
        rng = self._rng
        products = []
        selected = rng.sample(PRODUCT_CATALOG, 10)
        for pid, name, base_rev in selected:
            daily_sales = round(rng.uniform(5, 50), 1)
            capacity = rng.randint(40, 100)
            stock = rng.randint(0, int(capacity * 0.4))
            revenue = round(base_rev * rng.uniform(0.8, 1.2), 2)
            products.append({
                "product_id": pid, "name": name,
                "current_stock": stock, "daily_sales_rate": daily_sales,
                "shelf_capacity": capacity, "unit_revenue": revenue,
            })

        self._env_state = {
            "products": products,
            "restocked": [],  # IDs already restocked
            "slots_per_step": [2, 1, 1],  # 4 total
            "_events_applied": [],
        }

    def _step_shelf_restock(self, decision: Dict) -> Tuple[float, str]:
        s = self._env_state
        products = s["products"]
        step = self._step_num  # 1, 2, or 3
        slots = s["slots_per_step"][step - 1]

        # --- Apply dynamic event before grading step 2+ ---
        if step == 2:
            self._apply_shelf_event()

        # --- Grade the decision ---
        picks = decision.get("restock_products", [])
        if not isinstance(picks, list):
            picks = []

        valid_ids = {p["product_id"] for p in products} - set(s["restocked"])
        valid_picks = [p for p in picks if p in valid_ids][:slots]

        # Compute urgency ranking on current data (excluding already restocked)
        urgency = []
        for p in products:
            if p["product_id"] in s["restocked"]:
                continue
            stock = max(p["current_stock"], 0.1)
            u = (p["daily_sales_rate"] * p["unit_revenue"]) / stock
            urgency.append((p["product_id"], u))
        urgency.sort(key=lambda x: x[1], reverse=True)
        optimal = [u[0] for u in urgency[:slots]]

        # Selection score
        correct = set(valid_picks) & set(optimal)
        selection_score = len(correct) / slots if slots > 0 else 0.0

        # Feasibility
        if len(valid_picks) == slots:
            feasibility = 1.0
        elif len(valid_picks) > 0:
            feasibility = 0.5
        else:
            feasibility = 0.0

        reward = 0.6 * selection_score + 0.4 * feasibility

        # Apply restock: update stock for picked products
        for pid in valid_picks:
            for p in products:
                if p["product_id"] == pid:
                    p["current_stock"] = p["shelf_capacity"]
                    break
            s["restocked"].append(pid)

        feedback = (
            f"Picked: {valid_picks} | Optimal: {optimal} | "
            f"Correct: {len(correct)}/{slots}"
        )
        return reward, feedback

    def _apply_shelf_event(self):
        """Dynamic event before step 2: demand spike + surprise delivery."""
        rng = self._rng
        products = self._env_state["products"]
        available = [
            p for p in products
            if p["product_id"] not in self._env_state["restocked"]
        ]
        if len(available) >= 2:
            # One product's sales spike
            spike_product = rng.choice(available)
            old_rate = spike_product["daily_sales_rate"]
            spike_product["daily_sales_rate"] = round(old_rate * 1.8, 1)

            # Another product gets a surprise delivery
            others = [p for p in available if p != spike_product]
            delivery_product = rng.choice(others)
            bonus = rng.randint(15, 30)
            delivery_product["current_stock"] = min(
                delivery_product["current_stock"] + bonus,
                delivery_product["shelf_capacity"],
            )

            self._env_state["_events_applied"].append(
                f"DEMAND SPIKE: {spike_product['name']} sales rate jumped from "
                f"{old_rate} to {spike_product['daily_sales_rate']}/day. "
                f"DELIVERY: {delivery_product['name']} received +{bonus} units."
            )

    # ==================================================================
    # TASK 2: Delivery Routing (4 steps)
    #
    # Step 1: Assign 4 initial orders to 3 drivers
    # Step 2: 2 new urgent orders arrive + traffic delay
    # Step 3: Driver reports vehicle issue (capacity drop)
    # Step 4: Final adjustments
    # ==================================================================

    def _init_delivery_routing(self):
        rng = self._rng
        store_names = [s for s in STORE_LOCATIONS if s != "DC"]
        selected = rng.sample(store_names, 6)

        # Generate 4 initial orders (2 more added at step 2)
        orders = []
        for i in range(4):
            orders.append({
                "order_id": f"ORD{i+1:03d}",
                "destination": selected[i],
                "weight_kg": round(rng.uniform(20, 120), 1),
                "deadline_hours": round(rng.uniform(2.0, 5.0), 1),
                "priority": rng.choice(["standard", "standard", "express"]),
                "status": "pending",
            })

        # Pre-generate the 2 urgent orders for step 2
        urgent_orders = []
        for i in range(4, 6):
            urgent_orders.append({
                "order_id": f"ORD{i+1:03d}",
                "destination": selected[i],
                "weight_kg": round(rng.uniform(30, 80), 1),
                "deadline_hours": round(rng.uniform(1.0, 2.5), 1),
                "priority": "express",
                "status": "pending",
            })

        drivers = []
        for i in range(3):
            drivers.append({
                "driver_id": f"D{i+1}",
                "vehicle_capacity_kg": rng.randint(200, 400),
                "remaining_shift_hours": round(rng.uniform(5.0, 8.0), 1),
                "assigned_orders": [],
                "used_capacity_kg": 0.0,
            })

        distances = {}
        for store in selected:
            loc = STORE_LOCATIONS[store]
            dc = STORE_LOCATIONS["DC"]
            distances[store] = round(
                math.sqrt((loc[0] - dc[0]) ** 2 + (loc[1] - dc[1]) ** 2), 1
            )

        self._env_state = {
            "orders": orders,
            "drivers": drivers,
            "distances_from_dc": distances,
            "speed_kmh": 30,
            "_urgent_orders": urgent_orders,
            "_traffic_delay_store": selected[rng.randint(0, 3)],
            "_broken_driver": f"D{rng.randint(1, 3)}",
            "_events_applied": [],
        }

    def _step_delivery_routing(self, decision: Dict) -> Tuple[float, str]:
        s = self._env_state
        step = self._step_num

        # Apply dynamic events
        if step == 2:
            self._apply_routing_event_step2()
        elif step == 3:
            self._apply_routing_event_step3()

        # Parse assignments from this step
        assignments = decision.get("assignments", [])
        if not isinstance(assignments, list):
            assignments = []

        orders = s["orders"]
        drivers = s["drivers"]
        distances = s["distances_from_dc"]
        speed = s["speed_kmh"]

        order_map = {o["order_id"]: o for o in orders if o["status"] == "pending"}
        driver_map = {d["driver_id"]: d for d in drivers}

        newly_assigned = 0
        for a in assignments:
            if not isinstance(a, dict):
                continue
            oid = a.get("order_id", "")
            did = a.get("driver_id", "")
            if oid not in order_map or did not in driver_map:
                continue
            order = order_map[oid]
            driver = driver_map[did]

            # Apply assignment
            remaining_cap = driver["vehicle_capacity_kg"] - driver["used_capacity_kg"]
            if order["weight_kg"] <= remaining_cap:
                driver["assigned_orders"].append(oid)
                driver["used_capacity_kg"] += order["weight_kg"]
                order["status"] = "assigned"
                newly_assigned += 1

        # Grade this step
        pending = [o for o in orders if o["status"] == "pending"]
        total = len(orders)
        assigned_total = sum(1 for o in orders if o["status"] == "assigned")

        # On-time check for assigned orders
        on_time = 0
        for d in drivers:
            travel = 0.0
            for oid in d["assigned_orders"]:
                order = next((o for o in orders if o["order_id"] == oid), None)
                if order:
                    dist = distances.get(order["destination"], 20.0)
                    travel += dist / speed
                    if travel <= order["deadline_hours"]:
                        on_time += 1

        on_time_rate = on_time / assigned_total if assigned_total > 0 else 0.0

        # Capacity check
        violations = sum(
            1 for d in drivers if d["used_capacity_kg"] > d["vehicle_capacity_kg"]
        )
        cap_score = 1.0 - violations / len(drivers)

        # Coverage (how many orders assigned so far vs total)
        coverage = assigned_total / total if total > 0 else 0.0

        # Balance
        counts = [len(d["assigned_orders"]) for d in drivers]
        if sum(counts) > 0:
            mean_c = sum(counts) / len(counts)
            var = sum((c - mean_c) ** 2 for c in counts) / len(counts)
            balance = max(0.0, 1.0 - math.sqrt(var) / max(mean_c, 1))
        else:
            balance = 0.0

        reward = 0.30 * on_time_rate + 0.25 * cap_score + 0.25 * coverage + 0.20 * balance

        feedback = (
            f"Assigned this step: {newly_assigned} | "
            f"Total assigned: {assigned_total}/{total} | "
            f"On-time: {on_time} | Capacity violations: {violations}"
        )
        return reward, feedback

    def _apply_routing_event_step2(self):
        """2 new urgent orders + traffic delay."""
        s = self._env_state
        s["orders"].extend(s["_urgent_orders"])
        store = s["_traffic_delay_store"]
        if store in s["distances_from_dc"]:
            s["distances_from_dc"][store] = round(
                s["distances_from_dc"][store] * 2.0, 1
            )
        s["_events_applied"].append(
            f"NEW ORDERS: 2 urgent express orders added. "
            f"TRAFFIC: Route to {store} now takes 2x longer."
        )

    def _apply_routing_event_step3(self):
        """Driver reports vehicle issue — capacity reduced."""
        s = self._env_state
        did = s["_broken_driver"]
        for d in s["drivers"]:
            if d["driver_id"] == did:
                old_cap = d["vehicle_capacity_kg"]
                d["vehicle_capacity_kg"] = int(old_cap * 0.6)
                s["_events_applied"].append(
                    f"VEHICLE ISSUE: {did} capacity reduced from "
                    f"{old_cap}kg to {d['vehicle_capacity_kg']}kg."
                )
                break

    # ==================================================================
    # TASK 3: Demand Surge (5 steps)
    #
    # Step 1: Initial procurement plan (all suppliers active)
    # Step 2: Disruption — one supplier goes offline
    # Step 3: Demand forecast updates
    # Step 4: Warehouse capacity alert — redistribute
    # Step 5: Final review
    # ==================================================================

    def _init_demand_surge(self):
        rng = self._rng
        warehouses = []
        for i in range(3):
            inventory = {}
            cap = rng.randint(800, 1500)
            total = 0
            for cat in PRODUCT_CATEGORIES:
                qty = rng.randint(20, 150)
                inventory[cat] = qty
                total += qty
            warehouses.append({
                "warehouse_id": f"WH{i+1}", "inventory": inventory,
                "max_capacity": cap, "current_total": total,
            })

        suppliers = []
        for i, name in enumerate(SUPPLIER_NAMES):
            suppliers.append({
                "supplier_id": f"S{i+1}", "name": name,
                "price_per_unit": round(rng.uniform(2.0, 8.0), 2),
                "lead_time_days": rng.randint(1, 4),
                "reliability_score": round(rng.uniform(0.5, 1.0), 2),
                "max_order_qty": rng.randint(200, 600),
                "status": "ACTIVE",
            })

        stores = []
        for s_name in ["Downtown Store", "Mall Outlet", "Suburban Branch"]:
            demand = {cat: rng.randint(40, 200) for cat in PRODUCT_CATEGORIES}
            stores.append({"store_name": s_name, "demand_forecast": demand})

        budget = round(rng.uniform(5000, 9000), 0)

        total_demand = {}
        for cat in PRODUCT_CATEGORIES:
            total_demand[cat] = sum(s["demand_forecast"][cat] for s in stores)

        self._env_state = {
            "warehouses": warehouses,
            "suppliers": suppliers,
            "stores": stores,
            "budget": budget,
            "budget_remaining": budget,
            "surge_days": 5,
            "total_demand": total_demand,
            "procurement_log": [],
            "redistribution_log": [],
            "_offline_supplier": f"S{rng.randint(1, 4)}",
            "_demand_changes": {
                rng.choice(PRODUCT_CATEGORIES): 1.4,
                rng.choice(PRODUCT_CATEGORIES): 0.7,
            },
            "_capacity_alert_wh": f"WH{rng.randint(1, 3)}",
            "_events_applied": [],
        }

    def _step_demand_surge(self, decision: Dict) -> Tuple[float, str]:
        s = self._env_state
        step = self._step_num

        # Apply events at appropriate steps
        if step == 2:
            self._apply_surge_event_step2()
        elif step == 3:
            self._apply_surge_event_step3()
        elif step == 4:
            self._apply_surge_event_step4()

        # Process procurement orders
        procurement = decision.get("procurement_orders", [])
        if not isinstance(procurement, list):
            procurement = []

        suppliers = {sup["supplier_id"]: sup for sup in s["suppliers"]}
        ordered_offline = 0
        step_cost = 0.0

        for order in procurement:
            if not isinstance(order, dict):
                continue
            sid = order.get("supplier_id", "")
            product = order.get("product", "")
            qty = order.get("quantity", 0)
            dest = order.get("destination_warehouse", "")

            if not isinstance(qty, (int, float)) or qty <= 0:
                continue
            qty = int(qty)
            if sid not in suppliers or product not in PRODUCT_CATEGORIES:
                continue

            sup = suppliers[sid]
            if sup["status"] == "OFFLINE":
                ordered_offline += 1
                continue

            wh = next((w for w in s["warehouses"] if w["warehouse_id"] == dest), None)
            if not wh:
                continue

            actual_qty = min(qty, sup["max_order_qty"])
            cost = actual_qty * sup["price_per_unit"]
            if cost > s["budget_remaining"]:
                actual_qty = int(s["budget_remaining"] / sup["price_per_unit"])
                cost = actual_qty * sup["price_per_unit"]
            if actual_qty <= 0:
                continue

            s["budget_remaining"] -= cost
            step_cost += cost
            wh["inventory"][product] = wh["inventory"].get(product, 0) + actual_qty
            wh["current_total"] += actual_qty
            s["procurement_log"].append({
                "step": step, "supplier": sid, "product": product,
                "quantity": actual_qty, "cost": cost, "warehouse": dest,
            })

        # Process redistribution
        redistribution = decision.get("redistribution", [])
        if not isinstance(redistribution, list):
            redistribution = []

        for move in redistribution:
            if not isinstance(move, dict):
                continue
            from_wh = move.get("from_warehouse", "")
            to_wh = move.get("to_warehouse", "")
            product = move.get("product", "")
            qty = move.get("quantity", 0)
            if not isinstance(qty, (int, float)) or qty <= 0:
                continue
            qty = int(qty)
            if product not in PRODUCT_CATEGORIES:
                continue

            src = next((w for w in s["warehouses"] if w["warehouse_id"] == from_wh), None)
            dst = next((w for w in s["warehouses"] if w["warehouse_id"] == to_wh), None)
            if not src or not dst:
                continue

            actual = min(qty, src["inventory"].get(product, 0))
            if actual > 0:
                src["inventory"][product] -= actual
                src["current_total"] -= actual
                dst["inventory"][product] = dst["inventory"].get(product, 0) + actual
                dst["current_total"] += actual
                s["redistribution_log"].append({
                    "step": step, "from": from_wh, "to": to_wh,
                    "product": product, "quantity": actual,
                })

        # --- Grade this step ---
        total_demand = s["total_demand"]

        # Fulfillment
        total_supply = {cat: 0 for cat in PRODUCT_CATEGORIES}
        for wh in s["warehouses"]:
            for cat in PRODUCT_CATEGORIES:
                total_supply[cat] += wh["inventory"].get(cat, 0)

        fulfillment_rates = []
        for cat in PRODUCT_CATEGORIES:
            d = total_demand.get(cat, 1)
            sup = total_supply.get(cat, 0)
            fulfillment_rates.append(min(1.0, sup / d) if d > 0 else 1.0)
        fulfillment = sum(fulfillment_rates) / len(fulfillment_rates)

        # Budget
        budget_score = 1.0 if s["budget_remaining"] >= 0 else max(
            0.0, 1.0 + s["budget_remaining"] / s["budget"]
        )

        # Disruption
        disruption_score = 1.0 if ordered_offline == 0 else max(
            0.0, 1.0 - ordered_offline * 0.3
        )

        # Balance
        fill_rates = []
        for wh in s["warehouses"]:
            fill_rates.append(
                wh["current_total"] / wh["max_capacity"]
                if wh["max_capacity"] > 0 else 0.0
            )
        if len(fill_rates) > 1:
            mean_f = sum(fill_rates) / len(fill_rates)
            var = sum((r - mean_f) ** 2 for r in fill_rates) / len(fill_rates)
            balance = max(0.0, 1.0 - math.sqrt(var) * 2)
        else:
            balance = 1.0

        # Overstock
        overstock_penalties = []
        for cat in PRODUCT_CATEGORIES:
            d = total_demand.get(cat, 1)
            sup = total_supply.get(cat, 0)
            if sup > d * 1.2:
                overstock_penalties.append(min(1.0, (sup - d * 1.2) / d))
            else:
                overstock_penalties.append(0.0)
        waste = 1.0 - (
            sum(overstock_penalties) / len(overstock_penalties)
            if overstock_penalties else 0.0
        )

        reward = (
            0.30 * fulfillment + 0.20 * budget_score + 0.20 * disruption_score
            + 0.15 * balance + 0.15 * waste
        )

        feedback = (
            f"Fulfillment: {fulfillment:.0%} | "
            f"Budget left: ${s['budget_remaining']:.0f} | "
            f"Offline orders: {ordered_offline} | "
            f"Cost this step: ${step_cost:.0f}"
        )
        return reward, feedback

    def _apply_surge_event_step2(self):
        """Disruption: one supplier goes offline."""
        s = self._env_state
        sid = s["_offline_supplier"]
        for sup in s["suppliers"]:
            if sup["supplier_id"] == sid:
                sup["status"] = "OFFLINE"
                s["_events_applied"].append(
                    f"DISRUPTION: Supplier {sid} ({sup['name']}) is now OFFLINE "
                    f"and cannot fulfill any orders."
                )
                break

    def _apply_surge_event_step3(self):
        """Demand forecast updates."""
        s = self._env_state
        changes = s["_demand_changes"]
        for cat, multiplier in changes.items():
            for store in s["stores"]:
                old = store["demand_forecast"][cat]
                store["demand_forecast"][cat] = int(old * multiplier)
            old_total = s["total_demand"][cat]
            s["total_demand"][cat] = sum(
                st["demand_forecast"][cat] for st in s["stores"]
            )
            direction = "increased" if multiplier > 1 else "decreased"
            s["_events_applied"].append(
                f"DEMAND UPDATE: {cat} demand {direction} "
                f"({old_total} → {s['total_demand'][cat]} units)."
            )

    def _apply_surge_event_step4(self):
        """Warehouse capacity alert."""
        s = self._env_state
        wh_id = s["_capacity_alert_wh"]
        for wh in s["warehouses"]:
            if wh["warehouse_id"] == wh_id:
                wh["max_capacity"] = int(wh["max_capacity"] * 0.7)
                s["_events_applied"].append(
                    f"CAPACITY ALERT: {wh_id} max capacity reduced to "
                    f"{wh['max_capacity']} (section closed for maintenance)."
                )
                break

    # ==================================================================
    # Scenario text formatting
    # ==================================================================

    def _build_scenario_text(self) -> str:
        formatters = {
            "shelf_restock": self._format_shelf_restock,
            "delivery_routing": self._format_delivery_routing,
            "demand_surge": self._format_demand_surge,
        }
        return formatters[self._task_name]()

    def _format_shelf_restock(self) -> str:
        s = self._env_state
        step = self._step_num + 1  # next step to take
        slots = s["slots_per_step"][step - 1]
        restocked = s["restocked"]

        lines = [
            f"=== SHELF RESTOCK PRIORITY — Step {step}/{self._max_steps} ===",
            "",
        ]

        # Show events if any new ones
        events = s.get("_events_applied", [])
        if events:
            for e in events:
                lines.append(f"** EVENT: {e}")
            lines.append("")
            s["_events_applied"] = []

        if restocked:
            lines.append(f"Already restocked: {', '.join(restocked)}")
            lines.append("")

        lines.extend([
            f"Select {slots} product(s) to restock NOW (most urgent first).",
            "",
            f"{'ID':<6} {'Name':<25} {'Stock':>6} {'Daily Sales':>12} {'Capacity':>9} {'$/Unit':>8}",
            "-" * 70,
        ])

        for p in s["products"]:
            if p["product_id"] in restocked:
                continue
            days_left = round(
                p["current_stock"] / p["daily_sales_rate"], 1
            ) if p["daily_sales_rate"] > 0 else 999.0
            lines.append(
                f"{p['product_id']:<6} {p['name']:<25} {p['current_stock']:>6} "
                f"{p['daily_sales_rate']:>12.1f} {p['shelf_capacity']:>9} "
                f"${p['unit_revenue']:>7.2f}  ({days_left:.1f}d left)"
            )

        lines.extend([
            "",
            f'Respond with JSON: {{"restock_products": ["P001", "P002"]}}',
            f"Select exactly {slots} product ID(s) from the table above.",
        ])
        return "\n".join(lines)

    def _format_delivery_routing(self) -> str:
        s = self._env_state
        step = self._step_num + 1

        lines = [
            f"=== DELIVERY ROUTING — Step {step}/{self._max_steps} ===",
            "",
        ]

        events = s.get("_events_applied", [])
        if events:
            for e in events:
                lines.append(f"** EVENT: {e}")
            lines.append("")
            s["_events_applied"] = []

        # Pending orders
        pending = [o for o in s["orders"] if o["status"] == "pending"]
        if pending:
            lines.extend([
                "PENDING ORDERS:",
                f"{'Order':<8} {'Destination':<16} {'Weight':>8} {'Deadline':>10} {'Priority':<8}",
                "-" * 55,
            ])
            for o in pending:
                lines.append(
                    f"{o['order_id']:<8} {o['destination']:<16} {o['weight_kg']:>7.1f}kg "
                    f"{o['deadline_hours']:>9.1f}h {o['priority']:<8}"
                )
        else:
            lines.append("All orders assigned.")

        lines.extend(["", "DRIVERS:"])
        for d in s["drivers"]:
            remaining_cap = d["vehicle_capacity_kg"] - d["used_capacity_kg"]
            lines.append(
                f"  {d['driver_id']}: capacity {remaining_cap:.0f}kg remaining "
                f"({d['used_capacity_kg']:.0f}/{d['vehicle_capacity_kg']}kg used), "
                f"shift {d['remaining_shift_hours']:.1f}h left, "
                f"orders: {d['assigned_orders'] or 'none'}"
            )

        lines.extend(["", "DISTANCES (from DC):"])
        for dest, dist in s["distances_from_dc"].items():
            tt = round(dist / s["speed_kmh"], 2)
            lines.append(f"  {dest}: {dist}km (~{tt}h)")

        if pending:
            lines.extend([
                "",
                "Assign pending orders to drivers. Consider capacity, deadlines, and balance.",
                '{"assignments": [{"order_id": "ORD001", "driver_id": "D1"}, ...]}',
            ])
        else:
            lines.extend([
                "",
                "All orders assigned. You may reassign by providing new assignments.",
                '{"assignments": []}',
            ])
        return "\n".join(lines)

    def _format_demand_surge(self) -> str:
        s = self._env_state
        step = self._step_num + 1

        lines = [
            f"=== DEMAND SURGE PLANNING — Step {step}/{self._max_steps} ===",
            f"Budget remaining: ${s['budget_remaining']:.0f} / ${s['budget']:.0f}",
            "",
        ]

        events = s.get("_events_applied", [])
        if events:
            for e in events:
                lines.append(f"** EVENT: {e}")
            lines.append("")
            s["_events_applied"] = []

        lines.append("WAREHOUSE INVENTORY:")
        for wh in s["warehouses"]:
            lines.append(
                f"  {wh['warehouse_id']} ({wh['current_total']}/{wh['max_capacity']} used):"
            )
            for cat, qty in wh["inventory"].items():
                lines.append(f"    {cat}: {qty}")

        lines.extend(["", "SUPPLIERS:"])
        for sup in s["suppliers"]:
            status = " ** OFFLINE **" if sup["status"] == "OFFLINE" else ""
            lines.append(
                f"  {sup['supplier_id']} ({sup['name']}){status}: "
                f"${sup['price_per_unit']}/unit, {sup['lead_time_days']}d lead, "
                f"max {sup['max_order_qty']} units"
            )

        lines.extend(["", "TOTAL DEMAND FORECAST:"])
        for cat, qty in s["total_demand"].items():
            total_supply = sum(
                wh["inventory"].get(cat, 0) for wh in s["warehouses"]
            )
            gap = max(0, qty - total_supply)
            status = f"GAP: {gap}" if gap > 0 else "COVERED"
            lines.append(f"  {cat}: {qty} needed, {total_supply} available — {status}")

        lines.extend([
            "",
            "Place procurement orders and/or redistribute inventory.",
            "Do NOT order from OFFLINE suppliers. Stay within budget.",
            "",
            '{"procurement_orders": [{"supplier_id": "S1", "product": "rice", '
            '"quantity": 100, "destination_warehouse": "WH1"}, ...],',
            ' "redistribution": [{"from_warehouse": "WH1", "to_warehouse": "WH2", '
            '"product": "flour", "quantity": 50}, ...]}',
        ])
        return "\n".join(lines)
