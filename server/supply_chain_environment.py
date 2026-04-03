# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Supply Chain Retail Environment Implementation.

Three tasks with increasing difficulty:
1. shelf_restock (easy) — prioritize which products to restock
2. delivery_routing (medium) — assign delivery orders to drivers
3. demand_surge (hard) — plan procurement under disruption
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

VALID_TASKS = {"shelf_restock", "delivery_routing", "demand_surge"}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class SupplyChainEnvironment(Environment):
    """
    Supply Chain Retail environment with 3 tasks.

    Tasks:
        shelf_restock — Easy: prioritize products for restocking
        delivery_routing — Medium: assign orders to drivers
        demand_surge — Hard: plan procurement under disruption
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name: str = ""
        self._scenario_data: Dict[str, Any] = {}
        self._rng: random_module.Random = random_module.Random(42)
        self._done = False

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

        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0
        )
        self._task_name = task_name
        self._done = False
        self._rng = random_module.Random(seed if seed is not None else 42)

        generators = {
            "shelf_restock": self._generate_shelf_restock,
            "delivery_routing": self._generate_delivery_routing,
            "demand_surge": self._generate_demand_surge,
        }
        self._scenario_data = generators[task_name]()
        scenario_text = self._format_scenario_text(task_name, self._scenario_data)

        return SupplyChainObservation(
            task_name=task_name,
            scenario_text=scenario_text,
            scenario_data=self._scenario_data,
            done=False,
            reward=0.0,
        )

    def step(self, action: SupplyChainAction, **kwargs) -> SupplyChainObservation:
        self._state.step_count += 1
        self._done = True

        # If step called without reset (HTTP stateless mode), do a default reset
        if not self._task_name:
            self.reset(task_name="shelf_restock", seed=42)

        graders = {
            "shelf_restock": self._grade_shelf_restock,
            "delivery_routing": self._grade_delivery_routing,
            "demand_surge": self._grade_demand_surge,
        }

        score, breakdown, feedback = graders[self._task_name](action.decision)
        score = max(0.0, min(1.0, score))

        return SupplyChainObservation(
            task_name=self._task_name,
            scenario_text="",
            scenario_data={},
            score_breakdown=breakdown,
            feedback=feedback,
            done=True,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._state

    # ==================================================================
    # TASK 1: Shelf Restock Priority (Easy)
    # ==================================================================

    def _generate_shelf_restock(self) -> Dict[str, Any]:
        rng = self._rng
        num_products = 10
        restock_slots = 4
        products = []
        selected = rng.sample(PRODUCT_CATALOG, num_products)

        for pid, name, base_revenue in selected:
            daily_sales = round(rng.uniform(5, 50), 1)
            shelf_capacity = rng.randint(40, 100)
            current_stock = rng.randint(0, int(shelf_capacity * 0.4))
            unit_revenue = round(base_revenue * rng.uniform(0.8, 1.2), 2)
            products.append(
                {
                    "product_id": pid,
                    "name": name,
                    "current_stock": current_stock,
                    "daily_sales_rate": daily_sales,
                    "shelf_capacity": shelf_capacity,
                    "unit_revenue": unit_revenue,
                }
            )

        return {"products": products, "restock_slots": restock_slots}

    def _grade_shelf_restock(
        self, decision: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float], str]:
        products = self._scenario_data["products"]
        restock_slots = self._scenario_data["restock_slots"]

        # Compute optimal ranking by urgency
        urgency = []
        for p in products:
            stock = max(p["current_stock"], 0.1)
            score = (p["daily_sales_rate"] * p["unit_revenue"]) / stock
            urgency.append((p["product_id"], score))
        urgency.sort(key=lambda x: x[1], reverse=True)
        optimal_ids = [u[0] for u in urgency[:restock_slots]]

        # Parse agent's decision
        agent_picks = decision.get("restock_products", [])
        if not isinstance(agent_picks, list):
            agent_picks = []

        valid_product_ids = {p["product_id"] for p in products}

        # --- Feasibility (20%) ---
        valid_picks = [p for p in agent_picks if p in valid_product_ids]
        if len(agent_picks) == restock_slots and len(valid_picks) == restock_slots:
            feasibility = 1.0
        elif len(valid_picks) > 0:
            feasibility = 0.5 * (len(valid_picks) / restock_slots)
        else:
            feasibility = 0.0

        # --- Selection quality (50%) ---
        correct = set(valid_picks) & set(optimal_ids)
        selection = len(correct) / restock_slots if restock_slots > 0 else 0.0

        # --- Priority order (30%) ---
        if len(correct) >= 2:
            # Check ordering among correctly selected items
            optimal_rank = {pid: i for i, pid in enumerate(optimal_ids)}
            correct_in_agent_order = [p for p in valid_picks if p in correct]
            correct_optimal_order = sorted(
                correct_in_agent_order, key=lambda x: optimal_rank.get(x, 99)
            )
            # Count concordant pairs
            n = len(correct_in_agent_order)
            concordant = 0
            total_pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total_pairs += 1
                    rank_i_agent = correct_in_agent_order.index(
                        correct_in_agent_order[i]
                    )
                    rank_j_agent = correct_in_agent_order.index(
                        correct_in_agent_order[j]
                    )
                    rank_i_opt = correct_optimal_order.index(
                        correct_in_agent_order[i]
                    )
                    rank_j_opt = correct_optimal_order.index(
                        correct_in_agent_order[j]
                    )
                    if (rank_i_agent - rank_j_agent) * (rank_i_opt - rank_j_opt) > 0:
                        concordant += 1
            priority = concordant / total_pairs if total_pairs > 0 else 1.0
        elif len(correct) == 1:
            priority = 0.5
        else:
            priority = 0.0

        total = 0.50 * selection + 0.30 * priority + 0.20 * feasibility

        breakdown = {
            "selection_quality": round(selection, 3),
            "priority_order": round(priority, 3),
            "feasibility": round(feasibility, 3),
        }
        feedback_parts = [f"Score: {total:.2f}/1.00"]
        feedback_parts.append(
            f"Optimal top-{restock_slots}: {optimal_ids}"
        )
        feedback_parts.append(f"Your picks: {agent_picks}")
        feedback_parts.append(
            f"Correct selections: {len(correct)}/{restock_slots}"
        )
        return total, breakdown, " | ".join(feedback_parts)

    # ==================================================================
    # TASK 2: Delivery Route Assignment (Medium)
    # ==================================================================

    def _generate_delivery_routing(self) -> Dict[str, Any]:
        rng = self._rng
        store_names = list(STORE_LOCATIONS.keys())
        store_names.remove("DC")
        selected_stores = rng.sample(store_names, 6)

        orders = []
        for i, store in enumerate(selected_stores):
            order_id = f"ORD{i+1:03d}"
            weight = round(rng.uniform(20, 150), 1)
            deadline_hours = round(rng.uniform(1.5, 5.0), 1)
            priority = rng.choice(["standard", "express"])
            orders.append(
                {
                    "order_id": order_id,
                    "destination": store,
                    "weight_kg": weight,
                    "deadline_hours": deadline_hours,
                    "priority": priority,
                }
            )

        drivers = []
        for i in range(3):
            driver_id = f"D{i+1}"
            capacity = rng.randint(200, 400)
            shift_hours = round(rng.uniform(4.0, 8.0), 1)
            drivers.append(
                {
                    "driver_id": driver_id,
                    "vehicle_capacity_kg": capacity,
                    "remaining_shift_hours": shift_hours,
                }
            )

        # Precompute distances from DC to each store
        distances = {}
        for store in selected_stores:
            loc = STORE_LOCATIONS[store]
            dc = STORE_LOCATIONS["DC"]
            dist = math.sqrt((loc[0] - dc[0]) ** 2 + (loc[1] - dc[1]) ** 2)
            distances[store] = round(dist, 1)

        return {
            "orders": orders,
            "drivers": drivers,
            "distances_from_dc": distances,
            "speed_kmh": 30,
        }

    def _grade_delivery_routing(
        self, decision: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float], str]:
        orders = self._scenario_data["orders"]
        drivers = self._scenario_data["drivers"]
        distances = self._scenario_data["distances_from_dc"]
        speed = self._scenario_data["speed_kmh"]

        assignments_raw = decision.get("assignments", [])
        if not isinstance(assignments_raw, list):
            assignments_raw = []

        order_map = {o["order_id"]: o for o in orders}
        driver_map = {d["driver_id"]: d for d in drivers}
        valid_order_ids = set(order_map.keys())
        valid_driver_ids = set(driver_map.keys())

        # Parse assignments
        driver_orders: Dict[str, List[Dict]] = {d: [] for d in valid_driver_ids}
        assigned_orders = set()
        invalid_count = 0

        for a in assignments_raw:
            if not isinstance(a, dict):
                invalid_count += 1
                continue
            oid = a.get("order_id", "")
            did = a.get("driver_id", "")
            if oid in valid_order_ids and did in valid_driver_ids and oid not in assigned_orders:
                driver_orders[did].append(order_map[oid])
                assigned_orders.add(oid)
            else:
                invalid_count += 1

        unassigned = valid_order_ids - assigned_orders

        # --- On-time delivery (35%) ---
        on_time = 0
        total_orders = len(orders)
        for did, ord_list in driver_orders.items():
            travel_time = 0.0
            for o in ord_list:
                dest = o["destination"]
                dist = distances.get(dest, 20.0)
                travel_time += dist / speed
                if travel_time <= o["deadline_hours"]:
                    on_time += 1
        # Unassigned orders are never on-time
        on_time_score = on_time / total_orders if total_orders > 0 else 0.0

        # --- Capacity compliance (25%) ---
        capacity_violations = 0
        for did, ord_list in driver_orders.items():
            total_weight = sum(o["weight_kg"] for o in ord_list)
            cap = driver_map[did]["vehicle_capacity_kg"]
            if total_weight > cap:
                capacity_violations += 1
        capacity_score = 1.0 - (capacity_violations / len(drivers))

        # --- Efficiency (25%) ---
        total_distance = 0.0
        for did, ord_list in driver_orders.items():
            for o in ord_list:
                total_distance += distances.get(o["destination"], 20.0)

        # Greedy optimal: assign each order to nearest slot
        optimal_distance = sum(distances.get(o["destination"], 20.0) for o in orders)
        if optimal_distance > 0 and total_distance > 0:
            efficiency_score = min(1.0, optimal_distance / total_distance)
        elif total_distance == 0 and len(assigned_orders) == 0:
            efficiency_score = 0.0
        else:
            efficiency_score = 0.5

        # --- Driver balance (15%) ---
        counts = [len(v) for v in driver_orders.values()]
        if len(counts) > 1 and sum(counts) > 0:
            mean_count = sum(counts) / len(counts)
            variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
            std_dev = math.sqrt(variance)
            balance_score = max(0.0, 1.0 - std_dev / mean_count) if mean_count > 0 else 0.0
        elif sum(counts) > 0:
            balance_score = 1.0
        else:
            balance_score = 0.0

        # Penalty for unassigned orders
        coverage = len(assigned_orders) / total_orders if total_orders > 0 else 0.0
        coverage_penalty = coverage  # scale all scores by coverage

        total = coverage_penalty * (
            0.35 * on_time_score
            + 0.25 * capacity_score
            + 0.25 * efficiency_score
            + 0.15 * balance_score
        )

        breakdown = {
            "on_time_delivery": round(on_time_score, 3),
            "capacity_compliance": round(capacity_score, 3),
            "efficiency": round(efficiency_score, 3),
            "driver_balance": round(balance_score, 3),
            "coverage": round(coverage, 3),
        }
        feedback = (
            f"Score: {total:.2f}/1.00 | "
            f"Assigned: {len(assigned_orders)}/{total_orders} | "
            f"On-time: {on_time}/{total_orders} | "
            f"Capacity violations: {capacity_violations} | "
            f"Unassigned: {len(unassigned)}"
        )
        return total, breakdown, feedback

    # ==================================================================
    # TASK 3: Demand Surge Planning with Disruption (Hard)
    # ==================================================================

    def _generate_demand_surge(self) -> Dict[str, Any]:
        rng = self._rng

        # Warehouses
        warehouses = []
        for i in range(3):
            wh_id = f"WH{i+1}"
            inventory = {}
            max_capacity = rng.randint(800, 1500)
            current_total = 0
            for cat in PRODUCT_CATEGORIES:
                qty = rng.randint(20, 150)
                inventory[cat] = qty
                current_total += qty
            warehouses.append(
                {
                    "warehouse_id": wh_id,
                    "inventory": inventory,
                    "max_capacity": max_capacity,
                    "current_total": current_total,
                }
            )

        # Suppliers (one will be offline)
        suppliers = []
        for i, name in enumerate(SUPPLIER_NAMES):
            sid = f"S{i+1}"
            suppliers.append(
                {
                    "supplier_id": sid,
                    "name": name,
                    "price_per_unit": round(rng.uniform(2.0, 8.0), 2),
                    "lead_time_days": rng.randint(1, 4),
                    "reliability_score": round(rng.uniform(0.5, 1.0), 2),
                    "max_order_qty": rng.randint(200, 600),
                    "status": "ACTIVE",
                }
            )
        # Mark one supplier as offline
        offline_idx = rng.randint(0, len(suppliers) - 1)
        suppliers[offline_idx]["status"] = "OFFLINE"

        # Store demand forecasts (surge in 5 days)
        stores = []
        store_names = ["Downtown Store", "Mall Outlet", "Suburban Branch"]
        for s_name in store_names:
            demand = {}
            for cat in PRODUCT_CATEGORIES:
                # Surge = higher demand
                demand[cat] = rng.randint(40, 200)
            stores.append({"store_name": s_name, "demand_forecast": demand})

        # Budget
        budget = round(rng.uniform(3000, 6000), 0)

        # Total demand for reference
        total_demand = {}
        for cat in PRODUCT_CATEGORIES:
            total_demand[cat] = sum(s["demand_forecast"][cat] for s in stores)

        return {
            "warehouses": warehouses,
            "suppliers": suppliers,
            "stores": stores,
            "budget": budget,
            "surge_days": 5,
            "total_demand": total_demand,
        }

    def _grade_demand_surge(
        self, decision: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float], str]:
        data = self._scenario_data
        warehouses = {w["warehouse_id"]: dict(w) for w in data["warehouses"]}
        suppliers = {s["supplier_id"]: s for s in data["suppliers"]}
        total_demand = data["total_demand"]
        budget = data["budget"]

        procurement = decision.get("procurement_orders", [])
        if not isinstance(procurement, list):
            procurement = []
        redistribution = decision.get("redistribution", [])
        if not isinstance(redistribution, list):
            redistribution = []

        # --- Process procurement orders ---
        total_cost = 0.0
        ordered_to_offline = 0
        incoming: Dict[str, Dict[str, int]] = {
            wh: {cat: 0 for cat in PRODUCT_CATEGORIES}
            for wh in warehouses
        }

        for order in procurement:
            if not isinstance(order, dict):
                continue
            sid = order.get("supplier_id", "")
            product = order.get("product", "")
            qty = order.get("quantity", 0)
            dest_wh = order.get("destination_warehouse", "")

            if not isinstance(qty, (int, float)) or qty <= 0:
                continue
            qty = int(qty)

            if sid not in suppliers:
                continue
            supplier = suppliers[sid]

            if supplier["status"] == "OFFLINE":
                ordered_to_offline += 1
                continue  # Order won't be fulfilled

            if product not in PRODUCT_CATEGORIES:
                continue
            if dest_wh not in warehouses:
                continue

            actual_qty = min(qty, supplier["max_order_qty"])
            cost = actual_qty * supplier["price_per_unit"]
            total_cost += cost
            incoming[dest_wh][product] += actual_qty

        # --- Process redistribution ---
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

            if (
                from_wh not in warehouses
                or to_wh not in warehouses
                or product not in PRODUCT_CATEGORIES
            ):
                continue

            available = warehouses[from_wh]["inventory"].get(product, 0)
            actual_move = min(qty, available)
            warehouses[from_wh]["inventory"][product] -= actual_move
            if to_wh not in incoming:
                incoming[to_wh] = {cat: 0 for cat in PRODUCT_CATEGORIES}
            incoming[to_wh][product] += actual_move

        # --- Compute available supply per category ---
        total_supply: Dict[str, int] = {cat: 0 for cat in PRODUCT_CATEGORIES}
        for wh_id, wh in warehouses.items():
            for cat in PRODUCT_CATEGORIES:
                total_supply[cat] += wh["inventory"].get(cat, 0)
                total_supply[cat] += incoming.get(wh_id, {}).get(cat, 0)

        # --- Demand fulfillment (30%) ---
        fulfillment_rates = []
        for cat in PRODUCT_CATEGORIES:
            demand = total_demand.get(cat, 1)
            supply = total_supply.get(cat, 0)
            rate = min(1.0, supply / demand) if demand > 0 else 1.0
            fulfillment_rates.append(rate)
        demand_fulfillment = sum(fulfillment_rates) / len(fulfillment_rates)

        # --- Budget compliance (20%) ---
        if total_cost <= budget:
            budget_score = 1.0
        elif total_cost <= budget * 1.2:
            budget_score = 1.0 - (total_cost - budget) / (budget * 0.2)
        else:
            budget_score = max(0.0, 0.5 - (total_cost - budget * 1.2) / budget)

        # --- Disruption handling (20%) ---
        if ordered_to_offline == 0:
            disruption_score = 1.0
        else:
            disruption_score = max(0.0, 1.0 - ordered_to_offline * 0.3)

        # --- Inventory balance (15%) ---
        wh_fill_rates = []
        for wh_id, wh in warehouses.items():
            total_wh = sum(wh["inventory"].values()) + sum(
                incoming.get(wh_id, {}).values()
            )
            cap = wh["max_capacity"]
            wh_fill_rates.append(total_wh / cap if cap > 0 else 0.0)
        if len(wh_fill_rates) > 1:
            mean_fill = sum(wh_fill_rates) / len(wh_fill_rates)
            var = sum((r - mean_fill) ** 2 for r in wh_fill_rates) / len(wh_fill_rates)
            balance_score = max(0.0, 1.0 - math.sqrt(var) * 2)
        else:
            balance_score = 1.0

        # --- Waste prevention (15%) ---
        overstock_penalties = []
        for cat in PRODUCT_CATEGORIES:
            demand = total_demand.get(cat, 1)
            supply = total_supply.get(cat, 0)
            if supply > demand * 1.2:
                excess = (supply - demand * 1.2) / demand
                overstock_penalties.append(min(1.0, excess))
            else:
                overstock_penalties.append(0.0)
        waste_score = 1.0 - (
            sum(overstock_penalties) / len(overstock_penalties)
            if overstock_penalties
            else 0.0
        )

        total = (
            0.30 * demand_fulfillment
            + 0.20 * budget_score
            + 0.20 * disruption_score
            + 0.15 * balance_score
            + 0.15 * waste_score
        )

        # If agent submitted no orders at all, they still get credit for
        # existing inventory fulfillment + not overspending + not ordering
        # from offline supplier — this ensures score variance.

        breakdown = {
            "demand_fulfillment": round(demand_fulfillment, 3),
            "budget_compliance": round(budget_score, 3),
            "disruption_handling": round(disruption_score, 3),
            "inventory_balance": round(balance_score, 3),
            "waste_prevention": round(waste_score, 3),
        }
        feedback = (
            f"Score: {total:.2f}/1.00 | "
            f"Fulfillment: {demand_fulfillment:.0%} | "
            f"Cost: ${total_cost:.0f}/{budget:.0f} budget | "
            f"Offline orders: {ordered_to_offline}"
        )
        return total, breakdown, feedback

    # ==================================================================
    # Scenario formatting (human-readable for LLM)
    # ==================================================================

    def _format_scenario_text(
        self, task_name: str, data: Dict[str, Any]
    ) -> str:
        if task_name == "shelf_restock":
            return self._format_shelf_restock(data)
        elif task_name == "delivery_routing":
            return self._format_delivery_routing(data)
        elif task_name == "demand_surge":
            return self._format_demand_surge(data)
        return ""

    def _format_shelf_restock(self, data: Dict[str, Any]) -> str:
        lines = [
            "=== SHELF RESTOCK PRIORITY ===",
            "",
            f"You are a store manager. You have time to restock only {data['restock_slots']} products before the store opens.",
            "Choose which products to restock, ordered by priority (most urgent first).",
            "",
            "Product Inventory:",
            f"{'ID':<6} {'Name':<25} {'Stock':>6} {'Daily Sales':>12} {'Capacity':>9} {'Revenue/Unit':>13}",
            "-" * 75,
        ]
        for p in data["products"]:
            days_left = (
                round(p["current_stock"] / p["daily_sales_rate"], 1)
                if p["daily_sales_rate"] > 0
                else 999.0
            )
            lines.append(
                f"{p['product_id']:<6} {p['name']:<25} {p['current_stock']:>6} "
                f"{p['daily_sales_rate']:>12.1f} {p['shelf_capacity']:>9} "
                f"${p['unit_revenue']:>12.2f}   ({days_left:.1f} days left)"
            )
        lines.extend(
            [
                "",
                f"Select exactly {data['restock_slots']} products to restock, ordered by urgency.",
                "Consider: products with low stock, high sales rate, and high revenue should be prioritized.",
                "",
                'Respond with JSON: {"restock_products": ["P001", "P002", "P003", "P004"]}',
            ]
        )
        return "\n".join(lines)

    def _format_delivery_routing(self, data: Dict[str, Any]) -> str:
        lines = [
            "=== DELIVERY ROUTE ASSIGNMENT ===",
            "",
            "You are a distribution center dispatcher. Assign each delivery order to a driver.",
            "",
            "Delivery Orders:",
            f"{'Order':<8} {'Destination':<16} {'Weight(kg)':>10} {'Deadline(hrs)':>13} {'Priority':<10}",
            "-" * 60,
        ]
        for o in data["orders"]:
            lines.append(
                f"{o['order_id']:<8} {o['destination']:<16} {o['weight_kg']:>10.1f} "
                f"{o['deadline_hours']:>13.1f} {o['priority']:<10}"
            )
        lines.extend(
            [
                "",
                "Available Drivers:",
                f"{'Driver':<8} {'Capacity(kg)':>12} {'Shift Left(hrs)':>15}",
                "-" * 38,
            ]
        )
        for d in data["drivers"]:
            lines.append(
                f"{d['driver_id']:<8} {d['vehicle_capacity_kg']:>12} "
                f"{d['remaining_shift_hours']:>15.1f}"
            )
        lines.extend(
            [
                "",
                "Distances from DC (km):",
            ]
        )
        for dest, dist in data["distances_from_dc"].items():
            travel_time = round(dist / data["speed_kmh"], 2)
            lines.append(f"  {dest}: {dist}km (~{travel_time}hrs at {data['speed_kmh']}km/h)")
        lines.extend(
            [
                "",
                "Assign ALL orders to drivers. Consider capacity, deadlines, and balance workload.",
                "",
                'Respond with JSON: {"assignments": [{"order_id": "ORD001", "driver_id": "D1"}, ...]}',
            ]
        )
        return "\n".join(lines)

    def _format_demand_surge(self, data: Dict[str, Any]) -> str:
        lines = [
            "=== DEMAND SURGE PLANNING WITH DISRUPTION ===",
            "",
            f"A festival is approaching in {data['surge_days']} days. Plan procurement and redistribution.",
            f"Budget: ${data['budget']:.0f}",
            "",
            "Warehouse Inventory:",
        ]
        for wh in data["warehouses"]:
            lines.append(
                f"  {wh['warehouse_id']} (capacity: {wh['max_capacity']}, used: {wh['current_total']}):"
            )
            for cat, qty in wh["inventory"].items():
                lines.append(f"    {cat}: {qty} units")

        lines.extend(["", "Suppliers:"])
        for s in data["suppliers"]:
            status_marker = " ** OFFLINE **" if s["status"] == "OFFLINE" else ""
            lines.append(
                f"  {s['supplier_id']} ({s['name']}){status_marker}: "
                f"${s['price_per_unit']}/unit, {s['lead_time_days']}d lead time, "
                f"reliability {s['reliability_score']}, max qty {s['max_order_qty']}"
            )

        lines.extend(["", "Store Demand Forecasts (next 5 days):"])
        for store in data["stores"]:
            lines.append(f"  {store['store_name']}:")
            for cat, qty in store["demand_forecast"].items():
                lines.append(f"    {cat}: {qty} units")

        lines.extend(["", "Total Demand by Category:"])
        for cat, qty in data["total_demand"].items():
            lines.append(f"  {cat}: {qty} units")

        lines.extend(
            [
                "",
                "DISRUPTION: One supplier is OFFLINE and cannot fulfill orders.",
                "",
                "Create a plan with:",
                "1. procurement_orders: which suppliers to order from, what product, quantity, and destination warehouse",
                "2. redistribution: move inventory between warehouses to balance supply",
                "",
                "Stay within budget. Do NOT order from the OFFLINE supplier. Minimize overstock.",
                "",
                "Respond with JSON:",
                '{"procurement_orders": [{"supplier_id": "S1", "product": "rice", "quantity": 100, "destination_warehouse": "WH1"}, ...],',
                ' "redistribution": [{"from_warehouse": "WH1", "to_warehouse": "WH3", "product": "flour", "quantity": 50}, ...]}',
            ]
        )
        return "\n".join(lines)
