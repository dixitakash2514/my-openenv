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
from openenv.core.env_server.types import EnvironmentMetadata, State

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
# Grader economics — explicit unit prices used by the P&L reward functions.
#
# These are NOT arbitrary weights. They model the realized profit/loss the
# agent is graded on: a judge reading the grader sees a P&L, not a weighted
# sum of ad-hoc criteria. Every reward in this env is computed as
#
#     profit    = revenue − costs − penalties
#     reward    = clip((profit − baseline_profit) / (optimal_profit − baseline_profit), 0, 1)
#
# where `baseline_profit` is the do-nothing outcome and `optimal_profit` is an
# oracle that cannot be exceeded. Random policies collapse toward 0 naturally;
# optimal policies approach 1. No engineered "do nothing" punishments are
# needed because inaction is already dominated by the P&L math.
# ---------------------------------------------------------------------------

# delivery_routing (medium task) — realized route economics
DELIVERY_UNIT_REVENUE_PER_KG = 2.50          # revenue per kg of order delivered on time
DELIVERY_LATE_PENALTY_PER_KG = 4.00          # late penalty scales with cargo (> revenue)
DELIVERY_UNFULFILLED_PENALTY_PER_KG = 15.00  # lost-sale cost per kg of unassigned/dropped order
DELIVERY_DWELL_HOURS_PER_STOP = 0.35         # service time at each drop-off (VRP-TW)

# demand_surge (hard task) — realized procurement P&L
SURGE_UNIT_SALE_PRICE = 10.00                # retail unit sale price
SURGE_STORAGE_COST_PER_EXCESS_UNIT = 0.50    # holding cost for overstock above 1.1x demand
SURGE_STOCKOUT_COST_PER_UNIT = 8.00          # lost-sale cost per unit of undersupply
SURGE_OFFLINE_ORDER_PENALTY = 400.00         # fine per order attempted against offline supplier


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

    def get_metadata(self) -> EnvironmentMetadata:
        """Rich metadata for /metadata endpoint — what judges see when probing the env."""
        return EnvironmentMetadata(
            name="supply_chain_retail",
            description=(
                "Supply Chain Retail — a stateful, multi-step OpenEnv for training "
                "and evaluating LLM agents on real-world supply chain decisions. "
                "Every reward is a unit-priced P&L, normalized against a "
                "do-nothing baseline and an oracle optimum so scores sit in [0, 1] "
                "with no ad-hoc weights.\n\n"
                "Tasks with dynamic mid-episode events:\n"
                "  1. shelf_restock (easy, 3 steps): prioritize restocking as demand "
                "spikes and surprise deliveries arrive. Reward = fraction of picks "
                "matching the optimal (sales_rate * unit_revenue / stock) ranking.\n"
                "  2. delivery_routing (medium, 4 steps): last-mile VRP-TW with "
                "driver capacity, time windows, traffic, and breakdowns. "
                "P&L = delivered_revenue - late_penalty - unfulfilled_penalty.\n"
                "  3. demand_surge (hard, 5 steps): multi-supplier procurement under "
                "disruption. P&L = sale_revenue - procurement_cost - stockout "
                "- storage - offline_order_penalty.\n"
                "All graders are deterministic (seeded) and return rewards in [0, 1]. "
                "Action: {decision: dict, reasoning: str}. "
                "Observation: {task_name, step_number, total_steps, scenario_text, "
                "scenario_data, feedback, score_breakdown, done, reward}."
            ),
            version="2.1.0",
            author="Akash Dixit (BlackEagle)",
            documentation_url="https://github.com/dixitakash2514/my-openenv",
        )

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

    def _compute_surge_bounds(self) -> Tuple[float, float]:
        """Compute baseline (do-nothing) and optimal (oracle) P&L for demand_surge.

        These fixed bounds normalize the per-step profit signal so the reward
        always lands in [0, 1]:

            reward = clip((profit - baseline) / (optimal - baseline), 0, 1)

        Baseline models a policy that never takes an action: realize the
        starting inventory, pay stockout penalties for every uncovered unit.

        Optimal models an oracle that knows the upcoming disruption and buys
        exactly the shortfall from the cheapest sustainably-active supplier,
        subject to the episode budget. This is an upper bound — a real policy
        cannot beat it, so the reward is guaranteed to stay in [0, 1].
        """
        s = self._env_state
        cats = PRODUCT_CATEGORIES
        initial_supply = {
            c: sum(wh["inventory"].get(c, 0) for wh in s["warehouses"]) for c in cats
        }
        demand = s["total_demand"]
        budget = s["budget"]
        offline_sid = s["_offline_supplier"]

        # --- Baseline: do nothing, realize only starting inventory ---
        baseline_revenue = sum(
            min(initial_supply[c], demand[c]) * SURGE_UNIT_SALE_PRICE for c in cats
        )
        baseline_stockout = sum(
            max(0, demand[c] - initial_supply[c]) * SURGE_STOCKOUT_COST_PER_UNIT
            for c in cats
        )
        baseline_profit = baseline_revenue - baseline_stockout

        # --- Optimal: oracle knows reliability and picks the best expected
        # cost-per-delivered-unit supplier. Effective price = price / reliability
        # because to deliver `N` units in expectation you must order `N/r`,
        # so the realized cost per delivered unit is price/r. Cheap-unreliable
        # suppliers lose to premium-reliable suppliers whenever their price
        # ratio is worse than their reliability ratio.
        active_future = [
            sup for sup in s["suppliers"] if sup["supplier_id"] != offline_sid
        ]
        if not active_future:
            return baseline_profit, baseline_profit + 1.0  # degenerate guard

        def effective_price(sup):
            r = max(sup.get("reliability", 1.0), 0.01)
            return sup["price_per_unit"] / r

        best_effective_price = min(effective_price(sup) for sup in active_future)
        total_shortfall = sum(
            max(0, demand[c] - initial_supply[c]) for c in cats
        )
        max_units_delivered = (
            budget / best_effective_price if best_effective_price > 0 else 0.0
        )
        units_to_procure = min(total_shortfall, max_units_delivered)

        if total_shortfall > 0:
            procured_fraction = units_to_procure / total_shortfall
        else:
            procured_fraction = 1.0

        optimal_final_supply = {
            c: initial_supply[c]
            + max(0, demand[c] - initial_supply[c]) * procured_fraction
            for c in cats
        }
        optimal_revenue = sum(
            min(optimal_final_supply[c], demand[c]) * SURGE_UNIT_SALE_PRICE
            for c in cats
        )
        optimal_procurement_cost = units_to_procure * best_effective_price
        optimal_stockout = sum(
            max(0, demand[c] - optimal_final_supply[c]) * SURGE_STOCKOUT_COST_PER_UNIT
            for c in cats
        )
        optimal_profit = (
            optimal_revenue - optimal_procurement_cost - optimal_stockout
        )

        # Guard: ensure a strictly positive normalization denominator.
        if optimal_profit <= baseline_profit:
            optimal_profit = baseline_profit + max(1.0, abs(baseline_profit) * 0.1)

        return baseline_profit, optimal_profit

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

        # Pure-precision reward: fraction of picks matching the optimal ranking.
        # Do-nothing → 0.0. Random pick of `slots` from the pool → ~slots/pool.
        # Heuristic that ranks by (sales_rate × unit_revenue) / stock → 1.0.
        #
        # The previous grader added a 0.4 feasibility floor that made random
        # agents score ~0.44 just by returning any `slots`-length list. That
        # floor is removed: submitting the wrong picks earns zero credit even
        # if the JSON is well-formed.
        correct = set(valid_picks) & set(optimal)
        selection_score = len(correct) / slots if slots > 0 else 0.0
        reward = selection_score

        # Apply restock: update stock for picked products
        for pid in valid_picks:
            for p in products:
                if p["product_id"] == pid:
                    p["current_stock"] = p["shelf_capacity"]
                    break
            s["restocked"].append(pid)

        feedback = (
            f"Picked: {valid_picks} | Optimal: {optimal} | "
            f"Correct: {len(correct)}/{slots} | selection={selection_score:.2f}"
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

        # 4 initial orders with loose deadlines + 2 urgent orders (injected
        # at step 2) with very tight deadlines. The discrimination lever
        # is that step-2 urgents must go to a driver whose committed route
        # time is near zero — otherwise the urgent arrives late even on
        # closest stores. A smart heuristic (Best-Fit-Decreasing by weight)
        # packs all 4 initial orders onto 2 drivers, leaving the 3rd
        # reserved for urgents. A random policy spreads orders across all
        # 3 drivers and has no empty driver for the urgents, so every
        # urgent arrives late. Weights ensure max_pair ≤ min driver cap so
        # the packing is feasible.
        orders = []
        for i in range(4):
            orders.append({
                "order_id": f"ORD{i+1:03d}",
                "destination": selected[i],
                "weight_kg": round(rng.uniform(70, 100), 1),
                "deadline_hours": round(rng.uniform(1.8, 2.5), 1),
                "priority": rng.choice(["standard", "standard", "express"]),
                "status": "pending",
            })

        # Urgent orders with very tight deadlines — they arrive at step 2.
        # Their deadline is short enough that only a driver at route_time ≈ 0
        # (fully reserved / unused) can deliver them on time.
        urgent_orders = []
        for i in range(4, 6):
            urgent_orders.append({
                "order_id": f"ORD{i+1:03d}",
                "destination": selected[i],
                "weight_kg": round(rng.uniform(55, 80), 1),
                "deadline_hours": round(rng.uniform(0.7, 0.9), 1),
                "priority": "express",
                "status": "pending",
            })

        drivers = []
        for i in range(3):
            drivers.append({
                "driver_id": f"D{i+1}",
                "vehicle_capacity_kg": rng.randint(210, 240),
                "remaining_shift_hours": round(rng.uniform(3.5, 4.5), 1),
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
            "dwell_hours_per_stop": DELIVERY_DWELL_HOURS_PER_STOP,
            "_urgent_orders": urgent_orders,
            "_broken_driver": f"D{rng.randint(1, 3)}",
            "_events_applied": [],
        }

    def _step_delivery_routing(self, decision: Dict) -> Tuple[float, str]:
        s = self._env_state
        step = self._step_num

        # NOTE: dynamic events are applied at the END of this handler (after
        # the action is processed), NOT at the start. The reason is that the
        # agent acts on the observation returned by the PREVIOUS step — if we
        # applied events at the top, the agent would be graded against state
        # changes it never saw. Events applied at the bottom are visible in
        # the observation we return here, so the next action can respond.

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

        # --- Unified P&L grader ---
        #
        # profit = delivered_revenue − late_penalty − unfulfilled_penalty
        # reward = clip((profit − step_baseline) / (step_optimal − step_baseline), 0, 1)
        #
        # Bounds are recomputed per step (not cached) because the order set
        # grows at step 2 — 2 urgent orders are injected. Recomputing against
        # the currently-existing order set keeps the normalization fair.
        total = len(orders)
        assigned_total = sum(1 for o in orders if o["status"] == "assigned")
        pending = [o for o in orders if o["status"] == "pending"]

        delivered_revenue = 0.0
        late_penalty = 0.0
        on_time_count = 0
        late_count = 0

        # Track orders that were assigned but couldn't be completed in the
        # driver's shift window — these revert to "effectively unfulfilled"
        # in the P&L math so a smart agent is rewarded for picking routes
        # that actually fit the day.
        shift_dropped_weight = 0.0

        for d in drivers:
            route_time = 0.0
            shift_limit = d.get("remaining_shift_hours", 99.0)
            for oid in d["assigned_orders"]:
                order = next((o for o in orders if o["order_id"] == oid), None)
                if not order:
                    continue
                dist = distances.get(order["destination"], 20.0)
                leg = dist / speed
                arrive = route_time + leg
                completed = arrive + DELIVERY_DWELL_HOURS_PER_STOP
                if completed > shift_limit:
                    # Driver runs out of shift — this and all later stops
                    # on this driver go undelivered (no revenue, bleed as
                    # unfulfilled).
                    shift_dropped_weight += order["weight_kg"]
                    continue
                route_time = completed
                if arrive <= order["deadline_hours"]:
                    on_time_count += 1
                    delivered_revenue += (
                        order["weight_kg"] * DELIVERY_UNIT_REVENUE_PER_KG
                    )
                else:
                    late_count += 1
                    late_penalty += order["weight_kg"] * DELIVERY_LATE_PENALTY_PER_KG

        unfulfilled_penalty = sum(
            o["weight_kg"] * DELIVERY_UNFULFILLED_PENALTY_PER_KG for o in pending
        )
        unfulfilled_penalty += shift_dropped_weight * DELIVERY_UNFULFILLED_PENALTY_PER_KG

        step_profit = delivered_revenue - late_penalty - unfulfilled_penalty

        # Normalize profit into [0, 1] as a fraction of the theoretical max
        # revenue (all current orders delivered on time, no penalties).
        #
        # Important design choice: we do NOT subtract a negative do-nothing
        # baseline here. That subtraction flattens the reward into the raw
        # "fraction delivered on time", so random stacking at ~65-70% on-time
        # scores ~0.68 and cannot be distinguished from a careful heuristic
        # at ~80%. With straight profit/optimal and a penalty coefficient
        # greater than revenue, profits below ~67% on-time land in negative
        # territory and clip to zero — exactly the collapse we want to see
        # for an uninformed policy.
        total_weight_now = sum(o["weight_kg"] for o in orders)
        step_optimal = total_weight_now * DELIVERY_UNIT_REVENUE_PER_KG
        if step_optimal <= 0:
            reward = 0.0
        else:
            reward = step_profit / step_optimal
            reward = max(0.0, min(1.0, reward))

        feedback = (
            f"Assigned +{newly_assigned} "
            f"(total {assigned_total}/{total}) | "
            f"On-time: {on_time_count}, Late: {late_count}, "
            f"Pending: {len(pending)} | "
            f"Revenue ${delivered_revenue:.0f}, "
            f"LatePen ${late_penalty:.0f}, "
            f"UnfulfilledPen ${unfulfilled_penalty:.0f} | "
            f"P&L ${step_profit:.0f} / ${step_optimal:.0f} max"
        )

        # Apply dynamic events at END so they're visible in the returned
        # observation. Step 1 end → urgents + traffic appear in obs_1 and
        # the agent responds in action_2. Step 2 end → driver breakdown
        # appears in obs_2 and the agent responds in action_3. Step 3 end
        # triggers no further events (step 4 is the final wrap-up step).
        if step == 1:
            self._apply_routing_event_step2()
        elif step == 2:
            self._apply_routing_event_step3()
        return reward, feedback

    def _apply_routing_event_step2(self):
        """2 new urgent express orders arrive mid-shift.

        Previously also applied a traffic delay that randomly doubled one
        store's distance — this retroactively turned already-on-time
        deliveries into late ones even though the agent made the right
        decision with the information it had at step 1, so the heuristic
        ceiling dropped and the grader rewarded clairvoyance. The urgent
        injection is the only step-2 event now.
        """
        s = self._env_state
        s["orders"].extend(s["_urgent_orders"])
        s["_events_applied"].append(
            "NEW ORDERS: 2 urgent express orders added with tight deadlines."
        )

    def _apply_routing_event_step3(self):
        """Driver reports vehicle issue — locked out from accepting NEW
        orders (cap → 0). Existing assigned orders remain on this driver
        and still count in the P&L (the driver can finish in-progress
        deliveries but can't pick up anything new). The agent must route
        any remaining pending orders to other drivers.
        """
        s = self._env_state
        did = s["_broken_driver"]
        for d in s["drivers"]:
            if d["driver_id"] == did:
                old_cap = d["vehicle_capacity_kg"]
                d["vehicle_capacity_kg"] = 0
                s["_events_applied"].append(
                    f"VEHICLE ISSUE: {did} breaks down — locked out "
                    f"(cap {old_cap}kg → 0kg). Existing deliveries proceed, "
                    f"but no new orders can be assigned to this driver."
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
                # Low starting inventory: the agent MUST procure to meet demand.
                # Previously this was randint(20, 150) which pre-filled shelves
                # to ~70% of forecast demand, causing do-nothing to score 0.45
                # on the hard task — that windfall is gone.
                qty = rng.randint(0, 40)
                inventory[cat] = qty
                total += qty
            warehouses.append({
                "warehouse_id": f"WH{i+1}", "inventory": inventory,
                "max_capacity": cap, "current_total": total,
            })

        # Stochastic supplier reliability (Phase B, inverse-correlated with price).
        #
        # Cheap suppliers are UNRELIABLE: each order has a hidden per-order
        # success probability. On failure the budget is consumed but zero
        # units are delivered — so blind "pick cheapest" strategies lose
        # money to failed orders. A frontier agent must reason about
        # expected cost/unit = price / reliability and hedge across
        # suppliers. Reliability is exposed in scenario_data so the agent
        # CAN see it — the challenge is not information asymmetry, it's
        # choosing the right optimization objective.
        prices_sorted = sorted(round(rng.uniform(2.0, 8.0), 2) for _ in range(4))
        reliabilities_sorted = [0.60, 0.74, 0.86, 0.96]  # ascending with price
        # Shuffle which supplier slot (S1..S4) gets which price rank so the
        # LLM can't memorize "always use S4" across seeds.
        slot_rank = list(range(4))
        rng.shuffle(slot_rank)

        suppliers = []
        for i, name in enumerate(SUPPLIER_NAMES):
            rank = slot_rank[i]
            suppliers.append({
                "supplier_id": f"S{i+1}", "name": name,
                "price_per_unit": prices_sorted[rank],
                "reliability": reliabilities_sorted[rank],
                "lead_time_days": rng.randint(1, 4),
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
            "_total_offline_orders": 0,
        }

        # Cache P&L bounds — a fixed baseline (do-nothing) and optimal (oracle)
        # profit computed once at reset, used by the per-step grader to
        # normalize reward into [0, 1]. See _compute_surge_bounds.
        baseline_profit, optimal_profit = self._compute_surge_bounds()
        self._env_state["_baseline_profit"] = baseline_profit
        self._env_state["_optimal_profit"] = optimal_profit

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
        successful_orders = 0
        failed_orders = 0       # budget spent but no delivery (reliability roll)
        successful_redistributions = 0

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

            # Budget is always consumed (the supplier charges on order
            # placement) even if some units fail to ship.
            s["budget_remaining"] -= cost
            step_cost += cost

            # Stochastic per-unit reliability — delivered quantity is a
            # Binomial(n=actual_qty, p=reliability) outcome approximated
            # with a Gaussian. Per-unit sampling keeps aggregate variance
            # low (std ~ sqrt(n*p*(1-p))) while still introducing a real
            # penalty for choosing an unreliable supplier: if you order
            # from r=0.6, you expect 40% of every order to be lost.
            reliability = sup.get("reliability", 1.0)
            mean_delivered = actual_qty * reliability
            std_delivered = math.sqrt(
                max(actual_qty * reliability * (1 - reliability), 0.0)
            )
            delivered = int(
                round(mean_delivered + std_delivered * self._rng.gauss(0, 1))
            )
            delivered = max(0, min(actual_qty, delivered))
            if delivered > 0:
                successful_orders += 1
                wh["inventory"][product] = wh["inventory"].get(product, 0) + delivered
                wh["current_total"] += delivered
            if delivered < actual_qty:
                failed_orders += 1
            s["procurement_log"].append({
                "step": step, "supplier": sid, "product": product,
                "quantity_ordered": actual_qty, "quantity_delivered": delivered,
                "cost": cost, "warehouse": dest,
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
                successful_redistributions += 1
                s["redistribution_log"].append({
                    "step": step, "from": from_wh, "to": to_wh,
                    "product": product, "quantity": actual,
                })

        # --- Unified P&L grader ---
        #
        # profit = revenue − procurement_cost − stockout_penalty
        #                 − storage_penalty − offline_penalty
        # reward = clip((profit − baseline_profit) / (optimal_profit − baseline_profit), 0, 1)
        #
        # where baseline_profit and optimal_profit were cached at reset().
        # Random policies burn the budget on unused/offline orders and
        # collapse toward the baseline; the oracle procures at the cheapest
        # active supplier and approaches the ceiling.
        total_demand = s["total_demand"]

        total_supply = {cat: 0 for cat in PRODUCT_CATEGORIES}
        for wh in s["warehouses"]:
            for cat in PRODUCT_CATEGORIES:
                total_supply[cat] += wh["inventory"].get(cat, 0)

        # Revenue: realize every unit that actually finds a buyer.
        revenue = sum(
            min(total_supply[c], total_demand[c]) * SURGE_UNIT_SALE_PRICE
            for c in PRODUCT_CATEGORIES
        )

        # Cumulative procurement cost from the budget ledger.
        procurement_cost_total = s["budget"] - s["budget_remaining"]

        # Stockout: every unit of unmet demand bleeds margin.
        stockout_penalty = sum(
            max(0, total_demand[c] - total_supply[c]) * SURGE_STOCKOUT_COST_PER_UNIT
            for c in PRODUCT_CATEGORIES
        )

        # Storage: overstock above 10% demand buffer pays holding cost.
        storage_penalty = sum(
            max(0, total_supply[c] - total_demand[c] * 1.1)
            * SURGE_STORAGE_COST_PER_EXCESS_UNIT
            for c in PRODUCT_CATEGORIES
        )

        # Offline orders: accumulate across steps, severe fine per attempt.
        s["_total_offline_orders"] = (
            s.get("_total_offline_orders", 0) + ordered_offline
        )
        offline_penalty = s["_total_offline_orders"] * SURGE_OFFLINE_ORDER_PENALTY

        step_profit = (
            revenue
            - procurement_cost_total
            - stockout_penalty
            - storage_penalty
            - offline_penalty
        )

        baseline = s.get("_baseline_profit", 0.0)
        optimal = s.get("_optimal_profit", 1.0)
        denom = optimal - baseline
        if denom <= 0:
            reward = 0.0
        else:
            reward = (step_profit - baseline) / denom
            reward = max(0.0, min(1.0, reward))

        feedback = (
            f"Orders: {successful_orders} delivered, {failed_orders} failed "
            f"(offline attempts: {ordered_offline}) | "
            f"Redist: {successful_redistributions} | "
            f"Budget: ${s['budget_remaining']:.0f}/${s['budget']:.0f} | "
            f"Revenue: ${revenue:.0f}, Cost: ${procurement_cost_total:.0f}, "
            f"Stockout: ${stockout_penalty:.0f}, Storage: ${storage_penalty:.0f}, "
            f"OfflineFines: ${offline_penalty:.0f} | "
            f"P&L ${step_profit:.0f} in [${baseline:.0f}, ${optimal:.0f}]"
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
