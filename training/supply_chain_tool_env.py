"""
TRL-compatible environment wrapper for Supply Chain Retail Environment.

Wraps the OpenEnv environment as a class with typed tool methods,
compatible with TRL's GRPOTrainer `environment_factory` argument.

Ref: https://huggingface.co/docs/trl/openenv
"""

import os
import random
import sys

# Add parent dir so we can import the client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_env import SupplyChainEnv, SupplyChainAction

ENV_URL = os.getenv("ENV_URL", "http://localhost:8001")


class ShelfRestockToolEnv:
    """
    TRL environment wrapper for the shelf_restock task (3 steps).

    The model calls `select_products()` as a tool at each step.
    TRL's GRPOTrainer discovers this method automatically and handles
    the multi-turn tool-calling loop.
    """

    def __init__(self):
        self._async_client = SupplyChainEnv(base_url=ENV_URL)
        self.client = self._async_client.sync()
        self.reward = 0.0
        self.done = False
        self._scenario = ""

    def reset(self, **kwargs) -> str | None:
        """Reset the environment for a new episode."""
        self.reward = 0.0
        self.done = False
        seed = random.randint(0, 100_000)
        result = self.client.reset(task_name="shelf_restock", seed=seed)
        self._scenario = result.observation.scenario_text
        return self._scenario

    def select_products(self, product_ids: str) -> str:
        """
        Select products to restock, ordered by urgency.

        Analyze the inventory table and pick the most urgent products.
        Products with low stock relative to their daily sales rate AND
        high revenue per unit should be prioritized.

        Args:
            product_ids: Comma-separated product IDs to restock, most urgent first (e.g. "P009,P005")

        Returns:
            Feedback on your selection and the updated scenario for the next step,
            or the final score if the episode is complete.
        """
        if self.done:
            raise ValueError("Episode is over.")

        ids = [p.strip() for p in product_ids.split(",") if p.strip()]
        decision = {"restock_products": ids}
        action = SupplyChainAction(decision=decision)

        result = self.client.step(action)
        self.reward = result.reward or 0.0
        self.done = result.observation.done

        if self.done:
            return (
                f"Episode complete. Final score: {self.reward:.3f}. "
                f"{result.observation.feedback}"
            )

        self._scenario = result.observation.scenario_text
        return (
            f"Feedback: {result.observation.feedback}\n\n"
            f"Updated scenario:\n{self._scenario}"
        )


class DeliveryRoutingToolEnv:
    """
    TRL environment wrapper for the delivery_routing task (4 steps).
    """

    def __init__(self):
        self._async_client = SupplyChainEnv(base_url=ENV_URL)
        self.client = self._async_client.sync()
        self.reward = 0.0
        self.done = False
        self._scenario = ""

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        self.done = False
        seed = random.randint(0, 100_000)
        result = self.client.reset(task_name="delivery_routing", seed=seed)
        self._scenario = result.observation.scenario_text
        return self._scenario

    def assign_orders(self, assignments: str) -> str:
        """
        Assign pending delivery orders to drivers.

        Consider each driver's remaining capacity, delivery deadlines
        (travel time = distance / 30 km/h), shift hours, and workload balance.

        Args:
            assignments: Semicolon-separated assignments in format "order_id:driver_id" (e.g. "ORD001:D1;ORD002:D2;ORD003:D1")

        Returns:
            Feedback on assignments and the updated scenario for the next step,
            or the final score if the episode is complete.
        """
        if self.done:
            raise ValueError("Episode is over.")

        parsed = []
        for pair in assignments.split(";"):
            pair = pair.strip()
            if ":" in pair:
                oid, did = pair.split(":", 1)
                parsed.append({"order_id": oid.strip(), "driver_id": did.strip()})

        decision = {"assignments": parsed}
        action = SupplyChainAction(decision=decision)

        result = self.client.step(action)
        self.reward = result.reward or 0.0
        self.done = result.observation.done

        if self.done:
            return (
                f"Episode complete. Final score: {self.reward:.3f}. "
                f"{result.observation.feedback}"
            )

        self._scenario = result.observation.scenario_text
        return (
            f"Feedback: {result.observation.feedback}\n\n"
            f"Updated scenario:\n{self._scenario}"
        )


class DemandSurgeToolEnv:
    """
    TRL environment wrapper for the demand_surge task (5 steps).
    """

    def __init__(self):
        self._async_client = SupplyChainEnv(base_url=ENV_URL)
        self.client = self._async_client.sync()
        self.reward = 0.0
        self.done = False
        self._scenario = ""

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        self.done = False
        seed = random.randint(0, 100_000)
        result = self.client.reset(task_name="demand_surge", seed=seed)
        self._scenario = result.observation.scenario_text
        return self._scenario

    def plan_procurement(self, orders: str) -> str:
        """
        Place procurement orders with suppliers.

        Order inventory from ACTIVE suppliers only (never from OFFLINE ones).
        Stay within budget. Target demand gaps (needed minus available).
        Avoid overstocking beyond 120% of demand.

        Args:
            orders: Semicolon-separated orders in format "supplier_id,product,quantity,warehouse" (e.g. "S1,rice,100,WH1;S3,flour,200,WH2")

        Returns:
            Feedback on procurement and the updated scenario,
            or the final score if the episode is complete.
        """
        if self.done:
            raise ValueError("Episode is over.")

        procurement = []
        for entry in orders.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            parts = [p.strip() for p in entry.split(",")]
            if len(parts) >= 4:
                procurement.append({
                    "supplier_id": parts[0],
                    "product": parts[1],
                    "quantity": int(parts[2]) if parts[2].isdigit() else 0,
                    "destination_warehouse": parts[3],
                })

        decision = {
            "procurement_orders": procurement,
            "redistribution": [],
        }
        action = SupplyChainAction(decision=decision)

        result = self.client.step(action)
        self.reward = result.reward or 0.0
        self.done = result.observation.done

        if self.done:
            return (
                f"Episode complete. Final score: {self.reward:.3f}. "
                f"{result.observation.feedback}"
            )

        self._scenario = result.observation.scenario_text
        return (
            f"Feedback: {result.observation.feedback}\n\n"
            f"Updated scenario:\n{self._scenario}"
        )

    def redistribute_inventory(self, moves: str) -> str:
        """
        Redistribute inventory between warehouses to balance stock.

        Move products from warehouses with surplus to those with shortages.

        Args:
            moves: Semicolon-separated moves in format "from_wh,to_wh,product,quantity" (e.g. "WH1,WH3,flour,50;WH2,WH1,rice,100")

        Returns:
            Feedback on redistribution and the updated scenario,
            or the final score if the episode is complete.
        """
        if self.done:
            raise ValueError("Episode is over.")

        redistribution = []
        for entry in moves.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            parts = [p.strip() for p in entry.split(",")]
            if len(parts) >= 4:
                redistribution.append({
                    "from_warehouse": parts[0],
                    "to_warehouse": parts[1],
                    "product": parts[2],
                    "quantity": int(parts[3]) if parts[3].isdigit() else 0,
                })

        decision = {
            "procurement_orders": [],
            "redistribution": redistribution,
        }
        action = SupplyChainAction(decision=decision)

        result = self.client.step(action)
        self.reward = result.reward or 0.0
        self.done = result.observation.done

        if self.done:
            return (
                f"Episode complete. Final score: {self.reward:.3f}. "
                f"{result.observation.feedback}"
            )

        self._scenario = result.observation.scenario_text
        return (
            f"Feedback: {result.observation.feedback}\n\n"
            f"Updated scenario:\n{self._scenario}"
        )
