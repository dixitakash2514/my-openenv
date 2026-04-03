"""
Inference Script — Supply Chain Retail Environment
===================================================
Runs all 3 tasks (shelf_restock, delivery_routing, demand_surge) with an LLM agent.

MANDATORY variables:
    API_BASE_URL, MODEL_NAME, HF_TOKEN, LOCAL_IMAGE_NAME
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from my_env import SupplyChainEnv, SupplyChainAction, SupplyChainObservation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "my_env-env:latest")

TASKS = [
    {"name": "shelf_restock", "seed": 42},
    {"name": "delivery_routing", "seed": 42},
    {"name": "demand_surge", "seed": 42},
]
BENCHMARK = "supply_chain_retail"
TEMPERATURE = 0.2
MAX_TOKENS = 2000


# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Task-specific system prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "shelf_restock": (
        "You are a supply chain analyst helping a store manager decide which products to restock.\n"
        "Analyze the inventory data and select the most urgent products to restock.\n"
        "Consider: products with low stock relative to daily sales rate and high revenue per unit are most urgent.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"restock_products": ["P001", "P002", "P003", "P004"]}\n\n'
        "Select exactly the number of products specified. Order them by priority (most urgent first).\n"
        "No explanation, no markdown, just the JSON."
    ),
    "delivery_routing": (
        "You are a logistics dispatcher at a distribution center.\n"
        "Assign each delivery order to a driver considering:\n"
        "- Vehicle capacity (total weight per driver must not exceed capacity)\n"
        "- Delivery deadlines (orders must arrive within their time window)\n"
        "- Driver shift hours remaining\n"
        "- Balance workload across drivers\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"assignments": [{"order_id": "ORD001", "driver_id": "D1"}, {"order_id": "ORD002", "driver_id": "D2"}, ...]}\n\n'
        "Assign ALL orders. No explanation, no markdown, just the JSON."
    ),
    "demand_surge": (
        "You are a supply chain planner preparing for a demand surge during a festival.\n"
        "Create a procurement and redistribution plan considering:\n"
        "- Do NOT order from any supplier marked OFFLINE\n"
        "- Stay within the given budget\n"
        "- Maximize demand fulfillment across all product categories\n"
        "- Avoid overstocking (don't order more than ~120% of forecasted demand)\n"
        "- Balance inventory across warehouses\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"procurement_orders": [{"supplier_id": "S1", "product": "rice", "quantity": 100, "destination_warehouse": "WH1"}, ...],\n'
        ' "redistribution": [{"from_warehouse": "WH1", "to_warehouse": "WH3", "product": "flour", "quantity": 50}, ...]}\n\n'
        "If no redistribution is needed, use an empty list. No explanation, no markdown, just the JSON."
    ),
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call the LLM and return raw text response."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "{}"


def summarize_action(task_name: str, decision: Dict[str, Any]) -> str:
    """Create a short string summary of the action for [STEP] log."""
    if task_name == "shelf_restock":
        picks = decision.get("restock_products", [])
        return f"restock:[{','.join(str(p) for p in picks[:4])}]"
    elif task_name == "delivery_routing":
        assigns = decision.get("assignments", [])
        return f"assign:{len(assigns)}_orders"
    elif task_name == "demand_surge":
        orders = decision.get("procurement_orders", [])
        redist = decision.get("redistribution", [])
        return f"procure:{len(orders)}_orders,redist:{len(redist)}_moves"
    return "unknown"


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

async def run_task(
    env: SupplyChainEnv,
    client: OpenAI,
    task_name: str,
    seed: int,
) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name, seed=seed)
        scenario_text = result.observation.scenario_text

        system_prompt = SYSTEM_PROMPTS[task_name]
        user_prompt = f"Here is the scenario:\n\n{scenario_text}"

        raw_response = call_llm(client, system_prompt, user_prompt)
        decision = parse_json_from_text(raw_response)

        action = SupplyChainAction(decision=decision)
        result = await env.step(action)

        reward = result.reward or 0.0
        rewards.append(reward)
        steps_taken = 1
        score = reward
        success = score >= 0.1

        log_step(
            step=1,
            action=summarize_action(task_name, decision),
            reward=reward,
            done=True,
            error=None,
        )

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        rewards = [0.0]
        steps_taken = 1
        score = 0.0
        success = False
        log_step(step=1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = await SupplyChainEnv.from_docker_image(LOCAL_IMAGE_NAME)

    try:
        for task_config in TASKS:
            await run_task(env, client, task_config["name"], task_config["seed"])
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
