"""
Inference Script — Supply Chain Retail Environment (Multi-Step)
===============================================================
Runs all 3 tasks (shelf_restock, delivery_routing, demand_surge) with an LLM
agent. Each task has multiple steps with dynamic events requiring adaptive
decisions.

MANDATORY environment variables:
    HF_TOKEN       Your Hugging Face API key (no default — spec-required).
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.

Optional environment variables:
    BASE_URL          Base URL of a running OpenEnv server.
    HF_SPACE_URL      Alternative name for BASE_URL.
    LOCAL_IMAGE_NAME  Local Docker image to spin up if BASE_URL is unreachable.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI
from my_env import SupplyChainEnv, SupplyChainAction


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

DEFAULT_SPACE_URL = "https://blackeagle-my-env.hf.space"
BASE_URL = (
    os.getenv("BASE_URL")
    or os.getenv("HF_SPACE_URL")
    or os.getenv("OPENENV_BASE_URL")
    or DEFAULT_SPACE_URL
)

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = [
    {"name": "shelf_restock", "seed": 42, "max_steps": 3},
    {"name": "delivery_routing", "seed": 42, "max_steps": 4},
    {"name": "demand_surge", "seed": 42, "max_steps": 5},
]
BENCHMARK = "supply_chain_retail"
TEMPERATURE = 0.1
MAX_TOKENS = 600
LLM_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(bool(done)).lower()
    reward_val = float(reward) if reward else 0.0
    print(
        f"[STEP] step={step} action={action} reward={reward_val:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{float(r):.2f}" for r in rewards)
    score_val = float(score) if score else 0.0
    print(
        f"[END] success={str(bool(success)).lower()} steps={int(steps)} "
        f"score={score_val:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Task-specific system prompts — reflect the actual P&L graders, not the
# old ad-hoc weighted sums. Each includes a worked example so the model
# knows the exact JSON format expected.
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "shelf_restock": (
        "You are a supply chain analyst helping a store manager decide which "
        "products to restock before opening.\n\n"
        "GRADING: reward = fraction of your picks that match the optimal ranking.\n"
        "  urgency = (daily_sales_rate * unit_revenue) / max(current_stock, 0.1)\n"
        "  The optimal ranking picks the K products with HIGHEST urgency.\n\n"
        "STRATEGY:\n"
        "  1. Compute urgency for every NON-restocked product.\n"
        "  2. Sort descending. Pick the top-K where K = the number of slots "
        "requested in the prompt.\n"
        "  3. React to dynamic events — demand spikes and surprise deliveries "
        "change the urgency ranking.\n\n"
        "EXAMPLE RESPONSE (when asked to pick 2 products):\n"
        '{"restock_products": ["P003", "P007"]}\n\n'
        "Respond with ONLY the JSON object. No explanation, no markdown."
    ),
    "delivery_routing": (
        "You are a logistics dispatcher at a distribution center assigning "
        "deliveries to drivers (a VRP-TW problem).\n\n"
        "GRADING (realized P&L, not a weighted sum):\n"
        "  profit = on_time_revenue - late_penalty - unfulfilled_penalty\n"
        "  reward = clip(profit / max_possible_revenue, 0, 1)\n"
        "  - On-time delivery: +$2.50 per kg of cargo weight\n"
        "  - Late delivery (arrive > deadline): -$4.00 per kg penalty, NO revenue\n"
        "  - Unfulfilled (not assigned): -$15.00 per kg penalty\n"
        "  - travel_time = distance_from_DC / 30 km/h + 0.35h dwell per stop\n"
        "  - Each stop adds leg_time + 0.35h to that driver's route time\n"
        "  - Driver's shift limit is enforced: stops that exceed it are dropped\n\n"
        "STRATEGY:\n"
        "  1. Pack initial orders onto 2 of 3 drivers, leaving 1 driver EMPTY.\n"
        "  2. When urgent orders arrive (tight deadlines), assign them to the "
        "reserved empty driver — only an empty route can hit sub-1h deadlines.\n"
        "  3. A driver may break down (cap→0) mid-episode. Route remaining "
        "pending orders to surviving drivers.\n"
        "  4. NEVER leave orders unassigned — the $15/kg unfulfilled penalty "
        "dwarfs even the $4/kg late penalty.\n\n"
        "EXAMPLE RESPONSE:\n"
        '{"assignments": [{"order_id": "ORD001", "driver_id": "D1"}, '
        '{"order_id": "ORD002", "driver_id": "D1"}, '
        '{"order_id": "ORD003", "driver_id": "D2"}]}\n\n'
        "Respond with ONLY the JSON object. No explanation, no markdown."
    ),
    "demand_surge": (
        "You are a supply chain planner preparing for a demand surge during "
        "a festival (5-step procurement task).\n\n"
        "GRADING (realized P&L):\n"
        "  profit = sale_revenue - procurement_cost - stockout_penalty "
        "- storage_penalty - offline_order_fines\n"
        "  reward = clip((profit - baseline) / (optimal - baseline), 0, 1)\n"
        "  - Sale revenue: $10.00 per unit sold (min of supply, demand)\n"
        "  - Procurement cost: units_ordered * supplier_price (charged even if "
        "delivery fails)\n"
        "  - Stockout: $8.00 per unit of unmet demand\n"
        "  - Storage: $0.50 per unit above 110% of demand (overstock)\n"
        "  - Offline fine: $400.00 per order sent to an OFFLINE supplier\n\n"
        "CRITICAL — SUPPLIER RELIABILITY:\n"
        "  Each supplier has a reliability score (0.60 to 0.96). When you "
        "order N units, only ~N*reliability units actually arrive. Budget is "
        "charged for ALL N units regardless. Cheap suppliers are UNRELIABLE.\n"
        "  effective_cost_per_unit = price / reliability\n"
        "  Pick the supplier with lowest effective cost, NOT lowest price.\n\n"
        "STRATEGY:\n"
        "  1. Compute shortfall = demand - current_inventory per category.\n"
        "  2. Order shortfall/reliability units from the best effective-cost "
        "ACTIVE supplier to each warehouse.\n"
        "  3. NEVER order from OFFLINE suppliers ($400 fine per attempt).\n"
        "  4. Spread orders across steps to react to disruptions.\n"
        "  5. Use redistribution if one warehouse is overstocked.\n\n"
        "EXAMPLE RESPONSE:\n"
        '{"procurement_orders": [{"supplier_id": "S3", "product": "rice", '
        '"quantity": 120, "destination_warehouse": "WH1"}], '
        '"redistribution": []}\n\n'
        "Respond with ONLY the JSON object. No explanation, no markdown."
    ),
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if not isinstance(text, str) or not text.strip():
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def call_llm(
    client: OpenAI,
    system_prompt: str,
    messages: List[Dict[str, str]],
) -> str:
    """Call the LLM with conversation history. Retries once on timeout."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=full_messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                timeout=LLM_TIMEOUT_S,
            )
            content = completion.choices[0].message.content if completion.choices else ""
            return (content or "").strip()
        except Exception as exc:
            if attempt == 0:
                print(f"[DEBUG] LLM call failed (retry 1): {exc}", file=sys.stderr, flush=True)
            else:
                print(f"[DEBUG] LLM call failed (giving up): {exc}", file=sys.stderr, flush=True)
                raise
    return ""


# ---------------------------------------------------------------------------
# Environment connection
# ---------------------------------------------------------------------------


async def make_env() -> Any:
    """Connect to the OpenEnv server. Raises on failure."""
    if BASE_URL:
        print(f"[DEBUG] Connecting to OpenEnv server at {BASE_URL}", flush=True)
        env = SupplyChainEnv(base_url=BASE_URL)
        await env.connect()
        return env

    if LOCAL_IMAGE_NAME:
        print(f"[DEBUG] Starting local docker: {LOCAL_IMAGE_NAME}", flush=True)
        return await SupplyChainEnv.from_docker_image(LOCAL_IMAGE_NAME)

    raise RuntimeError("No BASE_URL or LOCAL_IMAGE_NAME configured")


# ---------------------------------------------------------------------------
# Task runner (multi-step)
# ---------------------------------------------------------------------------


async def run_task(
    env: Any,
    client: OpenAI,
    task_name: str,
    seed: int,
    max_steps: int,
) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name, seed=seed)
        scenario_text = result.observation.scenario_text
        done = bool(result.observation.done)

        system_prompt = SYSTEM_PROMPTS.get(task_name, "")
        conversation: List[Dict[str, str]] = []

        step_num = 0
        while not done and step_num < max_steps:
            step_num += 1

            if step_num == 1:
                user_msg = f"Here is the initial scenario:\n\n{scenario_text}"
            else:
                user_msg = (
                    f"Step {step_num} — the situation has updated:\n\n{scenario_text}"
                )
            conversation.append({"role": "user", "content": user_msg})

            raw_response = call_llm(client, system_prompt, conversation)
            decision = parse_json_from_text(raw_response)
            if not isinstance(decision, dict):
                decision = {}

            conversation.append({"role": "assistant", "content": raw_response or ""})

            # Emit the raw action JSON in logs for transparency
            action_json = json.dumps(decision, separators=(",", ":"))

            action = SupplyChainAction(decision=decision)
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            done = bool(result.observation.done)
            rewards.append(reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=action_json,
                reward=reward,
                done=done,
                error=None,
            )

            if not done:
                scenario_text = result.observation.scenario_text
                feedback = result.observation.feedback
                if feedback:
                    conversation.append(
                        {
                            "role": "user",
                            "content": f"Feedback from your last decision: {feedback}",
                        }
                    )

        score = rewards[-1] if rewards else 0.0
        success = score >= 0.1

    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(f"[DEBUG] Task {task_name} error: {exc}", file=sys.stderr, flush=True)
        if not rewards:
            rewards = [0.0]
        steps_taken = max(steps_taken, 1)
        score = 0.0
        success = False
        log_step(
            step=steps_taken,
            action="error",
            reward=0.0,
            done=True,
            error=str(exc),
        )

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM inference on supply chain env")
    parser.add_argument("--model", default=None, help="Override MODEL_NAME")
    args = parser.parse_args()

    global MODEL_NAME
    if args.model:
        MODEL_NAME = args.model

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr, flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = await make_env()

    try:
        for task_config in TASKS:
            await run_task(
                env,
                client,
                task_config["name"],
                task_config["seed"],
                task_config["max_steps"],
            )
    finally:
        try:
            await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
