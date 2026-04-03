"""
Inference Script — Supply Chain Retail Environment (Multi-Step)
===============================================================
Runs all 3 tasks (shelf_restock, delivery_routing, demand_surge) with an LLM agent.
Each task has multiple steps with dynamic events requiring adaptive decisions.

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
    {"name": "shelf_restock", "seed": 42, "max_steps": 3},
    {"name": "delivery_routing", "seed": 42, "max_steps": 4},
    {"name": "demand_surge", "seed": 42, "max_steps": 5},
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
        "The store has limited time and you will make decisions across multiple steps.\n"
        "Each step tells you how many products to select. Analyze urgency based on:\n"
        "- Low stock relative to daily sales rate (days of stock remaining)\n"
        "- High revenue per unit\n"
        "- Any dynamic events (demand spikes, surprise deliveries) that change priorities\n\n"
        "IMPORTANT: Pay attention to events announced at each step — they change the data.\n"
        "Products already restocked are removed from the list.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"restock_products": ["P001", "P002"]}\n\n'
        "Select exactly the number of products specified in the prompt. Order by urgency.\n"
        "No explanation, no markdown, just the JSON."
    ),
    "delivery_routing": (
        "You are a logistics dispatcher at a distribution center.\n"
        "You will assign delivery orders to drivers across multiple steps.\n"
        "New orders, traffic delays, and vehicle issues may occur between steps.\n\n"
        "Consider at each step:\n"
        "- Vehicle capacity (remaining kg for each driver)\n"
        "- Delivery deadlines (travel time = distance / 30 km/h)\n"
        "- Driver shift hours remaining\n"
        "- Balance workload across drivers\n"
        "- Any new events (urgent orders, traffic, vehicle breakdowns)\n\n"
        "IMPORTANT: Only assign PENDING orders. Already-assigned orders are handled.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"assignments": [{"order_id": "ORD001", "driver_id": "D1"}, ...]}\n\n'
        "Assign all pending orders if possible. No explanation, no markdown, just the JSON."
    ),
    "demand_surge": (
        "You are a supply chain planner preparing for a demand surge during a festival.\n"
        "You will make procurement and redistribution decisions across 5 steps.\n"
        "The situation evolves: suppliers may go offline, demand forecasts change,\n"
        "and warehouse capacity may be reduced.\n\n"
        "At each step, consider:\n"
        "- Do NOT order from any supplier marked OFFLINE\n"
        "- Stay within the remaining budget\n"
        "- Fill demand gaps (needed vs available) across all product categories\n"
        "- Avoid overstocking (don't order more than ~120% of remaining gap)\n"
        "- Balance inventory across warehouses\n"
        "- React to events: adjust plans when disruptions occur\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"procurement_orders": [{"supplier_id": "S1", "product": "rice", '
        '"quantity": 100, "destination_warehouse": "WH1"}, ...],\n'
        ' "redistribution": [{"from_warehouse": "WH1", "to_warehouse": "WH3", '
        '"product": "flour", "quantity": 50}, ...]}\n\n'
        "If no action needed this step, use empty lists. No explanation, no markdown, just the JSON."
    ),
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
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


def call_llm(client: OpenAI, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    """Call the LLM with conversation history and return raw text response."""
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=full_messages,
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
# Task runner (multi-step)
# ---------------------------------------------------------------------------

async def run_task(
    env: SupplyChainEnv,
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
        # Reset — get initial scenario
        result = await env.reset(task_name=task_name, seed=seed)
        scenario_text = result.observation.scenario_text
        done = result.observation.done

        system_prompt = SYSTEM_PROMPTS[task_name]
        # Build conversation history for multi-turn context
        conversation: List[Dict[str, str]] = []

        step_num = 0
        while not done and step_num < max_steps:
            step_num += 1

            # Add current scenario as user message
            if step_num == 1:
                user_msg = f"Here is the initial scenario:\n\n{scenario_text}"
            else:
                user_msg = (
                    f"Step {step_num} — the situation has updated:\n\n{scenario_text}"
                )
            conversation.append({"role": "user", "content": user_msg})

            # Call LLM
            raw_response = call_llm(client, system_prompt, conversation)
            decision = parse_json_from_text(raw_response)

            # Add assistant response to history
            conversation.append({"role": "assistant", "content": raw_response})

            # Step the environment
            action = SupplyChainAction(decision=decision)
            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.observation.done
            rewards.append(reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=summarize_action(task_name, decision),
                reward=reward,
                done=done,
                error=None,
            )

            # Get next scenario for the following step
            if not done:
                scenario_text = result.observation.scenario_text
                # Add feedback as context for next decision
                feedback = result.observation.feedback
                if feedback:
                    conversation.append({
                        "role": "user",
                        "content": f"Feedback from your last decision: {feedback}",
                    })

        score = rewards[-1] if rewards else 0.0  # Final step reward is the episode score
        success = score >= 0.1

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        if not rewards:
            rewards = [0.0]
        steps_taken = max(steps_taken, 1)
        score = 0.0
        success = False
        log_step(step=steps_taken, action="error", reward=0.0, done=True, error=str(e))

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
            await run_task(
                env, client,
                task_config["name"],
                task_config["seed"],
                task_config["max_steps"],
            )
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
