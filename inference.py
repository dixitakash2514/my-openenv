"""
Inference Script — Supply Chain Retail Environment (Multi-Step)
===============================================================
Runs all 3 tasks (shelf_restock, delivery_routing, demand_surge) with an LLM agent.
Each task has multiple steps with dynamic events requiring adaptive decisions.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Optional environment variables:
    BASE_URL          Base URL of a running OpenEnv server (defaults to the HF Space).
    HF_SPACE_URL      Alternative name for BASE_URL.
    LOCAL_IMAGE_NAME  Local Docker image to spin up if BASE_URL is unreachable.

The script connects to the live HF Space by default (no local docker required),
which is the same endpoint Phase 1 validation pings. Falls back gracefully and
ALWAYS emits [START]/[STEP]/[END] logs and exits with status 0 so the validator
can score what it gets.
"""

import asyncio
import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Hardened imports — never let an import error kill the process before we
# emit any logs the validator can score.
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI  # type: ignore
except Exception as _exc:  # pragma: no cover
    print(f"[DEBUG] openai import failed: {_exc}", flush=True)
    OpenAI = None  # type: ignore

try:
    from my_env import SupplyChainEnv, SupplyChainAction  # noqa: F401
except Exception as _exc:
    # Last-resort: try importing the local files directly as a package by
    # adding the script's parent directory to sys.path under the name "my_env".
    SupplyChainEnv = None  # type: ignore
    SupplyChainAction = None  # type: ignore
    try:
        import importlib.util

        _here = os.path.dirname(os.path.abspath(__file__))
        _init = os.path.join(_here, "__init__.py")
        if os.path.isfile(_init):
            spec = importlib.util.spec_from_file_location(
                "my_env", _init, submodule_search_locations=[_here]
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["my_env"] = module
                spec.loader.exec_module(module)
                SupplyChainEnv = getattr(module, "SupplyChainEnv", None)
                SupplyChainAction = getattr(module, "SupplyChainAction", None)
    except Exception as _exc2:
        print(
            f"[DEBUG] my_env import failed (primary={_exc}, fallback={_exc2})",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Live HF Space URL — same endpoint Phase 1 validation pings.
DEFAULT_SPACE_URL = "https://blackeagle-my-env.hf.space"
BASE_URL = (
    os.getenv("BASE_URL")
    or os.getenv("HF_SPACE_URL")
    or os.getenv("OPENENV_BASE_URL")
    or DEFAULT_SPACE_URL
)

# Optional fallback if you want to spin up a local container.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = [
    {"name": "shelf_restock", "seed": 42, "max_steps": 3},
    {"name": "delivery_routing", "seed": 42, "max_steps": 4},
    {"name": "demand_surge", "seed": 42, "max_steps": 5},
]
BENCHMARK = "supply_chain_retail"
TEMPERATURE = 0.2
MAX_TOKENS = 800
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
    try:
        reward_val = float(reward)
    except Exception:
        reward_val = 0.0
    print(
        f"[STEP] step={step} action={action} reward={reward_val:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    safe_rewards: List[float] = []
    for r in rewards:
        try:
            safe_rewards.append(float(r))
        except Exception:
            safe_rewards.append(0.0)
    rewards_str = ",".join(f"{r:.2f}" for r in safe_rewards)
    try:
        score_val = float(score)
    except Exception:
        score_val = 0.0
    print(
        f"[END] success={str(bool(success)).lower()} steps={int(steps)} "
        f"score={score_val:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Task-specific system prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "shelf_restock": (
        "You are a supply chain analyst helping a store manager decide which products to restock.\n"
        "You will make decisions across multiple steps. Each step tells you how many products to select.\n\n"
        "SCORING FORMULA (the environment grades against this exact rule):\n"
        "  urgency = (daily_sales_rate * unit_revenue) / max(current_stock, 0.1)\n"
        "  Pick the products with the HIGHEST urgency value.\n"
        "  reward = 0.6 * (your_picks ∩ optimal_picks)/slots + 0.4 * feasibility\n\n"
        "Steps:\n"
        "  1. Compute urgency for every NON-restocked product in the table.\n"
        "  2. Sort descending. Pick the top-K where K = the slot count requested in the prompt.\n"
        "  3. Pay attention to dynamic events (demand spikes, surprise deliveries) — they change urgency.\n"
        "  4. Products already restocked are NOT in the list anymore.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"restock_products": ["P001", "P002"]}\n\n'
        "Select EXACTLY the number of products requested. No explanation, no markdown, just the JSON."
    ),
    "delivery_routing": (
        "You are a logistics dispatcher at a distribution center.\n"
        "You will assign delivery orders to drivers across multiple steps.\n\n"
        "SCORING FORMULA (the environment grades against this exact rule):\n"
        "  reward = 0.30*on_time + 0.25*capacity + 0.25*coverage + 0.20*balance\n"
        "    on_time   = orders delivered before deadline / orders assigned\n"
        "    travel_time = distance_from_DC / 30 km/h. Must be <= deadline_hours.\n"
        "    capacity  = 1 - (drivers_over_capacity / total_drivers)\n"
        "    coverage  = orders_assigned / total_orders\n"
        "    balance   = how evenly orders are distributed across drivers\n\n"
        "Steps:\n"
        "  1. For every PENDING order, check which drivers still have remaining capacity_kg.\n"
        "  2. Pick the driver with the most remaining capacity that can fit the order.\n"
        "  3. Assign tightest-deadline orders first.\n"
        "  4. Already-assigned orders are NOT in the pending list.\n"
        "  5. React to mid-episode events (traffic delays, vehicle breakdowns).\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"assignments": [{"order_id": "ORD001", "driver_id": "D1"}, ...]}\n\n'
        "Assign EVERY pending order if capacity allows. No explanation, no markdown, just the JSON."
    ),
    "demand_surge": (
        "You are a supply chain planner preparing for a demand surge during a festival.\n"
        "You will make procurement and redistribution decisions across 5 steps.\n\n"
        "SCORING FORMULA (the environment grades against this exact rule):\n"
        "  reward = 0.30*fulfillment + 0.20*budget + 0.20*disruption + 0.15*balance + 0.15*waste\n"
        "    fulfillment = supply / demand per category, averaged\n"
        "    budget      = full credit only if you spent something AND stayed in budget\n"
        "    disruption  = full credit only if you ordered AND didn't order from OFFLINE suppliers\n"
        "    balance     = how evenly inventory is distributed across warehouses\n"
        "    waste       = penalised if supply > 120% of demand (overstock)\n"
        "  IMPORTANT: doing nothing gives a HEAVY penalty on budget, disruption, AND waste.\n\n"
        "Steps:\n"
        "  1. Compute gap[cat] = max(0, total_demand[cat] - total_available[cat]) per category.\n"
        "  2. Order roughly 60%% of each gap from the CHEAPEST ACTIVE supplier you can afford.\n"
        "  3. Spread orders across the 5 steps so you don't overstock in step 1.\n"
        "  4. NEVER order from a supplier marked OFFLINE.\n"
        "  5. Use redistribution moves to balance warehouses if one is full and another is empty.\n"
        "  6. React to events: supplier outages, demand shifts, warehouse capacity cuts.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"procurement_orders": [{"supplier_id": "S1", "product": "rice", '
        '"quantity": 100, "destination_warehouse": "WH1"}, ...],\n'
        ' "redistribution": [{"from_warehouse": "WH1", "to_warehouse": "WH3", '
        '"product": "flour", "quantity": 50}, ...]}\n\n'
        "Always include at least one procurement order unless every gap is already 0. "
        "No explanation, no markdown, just the JSON."
    ),
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if not isinstance(text, str):
        return {}
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
    except Exception:
        pass

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return {}


def call_llm(
    client: Optional[Any],
    system_prompt: str,
    messages: List[Dict[str, str]],
) -> str:
    """Call the LLM with conversation history and return raw text response."""
    if client is None:
        return "{}"
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
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
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "{}"


def summarize_action(task_name: str, decision: Dict[str, Any]) -> str:
    """Create a short string summary of the action for [STEP] log."""
    if not isinstance(decision, dict):
        return "invalid"
    if task_name == "shelf_restock":
        picks = decision.get("restock_products", []) or []
        if not isinstance(picks, list):
            picks = []
        return f"restock:[{','.join(str(p) for p in picks[:4])}]"
    elif task_name == "delivery_routing":
        assigns = decision.get("assignments", []) or []
        if not isinstance(assigns, list):
            assigns = []
        return f"assign:{len(assigns)}_orders"
    elif task_name == "demand_surge":
        orders = decision.get("procurement_orders", []) or []
        redist = decision.get("redistribution", []) or []
        if not isinstance(orders, list):
            orders = []
        if not isinstance(redist, list):
            redist = []
        return f"procure:{len(orders)}_orders,redist:{len(redist)}_moves"
    return "unknown"


# ---------------------------------------------------------------------------
# Environment connection — try several strategies, never raise.
# ---------------------------------------------------------------------------


async def make_env() -> Optional[Any]:
    """Connect to the OpenEnv server. Returns None on total failure."""
    if SupplyChainEnv is None:
        print("[DEBUG] SupplyChainEnv not importable", flush=True)
        return None

    # 1) Connect directly to a running server (HF Space by default).
    if BASE_URL:
        try:
            print(f"[DEBUG] Connecting to OpenEnv server at {BASE_URL}", flush=True)
            env = SupplyChainEnv(base_url=BASE_URL)
            await env.connect()
            return env
        except Exception as exc:
            print(
                f"[DEBUG] Direct connect to {BASE_URL} failed: {exc}",
                flush=True,
            )

    # 2) Optional fallback: spin up a local docker image if explicitly provided.
    if LOCAL_IMAGE_NAME:
        try:
            print(
                f"[DEBUG] Falling back to local docker image {LOCAL_IMAGE_NAME}",
                flush=True,
            )
            return await SupplyChainEnv.from_docker_image(LOCAL_IMAGE_NAME)
        except Exception as exc:
            print(
                f"[DEBUG] from_docker_image({LOCAL_IMAGE_NAME}) failed: {exc}",
                flush=True,
            )

    return None


# ---------------------------------------------------------------------------
# Task runner (multi-step) — never raises.
# ---------------------------------------------------------------------------


async def run_task(
    env: Optional[Any],
    client: Optional[Any],
    task_name: str,
    seed: int,
    max_steps: int,
) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    if env is None:
        log_step(step=1, action="error", reward=0.0, done=True, error="env_unavailable")
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        return

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

            try:
                action = SupplyChainAction(decision=decision)
                result = await env.step(action)
            except Exception as step_exc:
                print(f"[DEBUG] env.step() failed: {step_exc}", flush=True)
                log_step(
                    step=step_num,
                    action=summarize_action(task_name, decision),
                    reward=0.0,
                    done=True,
                    error=f"step_failed:{step_exc}",
                )
                rewards.append(0.0)
                steps_taken = step_num
                break

            reward = float(result.reward or 0.0)
            done = bool(result.observation.done)
            rewards.append(reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=summarize_action(task_name, decision),
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
        traceback.print_exc()
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
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
# Main — never raises, always exits 0.
# ---------------------------------------------------------------------------


async def main() -> None:
    client = None
    if OpenAI is not None:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception as exc:
            print(f"[DEBUG] OpenAI client init failed: {exc}", flush=True)
            client = None

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
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        # Last-resort safety net: never let inference.py exit non-zero.
        print(f"[DEBUG] Top-level error: {exc}", flush=True)
        traceback.print_exc()
    sys.exit(0)
