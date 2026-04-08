"""
Evaluation Script — Compare Base vs Fine-Tuned Model
=====================================================

Runs both models through the supply chain environment and produces
a comparison table + bar chart.

Usage:
    # Ensure env server is running: docker run -d --name sc_train -p 8001:8000 my_env-env:latest
    PYTORCH_ENABLE_MPS_FALLBACK=1 python eval_compare.py
"""

import json
import os
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from my_env import SupplyChainEnv, SupplyChainAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_URL = os.getenv("ENV_URL", "http://localhost:8001")
BASE_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
ADAPTER_PATH = "./checkpoints/supply-chain-grpo/final"

TASKS = ["shelf_restock", "delivery_routing", "demand_surge"]
# 15 seeds for tighter confidence intervals (was 5).
SEEDS = [42, 123, 456, 789, 1000, 7, 31, 65, 200, 555, 888, 1234, 1729, 2024, 9999]
MAX_STEPS = {"shelf_restock": 3, "delivery_routing": 4, "demand_surge": 5}

SYSTEM_PROMPT = (
    "You are a supply chain analyst. Analyze the scenario and respond "
    "with ONLY a JSON decision. No explanation, no markdown.\n"
    "- For shelf_restock: {\"restock_products\": [\"P001\", \"P002\"]}\n"
    "- For delivery_routing: {\"assignments\": [{\"order_id\": \"ORD001\", \"driver_id\": \"D1\"}, ...]}\n"
    "- For demand_surge: {\"procurement_orders\": [...], \"redistribution\": [...]}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_name, adapter_path=None):
    """Load model (optionally with LoRA adapter)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_decision(model, tokenizer, scenario_text, task_name):
    """Generate a JSON decision from the model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task_name}\n\n{scenario_text}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse JSON from response
    import re
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def run_episode(model, tokenizer, env, task_name, seed):
    """Run a full multi-step episode and return the final score."""
    result = env.reset(task_name=task_name, seed=seed)
    scenario_text = result.observation.scenario_text
    done = result.observation.done
    max_steps = MAX_STEPS[task_name]
    final_reward = 0.0

    for step in range(max_steps):
        if done:
            break
        decision = generate_decision(model, tokenizer, scenario_text, task_name)
        action = SupplyChainAction(decision=decision)
        result = env.step(action)
        final_reward = result.reward or 0.0
        done = result.observation.done
        if not done:
            scenario_text = result.observation.scenario_text

    return final_reward


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    results = {}

    for label, adapter in [("base", None), ("fine-tuned", ADAPTER_PATH)]:
        if label == "fine-tuned" and not os.path.exists(ADAPTER_PATH):
            print(f"Skipping fine-tuned model — adapter not found at {ADAPTER_PATH}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {label} model")
        print(f"{'='*60}")

        model, tokenizer = load_model(BASE_MODEL, adapter)
        env = SupplyChainEnv(base_url=ENV_URL).sync()
        results[label] = {}

        for task in TASKS:
            scores = []
            for seed in SEEDS:
                score = run_episode(model, tokenizer, env, task, seed)
                scores.append(score)
                print(f"  {task} seed={seed}: {score:.3f}")

            avg = sum(scores) / len(scores)
            results[label][task] = {"scores": scores, "avg": avg}
            print(f"  {task} AVG: {avg:.3f}")

        # Free memory
        del model, tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 60)

    for task in TASKS:
        base_avg = results.get("base", {}).get(task, {}).get("avg", 0.0)
        ft_avg = results.get("fine-tuned", {}).get(task, {}).get("avg", 0.0)
        delta = ft_avg - base_avg
        sign = "+" if delta >= 0 else ""
        print(f"{task:<25} {base_avg:>10.3f} {ft_avg:>12.3f} {sign}{delta:>9.3f}")

    # Save bar chart
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs("results", exist_ok=True)

        x = np.arange(len(TASKS))
        width = 0.35

        base_avgs = [results.get("base", {}).get(t, {}).get("avg", 0) for t in TASKS]
        ft_avgs = [results.get("fine-tuned", {}).get(t, {}).get("avg", 0) for t in TASKS]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, base_avgs, width, label="Base Model", color="#4C72B0")
        bars2 = ax.bar(x + width / 2, ft_avgs, width, label="Fine-tuned (GRPO)", color="#55A868")

        ax.set_ylabel("Average Score")
        ax.set_title("Base vs GRPO Fine-Tuned: Supply Chain Tasks")
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", " ").title() for t in TASKS])
        ax.legend()
        ax.set_ylim(0, 1.0)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        chart_path = "results/comparison.png"
        plt.savefig(chart_path, dpi=150)
        print(f"\nChart saved to {chart_path}")
    except ImportError:
        print("\nmatplotlib not installed — skipping chart generation")

    # Save raw results
    os.makedirs("results", exist_ok=True)
    with open("results/scores.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Raw scores saved to results/scores.json")


if __name__ == "__main__":
    main()
