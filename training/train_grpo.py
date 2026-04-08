"""
GRPO Training Script — Supply Chain Retail Environment
======================================================

Trains Qwen2.5-1.5B-Instruct with GRPO on the shelf_restock task using
TRL's GRPOTrainer + OpenEnv environment_factory integration.

Usage:
    # 1. Start env server:  docker run -d --name sc_train -p 8001:8000 my_env-env:latest
    # 2. Start TensorBoard: tensorboard --logdir ./runs &
    # 3. Run training:      PYTORCH_ENABLE_MPS_FALLBACK=1 python train_grpo.py
    # 4. Monitor:           http://localhost:6006
"""

import os

# MPS fallback for unsupported PyTorch ops on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from supply_chain_tool_env import ShelfRestockToolEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
OUTPUT_DIR = "./checkpoints/supply-chain-grpo"
LOG_DIR = "./runs/supply-chain-grpo"
NUM_EPISODES = 200  # Number of training episodes
NUM_EPOCHS = 3

# ---------------------------------------------------------------------------
# Training prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a supply chain analyst helping a store manager decide which "
    "products to restock before the store opens. You will make decisions "
    "across multiple rounds.\n\n"
    "At each step, analyze the inventory table and select the most urgent "
    "products to restock using the `select_products` tool. Prioritize:\n"
    "- Products with LOW stock relative to daily sales rate (days of supply)\n"
    "- Products with HIGH revenue per unit\n"
    "- React to any dynamic events (demand spikes, surprise deliveries)\n\n"
    "Select exactly the number of products specified in the scenario."
)

USER_PROMPT = (
    "The store opens soon and you need to decide which products to restock. "
    "Use the select_products tool for each round. "
    "The scenario will update after each decision with new information."
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# All episodes use the same prompt; the env wrapper randomizes seeds in reset()
dataset = Dataset.from_dict({
    "prompt": [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
    ] * NUM_EPISODES
})

# ---------------------------------------------------------------------------
# LoRA config — parameter-efficient fine-tuning
# ---------------------------------------------------------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def reward_func(environments, **kwargs) -> list[float]:
    """Extract final reward from each environment instance."""
    return [env.reward for env in environments]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_completion_length=1024,
        learning_rate=5e-6,
        warmup_steps=20,
        max_grad_norm=1.0,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        report_to="tensorboard",
        logging_dir=LOG_DIR,
        bf16=True,
        gradient_checkpointing=True,
        chat_template_kwargs={"enable_thinking": False},
        # No vLLM — use HF generate() on MPS
    )

    print(f"Starting GRPO training with {MODEL_NAME}")
    print(f"Episodes: {NUM_EPISODES} | Epochs: {NUM_EPOCHS}")
    print(f"Generations per prompt: {training_args.num_generations}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"TensorBoard: {LOG_DIR}")
    print()

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        train_dataset=dataset,
        peft_config=peft_config,
        reward_funcs=reward_func,
        args=training_args,
        environment_factory=ShelfRestockToolEnv,
    )

    trainer.train()

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    print(f"\nTraining complete. Model saved to {final_path}")
    print("Run eval_compare.py to compare base vs fine-tuned model.")


if __name__ == "__main__":
    main()
