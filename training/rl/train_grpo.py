"""
GRPO fine-tuning for the Perceiver agent.

Usage
-----
# Debug run (test.jsonl, 20 samples, 1 epoch)
CUDA_VISIBLE_DEVICES=3 python -m training.rl.train_grpo \
    --data training/data/test.jsonl \
    --model Qwen/Qwen3-4B \
    --output training/checkpoints/perceiver-debug \
    --epochs 1 --batch_size 2 --num_generations 4

# Full training
CUDA_VISIBLE_DEVICES=3 python -m training.rl.train_grpo \
    --data training/data/train.jsonl \
    --model Qwen/Qwen3-4B \
    --output training/checkpoints/perceiver-grpo \
    --epochs 3 --batch_size 4 --num_generations 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def reward_func(prompts, completions, target_dimensions, tool_required, **kwargs):
    """
    Reward function compatible with TRL GRPOTrainer.

    Parameters
    ----------
    prompts : list — prompt strings or message lists (unused, context only)
    completions : list — model-generated completion texts
    target_dimensions : list[list[str]] — ground truth dims per sample
    tool_required : list[list[str]] — ground truth tool decisions per sample

    Returns
    -------
    list[float] — one reward per completion
    """
    from training.rl.reward import compute_reward

    rewards = []
    for completion, tgt_dims, tgt_tool in zip(
        completions, target_dimensions, tool_required
    ):
        # completion may be a list of message dicts from chat-based generation
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)
        r = compute_reward(text, tgt_dims, tgt_tool)
        rewards.append(r)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Perceiver")
    parser.add_argument("--data", type=str, required=True,
                        help="JSONL training data path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="Base model name or path")
    parser.add_argument("--output", type=str,
                        default="training/checkpoints/perceiver-grpo",
                        help="Output directory for LoRA adapter")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size (unique prompts per step)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # GRPO
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Completions per prompt for GRPO grouping")
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient")
    parser.add_argument("--temperature", type=float, default=0.7)

    # Misc
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    # ── Imports (heavy, only after arg parsing) ──────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from training.rl.data_loader import load_dataset

    # ── Validate GRPO constraint: batch_size must be divisible by num_generations
    if args.batch_size % args.num_generations != 0:
        adjusted = max(1, args.batch_size // args.num_generations)
        print(f"[WARN] batch_size ({args.batch_size}) not divisible by "
              f"num_generations ({args.num_generations}). "
              f"Reducing num_generations to {adjusted}.")
        args.num_generations = adjusted

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = load_dataset(args.data)
    print(f"Loaded {len(dataset)} samples from {args.data}")

    # ── Model & tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3 defaults to thinking mode (<think>…</think>) which fills the entire
    # completion budget before producing JSON.  Wrap apply_chat_template so that
    # every call (including TRL-internal ones) passes enable_thinking=False.
    _orig_act = tokenizer.apply_chat_template
    def _no_think_act(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return _orig_act(*args, **kwargs)
    tokenizer.apply_chat_template = _no_think_act

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    # ── GRPO config ──────────────────────────────────────────────────────────
    training_args = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        report_to="none",
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nStarting GRPO training")
    print(f"  Model       : {args.model}")
    print(f"  LoRA        : rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  Batch       : {args.batch_size} × {args.gradient_accumulation_steps} accum")
    print(f"  Generations : {args.num_generations} per prompt")
    print(f"  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")

    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nLoRA adapter saved to {args.output}")
    print(f"Deploy with vLLM:")
    print(f"  vllm serve {args.model} \\")
    print(f"    --enable-lora \\")
    print(f"    --lora-modules perceiver={args.output} \\")
    print(f"    --port 8000")


if __name__ == "__main__":
    main()
