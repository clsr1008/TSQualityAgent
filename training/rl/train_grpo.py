"""
GRPO fine-tuning for the Perceiver agent.

Usage
-----
# Debug run (test.jsonl, 20 samples, 1 epoch)
CUDA_VISIBLE_DEVICES=3 python -m training.rl.train_grpo \
    --data training/data/test.jsonl \
    --model Qwen/Qwen3-4B \
    --output training/checkpoints/perceiver-grpo-debug \
    --epochs 1 --batch_size 1 --num_generations 8

# Full training (hint-filtered dataset) — single GPU
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 python -m training.rl.train_grpo \
    --data training/data/perceiver_train_filtered.jsonl \
    --val_data training/data/perceiver_val.jsonl \
    --model Qwen/Qwen3-4B \
    --output training/checkpoints/perceiver-grpo-v1 \
    --epochs 1 \
    --batch_size 1 \
    --num_generations 8 \
    --gradient_accumulation_steps 4

# Full training — dual GPU (DDP, 4B fits per card; halve accum steps since global batch doubles)
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m training.rl.train_grpo \
        --data training/data/perceiver_train_filtered.jsonl \
        --val_data training/data/perceiver_val.jsonl \
        --model Qwen/Qwen3-4B \
        --output training/checkpoints/perceiver-grpo-v1 \
        --epochs 1 \
        --batch_size 1 \
        --num_generations 8 \
        --gradient_accumulation_steps 4

# Resume from existing LoRA (warm restart — loads adapter weights, resets optimizer)
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m training.rl.train_grpo \
        --data training/data/perceiver_train_filtered.jsonl \
        --val_data training/data/perceiver_val.jsonl \
        --model Qwen/Qwen3-4B \
        --output training/checkpoints/perceiver-grpo-v2 \
        --epochs 1 \
        --batch_size 1 \
        --num_generations 8 \
        --gradient_accumulation_steps 4 \
        --resume_from_lora training/checkpoints/perceiver-grpo-v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from training.rl.reward import grpo_reward_format, grpo_reward_dim


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Perceiver")
    parser.add_argument("--data", type=str, required=True,
                        help="JSONL training data path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="Base model name or path")
    parser.add_argument("--output", type=str,
                        default="training/checkpoints/perceiver-rl",
                        help="Output directory for LoRA adapter")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Unique prompts per micro-batch per GPU. "
                             "TRL per_device_train_batch_size = batch_size × num_generations. "
                             "With N GPUs, global effective batch = batch_size × num_generations × gradient_accumulation_steps × N.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # GRPO
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Completions per prompt for GRPO grouping")
    parser.add_argument("--max_completion_length", type=int, default=300)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient")
    parser.add_argument("--temperature", type=float, default=1.0)

    # Validation
    parser.add_argument("--val_data", type=str, default=None,
                        help="Validation JSONL path for periodic reward eval during training")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate on val set every N optimizer steps")
    parser.add_argument("--eval_n_samples", type=int, default=100,
                        help="Number of val samples to evaluate per eval step")

    # Resume
    parser.add_argument("--resume_from_lora", type=str, default=None,
                        help="Path to an existing LoRA adapter directory to resume training from. "
                             "Loads the adapter weights as the starting point; optimizer/scheduler "
                             "are reset (warm restart). Use when adapter weights exist but no "
                             "full HF trainer checkpoint is available.")

    # Misc
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    # ── Imports (heavy, only after arg parsing) ──────────────────────────────
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from peft import LoraConfig, PeftModel
    from trl import GRPOConfig, GRPOTrainer

    from training.rl.data_loader import load_dataset

    # batch_size = unique prompts per micro-batch (user-facing)
    # TRL expects per_device_train_batch_size = total completions = prompts × generations
    trl_batch_size = args.batch_size * args.num_generations

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = load_dataset(args.data)
    print(f"Loaded {len(dataset)} samples from {args.data}")

    val_dataset = None
    if args.val_data:
        val_dataset = load_dataset(args.val_data)
        if args.eval_n_samples and args.eval_n_samples < len(val_dataset):
            val_dataset = val_dataset.select(range(args.eval_n_samples))
        print(f"Loaded {len(val_dataset)} val samples from {args.val_data}")

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
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    if args.resume_from_lora:
        # Load existing LoRA adapter and continue training it.
        # peft_config must be None so GRPOTrainer doesn't wrap the model again.
        print(f"  Loading existing LoRA from {args.resume_from_lora} …")
        model = PeftModel.from_pretrained(model, args.resume_from_lora, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )

    # ── State-saver callback ──────────────────────────────────────────────────
    # Writes trainer_state.json to the output dir after every log step so that
    # training metrics are visible in real time without waiting for a checkpoint.
    state_path = Path(args.output) / "trainer_state.json"

    class StateSaverCallback(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero:
                return
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state.save_to_json(str(state_path))
            # Print eval results to stdout
            eval_items = {k: v for k, v in (logs or {}).items() if k.startswith("eval_")}
            if eval_items:
                print(f"[eval step {state.global_step}] " +
                      " | ".join(f"{k}={v:.4f}" for k, v in eval_items.items()))

    # ── GRPO config ──────────────────────────────────────────────────────────
    training_args = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=trl_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.epochs,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=args.eval_steps if val_dataset is not None else None,
        report_to="none",
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        reward_funcs=[grpo_reward_format, grpo_reward_dim],
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[StateSaverCallback()],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nStarting GRPO training")
    print(f"  Model       : {args.model}")
    if args.resume_from_lora:
        print(f"  LoRA        : resumed from {args.resume_from_lora}")
    else:
        print(f"  LoRA        : rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  Batch       : {args.batch_size} prompts × {args.num_generations} gen = {trl_batch_size} completions, ×{args.gradient_accumulation_steps} accum")
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
