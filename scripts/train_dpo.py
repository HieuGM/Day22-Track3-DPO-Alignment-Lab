#!/usr/bin/env python3
"""CLI wrapper for NB3 logic — trains a DPO adapter.

Usage:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --beta 0.05 --output-dir adapters/dpo-b0.05
    python scripts/train_dpo.py --beta 0.5  --output-dir adapters/dpo-b0.50

Mirrors `notebooks/03_dpo_train.py`. Used by `make beta-sweep` for the rigor
add-on +6 (β-sweep mini-experiment).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sft-path", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--pref-path", default=str(REPO / "data" / "pref" / "train.parquet"))
    parser.add_argument("--output-dir", default=str(REPO / "adapters" / "dpo"))
    args = parser.parse_args()

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    assert tier in {"LOWVRAM", "T4", "BIGGPU"}, f"Invalid COMPUTE_TIER: {tier}"
    if tier == "LOWVRAM":
        base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        max_len, max_prompt = 192, 96
        batch, grad_accum = 1, 4
    elif tier == "T4":
        base_model = "unsloth/Qwen2.5-3B-bnb-4bit"
        max_len, max_prompt = 512, 256
        batch, grad_accum = 1, 8
    else:
        base_model = "unsloth/Qwen2.5-7B-bnb-4bit"
        max_len, max_prompt = 1024, 512
        batch, grad_accum = 1, 4

    lr = args.lr if args.lr is not None else (5e-6 if tier == "LOWVRAM" else 5e-7)
    epochs = args.epochs if args.epochs is not None else (2 if tier == "LOWVRAM" else 1)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Tier:       {tier}")
    print(f"Base:       {base_model}")
    print(f"Beta / LR:  {args.beta} / {lr}")
    print(f"Epochs:     {epochs}")
    print(f"Output:     {output}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import DPOConfig, DPOTrainer

    if tier == "LOWVRAM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda")
    else:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model, max_seq_length=max_len, dtype=None, load_in_4bit=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, args.sft_path, is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    if tier == "LOWVRAM":
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                target_modules=target_modules,
            ),
        )
        model.enable_input_require_grads()
    else:
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=42, use_rslora=False, loftq_config=None,
        )

    config = DPOConfig(
        output_dir=str(output.parent / f"{output.name}-checkpoints"),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=args.beta,
        max_length=max_len,
        max_prompt_length=max_prompt,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=2 if tier == "LOWVRAM" else 10,
        save_strategy="no",
        optim="adamw_torch" if tier == "LOWVRAM" else "adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=tier == "LOWVRAM",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
        loss_type="sigmoid",
        report_to="none",
    )

    pref = Dataset.from_parquet(args.pref_path)
    trainer = DPOTrainer(
        model=model, ref_model=None, args=config,
        train_dataset=pref, processing_class=tokenizer,
    )
    started_at = time.perf_counter()
    train_result = trainer.train()
    training_seconds = time.perf_counter() - started_at

    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # Headline metrics
    import pandas as pd

    logs = pd.DataFrame(trainer.state.log_history)
    chosen_col = "rewards/chosen" if "rewards/chosen" in logs.columns else None
    rejected_col = "rewards/rejected" if "rewards/rejected" in logs.columns else None

    metrics = {
        "compute_tier": tier,
        "base_model": base_model,
        "beta": args.beta,
        "lr": lr,
        "epochs": epochs,
        "training_samples": len(pref),
        "max_length": max_len,
        "final_train_loss": float(train_result.training_loss),
        "end_chosen_reward": float(logs[chosen_col].iloc[-5:].mean()) if chosen_col else None,
        "end_rejected_reward": float(logs[rejected_col].iloc[-5:].mean()) if rejected_col else None,
        "training_seconds": training_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    if metrics["end_chosen_reward"] is not None and metrics["end_rejected_reward"] is not None:
        metrics["end_reward_gap"] = metrics["end_chosen_reward"] - metrics["end_rejected_reward"]

    (output / "dpo_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nFinal loss:     {train_result.training_loss:.4f}")
    if "end_reward_gap" in metrics:
        print(f"End reward gap: {metrics['end_reward_gap']:+.3f}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
