#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


FALLBACK_QWEN_CHAT_TEMPLATE = """{% if messages and messages[0]['role'] == 'system' %}
{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif %}
{% for message in loop_messages %}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{% endif %}"""

RESPONSE_PART = "<|im_start|>assistant\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-prompt completion-only SFT for Qwen3.5-9B.")
    parser.add_argument("--base", default="unsloth/Qwen3.5-9B")
    parser.add_argument("--init-lora")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--loss-type", choices=("nll", "dft"), default="nll")
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--top-layers", "--lora-top-layers", dest="top_layers", type=int, default=8)
    parser.add_argument("--cuda-visible-devices", default="1")
    return parser.parse_args()


def _ensure_tokenizer_chat_template(tokenizer) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = FALLBACK_QWEN_CHAT_TEMPLATE


def _messages_to_chat_text(messages, tokenizer) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def _tokenize_completion_example(example, tokenizer):
    prompt = example.get("prompt") or []
    completion = example.get("completion") or []
    if not prompt or not completion:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prompt_text = _messages_to_chat_text(prompt, tokenizer)
    full_text = _messages_to_chat_text(prompt + completion, tokenizer)
    if not prompt_text or not full_text:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prefix_text = prompt_text + RESPONSE_PART
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_tokens = tokenizer(full_text, add_special_tokens=False)
    full_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    common_prefix = 0
    max_prefix = min(len(prefix_ids), len(full_ids))
    while common_prefix < max_prefix and prefix_ids[common_prefix] == full_ids[common_prefix]:
        common_prefix += 1

    if common_prefix <= 0 or common_prefix >= len(full_ids):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    labels = [-100] * common_prefix + full_ids[common_prefix:]
    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _override_system_prompt(example, system_prompt_text: str):
    prompt = list(example.get("prompt") or [])
    if not prompt:
        return example

    updated_prompt: list[dict[str, str]] = []
    replaced = False
    for idx, message in enumerate(prompt):
        if idx == 0 and message.get("role") == "system":
            updated_prompt.append({"role": "system", "content": system_prompt_text})
            replaced = True
        else:
            updated_prompt.append(message)

    if not replaced:
        updated_prompt.insert(0, {"role": "system", "content": system_prompt_text})

    return {
        **example,
        "prompt": updated_prompt,
    }


def _shift_logits_and_labels(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels


def main() -> int:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    import unsloth  # noqa: F401
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from unsloth import FastLanguageModel, is_bfloat16_supported

    class CompletionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.logits
            shift_logits, shift_labels = _shift_logits_and_labels(logits, labels)
            vocab_size = shift_logits.size(-1)

            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)
            valid_mask = shift_labels.ne(-100)

            if args.loss_type == "dft":
                safe_labels = shift_labels.masked_fill(~valid_mask, 0)
                token_probs = (
                    torch.softmax(shift_logits, dim=-1)
                    .gather(-1, safe_labels.unsqueeze(-1))
                    .squeeze(-1)
                    .detach()
                )
                weights = torch.where(valid_mask, token_probs, torch.zeros_like(token_probs))
                loss = (per_token_loss * weights).sum() / valid_mask.sum().clamp(min=1)
            else:
                loss = per_token_loss.masked_select(valid_mask).mean()

            if args.anchor_weight > 0.0:
                anchor_penalty = torch.zeros((), device=loss.device, dtype=torch.float32)
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    anchor_value = anchor_params.get(name)
                    if anchor_value is None:
                        continue
                    anchor_penalty = anchor_penalty + (param.float() - anchor_value).pow(2).mean()
                loss = loss + args.anchor_weight * anchor_penalty

            return (loss, outputs) if return_outputs else loss

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("/home/gavin/tuning/runs")
        / f"qwen35_9b_completionsft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    print("[SFT] loading base model", args.base)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        dtype=dtype,
        load_in_4bit=True,
        auto_model=AutoModelForCausalLM,
    )
    _ensure_tokenizer_chat_template(tokenizer)

    num_hidden_layers = int(model.config.num_hidden_layers)
    first_lora_layer = max(0, num_hidden_layers - args.top_layers)
    layers_to_transform = list(range(first_lora_layer, num_hidden_layers))

    if args.init_lora:
        print("[SFT] loading init adapter", args.init_lora)
        model = PeftModel.from_pretrained(model, str(Path(args.init_lora).resolve()), is_trainable=True)
        peft_cfg = model.peft_config["default"]
        layers = list(getattr(peft_cfg, "layers_to_transform", []) or [])
        if layers:
            layers_to_transform = layers
            first_lora_layer = min(layers)
        args.lora_r = int(getattr(peft_cfg, "r", args.lora_r))
        args.lora_alpha = int(getattr(peft_cfg, "lora_alpha", args.lora_alpha))
        args.lora_dropout = float(getattr(peft_cfg, "lora_dropout", args.lora_dropout))
        use_rslora = bool(getattr(peft_cfg, "use_rslora", False))
    else:
        print("[SFT] creating fresh adapter")
        use_rslora = True
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            layers_to_transform=layers_to_transform,
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=use_rslora,
        )
    model.config.use_cache = False

    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    anchor_params = {}
    if args.anchor_weight > 0.0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                anchor_params[name] = param.detach().clone().to(device=param.device, dtype=torch.float32)

    dataset = load_dataset("json", data_files=str(Path(args.dataset).resolve()), split="train")
    system_prompt_text = None
    if args.system_prompt_file:
        system_prompt_text = Path(args.system_prompt_file).resolve().read_text(encoding="utf-8").strip()
        dataset = dataset.map(lambda ex: _override_system_prompt(ex, system_prompt_text))
    dataset = dataset.map(
        lambda ex: _tokenize_completion_example(ex, tokenizer),
        remove_columns=dataset.column_names,
    )
    dataset = dataset.filter(lambda ex: len(ex["input_ids"]) > 1 and any(v != -100 for v in ex["labels"]))
    print("[SFT] dataset size", len(dataset))
    print(
        "[SFT] config",
        json.dumps(
            {
                "base": args.base,
                "init_lora": str(Path(args.init_lora).resolve()) if args.init_lora else None,
                "dataset": str(Path(args.dataset).resolve()),
                "dataset_size": len(dataset),
                "system_prompt_file": str(Path(args.system_prompt_file).resolve()) if args.system_prompt_file else None,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "max_seq_len": args.max_seq_len,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "loss_type": args.loss_type,
                "anchor_weight": args.anchor_weight,
                "lora": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "use_rslora": use_rslora,
                    "top_layers": args.top_layers,
                    "layer_range": [first_lora_layer, num_hidden_layers - 1],
                },
            },
            indent=2,
        ),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        save_strategy="no",
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = CompletionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    )

    print("[SFT] starting training")
    trainer.train()

    print("[SFT] saving", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "base": args.base,
                "init_lora": str(Path(args.init_lora).resolve()) if args.init_lora else None,
                "dataset": str(Path(args.dataset).resolve()),
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "max_seq_len": args.max_seq_len,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "loss_type": args.loss_type,
                "anchor_weight": args.anchor_weight,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "top_layers": args.top_layers,
                "use_rslora": use_rslora,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[SFT] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
