#!/usr/bin/env python3
"""
Export Luna's Qwen3.5-9B LoRA by doing a plain PEFT CPU merge first.

This avoids Unsloth's merged-save path entirely and validates merged-HF behavior
before any GGUF conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


LUNA_CHAT_TEMPLATE = """{% if messages and messages[0]['role'] == 'system' %}
{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif %}
{% for message in loop_messages %}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}
{% endfor %}
{% if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{% endif %}"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Luna Qwen3.5-9B LoRA via CPU merge.")
    parser.add_argument("--lora", required=True)
    parser.add_argument("--base", default="unsloth/Qwen3.5-9B")
    parser.add_argument("--out-dir", default="/home/gavin/tuning/exports")
    parser.add_argument("--name", default=None)
    parser.add_argument("--quant", default="Q6_K")
    parser.add_argument("--llama-cpp", default="/home/gavin/llama.cpp")
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def _derive_name(lora_path: Path) -> str:
    if lora_path.name in {"final_lora", "best_lora", "last_lora"} and lora_path.parent.name:
        return lora_path.parent.name
    return lora_path.name


ARGS = _parse_args()
LORA_PATH = Path(ARGS.lora).expanduser().resolve()
EXPORT_ROOT = Path(ARGS.out_dir).expanduser().resolve()
EXPORT_NAME = ARGS.name or _derive_name(LORA_PATH)
LLAMA_CPP = Path(ARGS.llama_cpp).expanduser().resolve()
QUANT_METHOD = ARGS.quant
QUANT_SLUG = QUANT_METHOD.lower()

MERGED_HF_DIR = EXPORT_ROOT / f"{EXPORT_NAME}_merged_hf_cpumerge"
OUTPUT_F16 = EXPORT_ROOT / f"{EXPORT_NAME}_f16_cpumerge.gguf"
OUTPUT_GGUF = EXPORT_ROOT / f"{EXPORT_NAME}_{QUANT_SLUG}_cpumerge.gguf"
LOG_PATH = EXPORT_ROOT / f"{EXPORT_NAME}_cpumerge_export.log"


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _patch_adapter_base() -> None:
    config_path = LORA_PATH / "adapter_config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("base_model_name_or_path") == ARGS.base:
        return
    cfg["base_model_name_or_path"] = ARGS.base
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log(f"Patched adapter_config base_model_name_or_path -> {ARGS.base}")


def _patch_merged_metadata(path: Path) -> None:
    config_path = path / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    template_path = path / "chat_template.jinja"
    template_path.write_text(LUNA_CHAT_TEMPLATE + "\n", encoding="utf-8")

    tokenizer_config_path = path / "tokenizer_config.json"
    tok_cfg = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tok_cfg["chat_template"] = LUNA_CHAT_TEMPLATE
    tokenizer_config_path.write_text(json.dumps(tok_cfg, indent=2), encoding="utf-8")
    log("Patched merged HF config + Luna chat template")


def _build_messages(system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Runtime Context For Luna (not user speech):\n"
                "This is metadata about Luna's own current state. Use it as context only.\n"
                "Luna State\n"
                "Luna discord status: vibing\n"
                "the thing everyone is doing rn: chatting"
            ),
        },
        {"role": "user", "content": "[Gavin]: hey luna"},
    ]


def _smoke_generate(model_path: str, *, is_adapter: bool) -> str:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.cuda_visible_devices)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    system_prompt = Path("/home/gavin/tuning/luna_system_prompt_no_inner_thoughts.txt").read_text(encoding="utf-8").strip()
    messages = _build_messages(system_prompt)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ARGS.base if is_adapter else model_path, trust_remote_code=True)
    tokenizer.chat_template = LUNA_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        ARGS.base if is_adapter else model_path,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if is_adapter:
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = re.sub(r"<think>\s*</think>\s*", "", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.55,
            top_k=18,
            top_p=0.8,
            min_p=0.07,
            repetition_penalty=1.1,
            max_new_tokens=ARGS.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    return text


def _validate_merged_hf_behavior() -> None:
    adapter_text = _smoke_generate(str(LORA_PATH), is_adapter=True)
    merged_text = _smoke_generate(str(MERGED_HF_DIR), is_adapter=False)
    log(f"adapter smoke: {adapter_text[:240]!r}")
    log(f"merged smoke: {merged_text[:240]!r}")

    if "Thinking Process:" in merged_text:
        raise RuntimeError("Merged HF smoke test failed: merged artifact still emits reasoning trace")
    if merged_text.lower().count("potato") > 0 and "potato" not in adapter_text.lower():
        raise RuntimeError("Merged HF smoke test failed: merged artifact drifted into generic potato spillover")


def _validate_gguf_chat_template(path: Path) -> None:
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(path, "r")
    field = reader.fields.get("tokenizer.chat_template")
    if field is None:
        raise RuntimeError("GGUF is missing tokenizer.chat_template metadata")
    value = field.parts[field.data[0]]
    if hasattr(value, "tobytes"):
        value = value.tobytes().decode("utf-8")
    elif not isinstance(value, str):
        value = str(value)
    if "<think>" in value:
        raise RuntimeError("GGUF chat template still contains <think>")


def main() -> int:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if MERGED_HF_DIR.exists():
        shutil.rmtree(MERGED_HF_DIR)
    if OUTPUT_F16.exists():
        OUTPUT_F16.unlink()
    if OUTPUT_GGUF.exists():
        OUTPUT_GGUF.unlink()
    LOG_PATH.write_text("", encoding="utf-8")

    log(f"Starting CPU-merge export for {LORA_PATH}")
    _patch_adapter_base()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("Loading base model on CPU")
    base_model = AutoModelForCausalLM.from_pretrained(
        ARGS.base,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ARGS.base, trust_remote_code=True, use_fast=False)
    tokenizer.chat_template = LUNA_CHAT_TEMPLATE

    log("Loading LoRA and merging on CPU")
    model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    model = model.merge_and_unload()

    log(f"Saving merged HF shards to {MERGED_HF_DIR}")
    model.save_pretrained(str(MERGED_HF_DIR), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(MERGED_HF_DIR))
    del model, base_model, tokenizer
    torch.cuda.empty_cache()

    _patch_merged_metadata(MERGED_HF_DIR)
    _validate_merged_hf_behavior()

    convert_script = LLAMA_CPP / "convert_hf_to_gguf.py"
    quantize_bin = LLAMA_CPP / "llama-quantize"

    log(f"Converting merged HF -> F16 GGUF at {OUTPUT_F16}")
    convert = subprocess.run(
        [sys.executable, str(convert_script), str(MERGED_HF_DIR), "--outfile", str(OUTPUT_F16), "--outtype", "f16"],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if convert.returncode != 0:
        log(f"convert stdout tail:\n{convert.stdout[-4000:]}")
        log(f"convert stderr tail:\n{convert.stderr[-4000:]}")
        raise RuntimeError("GGUF F16 conversion failed")

    _validate_gguf_chat_template(OUTPUT_F16)

    log(f"Quantizing F16 -> {QUANT_METHOD} at {OUTPUT_GGUF}")
    quant = subprocess.run(
        [str(quantize_bin), str(OUTPUT_F16), str(OUTPUT_GGUF), QUANT_METHOD],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if quant.returncode != 0:
        log(f"quant stdout tail:\n{quant.stdout[-4000:]}")
        log(f"quant stderr tail:\n{quant.stderr[-4000:]}")
        raise RuntimeError(f"{QUANT_METHOD} quantization failed")

    _validate_gguf_chat_template(OUTPUT_GGUF)
    if OUTPUT_F16.exists():
        OUTPUT_F16.unlink()
        log(f"Removed intermediate F16 GGUF {OUTPUT_F16}")
    size_gb = OUTPUT_GGUF.stat().st_size / (1024 ** 3)
    log(f"Export complete -> {OUTPUT_GGUF} ({size_gb:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
