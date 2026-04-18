#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from itertools import cycle, islice
from pathlib import Path


ROOT = Path("/home/gavin/tuning")
SYSTEM_PROMPT_PATH = ROOT / "luna_system_prompt_smart_human.txt"
V28_PATH = ROOT / "formatted_v28_sleepiecore_gavinuser_completion" / "Direct_Messages_-_sleepie_1161959711390830643__sleepiecore__v28.jsonl"
V32_PATH = ROOT / "formatted_v32_textsocial_targeted_patch_completion" / "Direct_Messages_-_textsocial_targeted_patch__synthetic__v32.jsonl"
OUTPUT_PATH = ROOT / "datasets" / "sft_luna_antispam_sledgehammer_v28_v32.jsonl"
META_PATH = ROOT / "datasets" / "sft_luna_antispam_sledgehammer_v28_v32.meta.json"

PRIMARY_SHARE = 0.85
SECONDARY_SHARE = 0.15
ONE_LINER_CULL_RATE = 0.55

NAKED_ONE_LINERS = {
    "aight",
    "alright",
    "bet",
    "cool",
    "fine",
    "fr",
    "good",
    "k",
    "kk",
    "lmao",
    "lmfao",
    "lol",
    "nah",
    "no",
    "nope",
    "ok",
    "okay",
    "oki",
    "okie",
    "okii",
    "okiii",
    "real",
    "sure",
    "true",
    "ya",
    "yeah",
    "yep",
    "yup",
}

STALL_ONLY_REPLIES = {
    "brb",
    "give me a sec",
    "gimme a min",
    "gimme a sec",
    "gimme sec",
    "hold on",
    "hold up",
    "just a sec",
    "lemme check",
    "lemme look",
    "lemme think",
    "let me check",
    "let me look",
    "let me think",
    "one sec",
    "wait a sec",
}


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_reply(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _hash_ratio(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _completion_text(row: dict) -> str:
    completion = row.get("completion") or []
    if not completion:
        return ""
    return (completion[0].get("content") or "").strip()


def _is_naked_one_liner(text: str) -> bool:
    normalized = _normalize_reply(text)
    if not normalized or "\n" in text:
        return False
    if normalized not in NAKED_ONE_LINERS:
        return False
    return len(normalized.split()) <= 2


def _is_stall_only(text: str) -> bool:
    normalized = _normalize_reply(text)
    return normalized in STALL_ONLY_REPLIES


def _should_keep_one_liner(row: dict) -> bool:
    seed = json.dumps(row.get("prompt") or [], ensure_ascii=False, sort_keys=True) + "\n" + _completion_text(row)
    return _hash_ratio(seed) >= ONE_LINER_CULL_RATE


def _scrub_primary_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    stats = {
        "source_rows": len(rows),
        "stall_only_removed": 0,
        "naked_one_liners_seen": 0,
        "naked_one_liners_removed": 0,
    }

    for row in rows:
        text = _completion_text(row)
        if not text:
            continue
        if _is_stall_only(text):
            stats["stall_only_removed"] += 1
            continue
        if _is_naked_one_liner(text):
            stats["naked_one_liners_seen"] += 1
            if not _should_keep_one_liner(row):
                stats["naked_one_liners_removed"] += 1
                continue
        kept.append(row)

    stats["kept_rows"] = len(kept)
    stats["naked_one_liners_kept"] = (
        stats["naked_one_liners_seen"] - stats["naked_one_liners_removed"]
    )
    return kept, stats


def _wrap_with_system_prompt(row: dict, system_prompt: str) -> dict:
    prompt = row.get("prompt") or []
    completion = row.get("completion") or []
    return {
        "prompt": [{"role": "system", "content": system_prompt}, *prompt],
        "completion": completion,
    }


def _target_secondary_count(primary_count: int) -> int:
    return round(primary_count * SECONDARY_SHARE / PRIMARY_SHARE)


def main() -> int:
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    v28_rows = _load_rows(V28_PATH)
    v32_rows = _load_rows(V32_PATH)

    if not system_prompt:
        raise RuntimeError("System prompt file is empty.")
    if not v28_rows:
        raise RuntimeError("Primary v28 dataset is empty.")
    if not v32_rows:
        raise RuntimeError("Secondary v32 dataset is empty.")

    scrubbed_v28, scrub_stats = _scrub_primary_rows(v28_rows)
    if not scrubbed_v28:
        raise RuntimeError("Primary dataset became empty after anti-spam scrub.")

    target_v32 = _target_secondary_count(len(scrubbed_v28))
    repeated_v32 = list(islice(cycle(v32_rows), target_v32))

    blended_rows = [_wrap_with_system_prompt(row, system_prompt) for row in scrubbed_v28]
    blended_rows.extend(_wrap_with_system_prompt(row, system_prompt) for row in repeated_v32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in blended_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "dataset": str(OUTPUT_PATH),
        "system_prompt": str(SYSTEM_PROMPT_PATH),
        "base_model": "unsloth/Qwen3.5-9B",
        "blend": {
            "primary_dataset": str(V28_PATH),
            "secondary_dataset": str(V32_PATH),
            "primary_share": PRIMARY_SHARE,
            "secondary_share": SECONDARY_SHARE,
            "primary_examples_after_scrub": len(scrubbed_v28),
            "secondary_source_examples": len(v32_rows),
            "secondary_target_examples": target_v32,
            "total_examples": len(blended_rows),
            "actual_secondary_share": target_v32 / len(blended_rows),
        },
        "scrub": {
            "same_speaker_short_gap_concat": "already present in v28 source formatting",
            "naked_one_liner_cull_rate": ONE_LINER_CULL_RATE,
            **scrub_stats,
        },
        "training_plan": {
            "learning_rate": 1.5e-5,
            "max_steps": 180,
            "batch_size": 1,
            "grad_accum": 4,
            "lora_r": 16,
            "lora_alpha": 64,
            "top_layers": 24,
        },
        "notes": [
            "Anti-Spam Sledgehammer instruct tune dataset.",
            "Primary data is the same v28 sleepie core slice used in the prior successful run.",
            "v28 already preserves short-gap same-speaker concatenation from the raw sleepie DM CSV.",
            "Applied stall-only removal and a deterministic 55% cull of naked short one-liners on v28 only.",
            "v32 short-answer anchor rows are left untouched, then oversampled back to a 15% blend.",
        ],
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote dataset -> {OUTPUT_PATH}")
    print(f"Wrote metadata -> {META_PATH}")
    print(f"v28 source rows: {len(v28_rows)}")
    print(f"v28 kept rows: {len(scrubbed_v28)}")
    print(f"stall-only removed: {scrub_stats['stall_only_removed']}")
    print(
        "naked one-liners: "
        f"{scrub_stats['naked_one_liners_seen']} seen, {scrub_stats['naked_one_liners_removed']} removed"
    )
    print(f"v32 blended rows: {target_v32}")
    print(f"total rows: {len(blended_rows)}")
    print(f"actual v32 share: {target_v32 / len(blended_rows):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
