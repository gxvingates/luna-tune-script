# luna-tune-script

Small shareable subset of my Luna fine-tuning workflow.

This repo is not the full project. It is the minimal script/notebook bundle I use to:

- build a completion-only JSONL dataset for a Luna run
- launch a Qwen 3.5 9B LoRA fine-tune from a notebook
- override the system prompt at train time
- export the finished adapter to GGUF through a safe CPU merge path

## Files

- `training_and_quant.ipynb`
  - the notebook entry point
  - right now it only contains 2 cells: a config/status cell and a train cell
- `train_qwen35_9b_completion_sft.py`
  - the actual trainer
  - supports completion-only loss, prompt override, fresh LoRA or init LoRA, and top-layer targeting
- `build_luna_antispam_sledgehammer_dataset.py`
  - example dataset builder for the anti-spam instruct run
  - mixes a primary v28-style corpus with a small v32 patch set and scrubs weak one-liners / stall-only replies
- `luna_system_prompt_smart_human.txt`
  - example system prompt used by the notebook + trainer
- `export_luna_qwen35_9b_cpu_merge.py`
  - export path that merges the LoRA on CPU first, validates the merged HF behavior, then converts to GGUF and quantizes

## Expected Environment

This was written around my local setup, not as a generic pip package.

Main assumptions:

- Python env already has `unsloth`, `transformers`, `datasets`, `peft`, `torch`, and `bitsandbytes`
- `llama.cpp` exists locally if you want GGUF export
- your dataset is already in prompt/completion JSONL format

The notebook and scripts currently use hard-coded absolute paths. If someone else uses this repo, they should change those paths first.

## Dataset Format

The trainer expects each JSONL row to look like this:

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "completion": [
    {"role": "assistant", "content": "..."}
  ]
}
```

The completion-only masking logic is handled inside `train_qwen35_9b_completion_sft.py`.

## Notebook Flow

`training_and_quant.ipynb` is the top-level launcher.

Cell 1:
- defines the run config
- prints the exact values that will be used
- sets the output directory name

Cell 2:
- calls `train_qwen35_9b_completion_sft.py` with the config from cell 1

Run order is always:

1. run the first cell
2. run the second cell

## Trainer Notes

The trainer supports:

- `--init-lora` for continuation / stage-2 style runs
- `--system-prompt-file` to replace or inject the system prompt into every row at train time
- `--top-layers` / `--lora-top-layers` to target only the top N transformer layers
- completion-only loss masking
- a fresh LoRA path or a resume-from-adapter path

The current notebook config is set up for a fresh LoRA run on `unsloth/Qwen3.5-9B` with:

- `r=16`
- `alpha=64`
- `top_layers=24`
- `max_steps=180`

## Dataset Builder Notes

`build_luna_antispam_sledgehammer_dataset.py` is an example of how I prepare a Luna dataset before training.

What it does:

- loads a primary dataset and a smaller patch dataset
- removes stall-only replies from the primary set
- deterministically culls a chunk of naked one-word / short-word replies
- rebalances the patch set back to a target share
- wraps every row with a chosen system prompt

It writes both:

- the final training JSONL
- a metadata JSON file describing the blend and scrub

## GGUF Export Path

`export_luna_qwen35_9b_cpu_merge.py` exists because some earlier merged export paths were corrupting behavior.

What this script does:

1. loads the base model on CPU
2. loads the LoRA and calls `merge_and_unload()`
3. saves merged HF shards
4. patches the chat template metadata
5. runs a small adapter-vs-merged smoke check
6. converts merged HF to F16 GGUF
7. quantizes to `Q6_K`
8. deletes the temporary F16 file after success

## Typical Usage

Build the dataset:

```bash
python build_luna_antispam_sledgehammer_dataset.py
```

Then open the notebook and run the 2 cells.

If you want a pure CLI training launch instead of the notebook, the notebook is just calling:

```bash
python train_qwen35_9b_completion_sft.py \
  --base unsloth/Qwen3.5-9B \
  --dataset /path/to/dataset.jsonl \
  --system-prompt-file /path/to/prompt.txt \
  --output-dir /path/to/run_dir \
  --learning-rate 1.5e-5 \
  --epochs 1 \
  --max-steps 180 \
  --batch-size 1 \
  --grad-accum 4 \
  --max-seq-len 1536 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --lora-r 16 \
  --lora-alpha 64 \
  --top-layers 24
```

Then export:

```bash
python export_luna_qwen35_9b_cpu_merge.py \
  --lora /path/to/adapter_dir \
  --base unsloth/Qwen3.5-9B \
  --quant Q6_K \
  --name my_run_name
```

## Important

This is intentionally the working script bundle, not a polished general-purpose framework. Anyone reusing it should expect to edit:

- local paths
- dataset source paths
- output paths
- prompt text
- model name
