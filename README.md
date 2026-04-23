# Dynamic Expert Pruning

This repository implements a lightweight dynamic expert pruning workflow for a MiniMind MoE checkpoint. It profiles expert usage, builds a pruning plan, disables low-importance experts at inference time, and compares base versus pruned behavior on the same model weights.

The pruning pipeline is written in PyTorch. Experts are ranked using routing frequency and gate confidence, and the final pruning plan is stored as a layer-to-expert mapping in `./out/pruning_plan.json`.

## What Is Included

- `prune_experts.py` profiles experts, builds the pruning plan, applies it, and prints summary metrics.
- `benchmark.py` compares base and pruned generation on a JSONL prompt set and writes JSON and CSV reports.
- `load_model.py` loads the same checkpoint twice and shows side-by-side outputs before and after pruning.
- `eval_llm.py` provides interactive chat-style inference for the MiniMind model.
- `trainer/` contains the pretraining, SFT, distillation, and RL scripts used to produce the checkpoint.

## Repository Layout

- `model/` model definition and MoE routing logic
- `trainer/` training, distillation, and RL scripts
- `dataset/` dataset utilities and the dataset notebook
- `benchmark.py` benchmark runner for base versus pruned inference
- `prune_experts.py` pruning plan generation
- `load_model.py` quick sanity-check demo
- `eval_llm.py` interactive inference entry point

## Setup

Install dependencies first:

```bash
pip install -r requirements.txt
```

If you plan to reproduce training, start from `dataset/build_datasets.ipynb` and then run the scripts under `trainer/`.

## Typical Workflow

1. Train or obtain a checkpoint such as `./out/full_sft_768_moe.pth`.
2. Run `prune_experts.py` to profile experts and save the pruning plan.
3. Run `benchmark.py` to compare accuracy, fidelity, latency, and speedup before and after pruning.
4. Optionally run `load_model.py` to inspect generation from the base and pruned models.

## Pruning

```bash
python prune_experts.py
```

This produces a pruning plan in `./out/pruning_plan.json` and prints summary metrics such as perplexity, throughput, parameter counts, and estimated FLOPs.

## Benchmark

```bash
python benchmark.py \
  --benchmark_file ./benchmark/fixed_prompts.jsonl \
  --checkpoint ./out/full_sft_768_moe.pth \
  --pruning_plan ./out/pruning_plan.json \
  --report_dir ./benchmark/reports \
  --num_runs 5
```

The benchmark file is JSONL. Each case must include `id` and `prompt`, and may also include fields such as `match_type`, `expected`, `answer`, `pattern`, `tolerance`, and `weight`.

Supported match types include:

- `contains_any`
- `contains_all`
- `exact`
- `numeric`
- `regex`

Benchmark reports are written to `./benchmark/reports` as timestamped JSON and CSV files.

## Inference

```bash
python eval_llm.py --load_from model --weight full_sft
```

If you are using a local PyTorch checkpoint, keep the weights under `./out/` and point `--weight` at the saved prefix.

```bash
python load_model.py
```

This script loads the base model and the pruned model, applies `./out/pruning_plan.json`, and prints side-by-side generations for a few test prompts.

## Notes

- The pruning plan is applied by disabling experts layer by layer, while keeping at least one expert active in each MoE layer.
- `benchmark.py` and `load_model.py` both use the same checkpoint-loading path so the comparison stays consistent.
