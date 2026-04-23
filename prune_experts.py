import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from model.model_minimind import MOEFeedForward, MiniMindConfig, MiniMindForCausalLM




DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT    = "./out/full_sft_768_moe.pth"
MODEL_DIR     = "model"
JSONL_PATH    = "./dataset/data/sft_valid.jsonl"

PROFILE_BATCH = 512 # tokens per batch during profiling forward pass
PROFILE_MAXLEN = 256 # max token length per prompt during profiling
PPL_SAMPLES   = 500 # number of prompts used to compute perplexity
PPL_MAXLEN    = 512 # max token length per prompt during perplexity eval
BENCH_PROMPT  = "解释一下机器学习的基本原理，包括监督学习、无监督学习和强化学习的区别。"
BENCH_TOKENS  = 150 # tokens to generate per benchmark run
BENCH_RUNS    = 5 # averaged runs for stable TPS measurement

# Pruning threshold on the combined (harmonic-mean) importance score.
# Experts below this value will be disabled at inference time.
#   0.10 → conservative (only obvious dead-weight experts)
#   0.15 → moderate
#   0.20 → aggressive
PRUNE_THRESHOLD = 0.15


# 1. Model + tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = MiniMindForCausalLM(MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    use_moe=True,
    inference_rope_scaling=False,
))
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE), strict=True)
model.to(DEVICE).eval()
print(f"Model loaded on {DEVICE}.\n")


# 2. Dataset utilities

def extract_user_prompts(path: str) -> list[str]:
    """Read all user-turn content strings from a JSONL conversation file."""
    prompts = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for turn in obj.get("conversations") or []:
                if isinstance(turn, dict) and turn.get("role") == "user":
                    content = (turn.get("content") or "").strip()
                    if content:
                        prompts.append(content)
    return prompts


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str]):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]


def build_dataloader(prompts: list[str], batch_size: int) -> DataLoader:
    return DataLoader(PromptDataset(prompts), batch_size=batch_size,
                      shuffle=False, drop_last=False)


# 3. Expert profiling

def profile_experts(model, prompts: list[str]):
    """
    Run a full forward pass over all prompts and accumulate per-expert
    frequency and gate-confidence statistics inside each MOEFeedForward layer.
    No gradient computation needed — inference only.
    """
    # Reset any stale stats from a previous run
    for layer in model.model.layers:
        if isinstance(layer.mlp, MOEFeedForward):
            layer.mlp.reset_eval_stats()

    dl = build_dataloader(prompts, PROFILE_BATCH)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc="Profiling experts"):
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=PROFILE_MAXLEN,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            model(input_ids=inputs["input_ids"],
                  attention_mask=inputs["attention_mask"])


def print_importance_table(model):
    """Pretty-print per-layer, per-expert importance scores."""
    print("\n" + "=" * 62)
    print(f"{'Expert Importance Scores':^62}")
    print("=" * 62)
    print(f"  {'Layer':<6} {'Expert':<8} {'Freq':>8} {'AvgGate':>10} {'Combined':>10}")
    print("-" * 62)

    all_scores = {}
    for i, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, MOEFeedForward):
            continue
        scores = layer.mlp.get_expert_importance()
        all_scores[i] = scores
        for eid in range(model.config.num_experts):
            flag = "  ← low" if scores["combined"][eid].item() < PRUNE_THRESHOLD else ""
            print(
                f"  {i:<6} {eid:<8} "
                f"{scores['frequency'][eid].item():>8.3f} "
                f"{scores['confidence'][eid].item():>10.3f} "
                f"{scores['combined'][eid].item():>10.3f}"
                f"{flag}"
            )
    print("=" * 62)
    return all_scores


# 4. Pruning plan

def build_pruning_plan(model, threshold: float) -> dict[int, list[int]]:
    """
    Return {layer_idx: [expert_ids_to_disable]} for all experts whose
    combined importance score falls below `threshold`.

    Safety constraint: at least one expert must remain active per layer.
    """
    plan = {}
    for i, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, MOEFeedForward):
            continue
        scores = layer.mlp.get_expert_importance()
        combined = scores["combined"]
        to_disable = [
            e for e in range(model.config.num_experts)
            if combined[e].item() < threshold
        ]
        # Never disable every expert in a layer
        if 0 < len(to_disable) < model.config.num_experts:
            plan[i] = to_disable
    return plan


def apply_pruning_plan(model, plan: dict[int, list[int]]):
    """Disable the experts listed in `plan` and print a summary."""
    model.enable_all_experts()   # start from a clean state

    if not plan:
        print("\nNo experts fell below the threshold — nothing to prune.")
        return

    print(f"\nPruning plan (threshold = {PRUNE_THRESHOLD}):")
    for layer_idx, expert_ids in sorted(plan.items()):
        for eid in expert_ids:
            model.disable_expert(eid, layer_idx=layer_idx)
        print(f"  Layer {layer_idx}: disabled experts {expert_ids}")


# 5. Perplexity
def compute_perplexity(model, prompts: list[str]) -> float:
    """
    Compute token-level perplexity on the first PPL_SAMPLES prompts.
    Lower is better; large jumps after pruning signal accuracy damage.
    """
    subset = prompts[:PPL_SAMPLES]
    total_nll, total_tokens = 0.0, 0

    model.eval()
    with torch.no_grad():
        for text in tqdm(subset, desc="Perplexity", leave=False):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=PPL_MAXLEN,
            ).to(DEVICE)
            input_ids = inputs["input_ids"]
            # Use input as both input and label; model computes mean CE loss
            out = model(input_ids, labels=input_ids)
            n_tokens = input_ids.shape[1] - 1   # model shifts internally
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens

    return math.exp(total_nll / total_tokens)


# 6. Throughput benchmark (tokens per second)

def benchmark_tps(model, prompt: str = BENCH_PROMPT,
                  max_new_tokens: int = BENCH_TOKENS,
                  runs: int = BENCH_RUNS) -> float:
    """
    Measure average generation throughput in tokens/second.
    Runs a short warmup first, then averages `runs` timed generations.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    def _sync():
        if DEVICE != "cpu":
            torch.cuda.synchronize()

    # Warmup (avoids cold-start GPU overhead polluting measurements)
    model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,
        do_sample=False,
    )
    _sync()

    tps_list = []
    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        out = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        _sync()
        elapsed = time.perf_counter() - t0
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        tps_list.append(n_new / elapsed)

    return sum(tps_list) / len(tps_list)


# 7. Complexity metrics

def count_parameters(model) -> dict[str, int]:
    """
    Break down parameter counts into:
      - total: everything in the model
      - active: only parameters in non-disabled experts
      - disabled: parameters sitting in disabled (but still allocated) experts
    """
    total_params = sum(p.numel() for p in model.parameters())
    disabled_params = 0

    for layer in model.model.layers:
        mlp = layer.mlp
        if not isinstance(mlp, MOEFeedForward):
            continue
        for eid, expert in enumerate(mlp.experts):
            if eid in mlp.disabled_experts:
                disabled_params += sum(p.numel() for p in expert.parameters())

    return {
        "total":    total_params,
        "active":   total_params - disabled_params,
        "disabled": disabled_params,
    }


def count_disabled_experts(model) -> dict[str, int]:
    """Count total and per-layer disabled experts across all MoE layers."""
    total_experts  = 0
    total_disabled = 0
    per_layer      = {}

    for i, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, MOEFeedForward):
            continue
        n = model.config.num_experts
        d = len(layer.mlp.disabled_experts)
        total_experts  += n
        total_disabled += d
        per_layer[i]    = {"total": n, "disabled": d}

    return {
        "total_experts":   total_experts,
        "total_disabled":  total_disabled,
        "sparsity":        total_disabled / max(total_experts, 1),
        "per_layer":       per_layer,
    }


def estimate_moe_flops_per_token(config: MiniMindConfig,
                                  n_disabled_per_layer: dict) -> dict[str, float]:
    """
    Rough FLOPs estimate for a single token passing through all MoE FFN layers.
    Each FeedForward does 2 matmuls: (hidden → intermediate) and (intermediate → hidden).
    We count active experts only (disabled ones are never called).

    Note: this is a lower-bound proxy, not a hardware-accurate FLOP count.
    """
    h  = config.hidden_size
    im = config.moe_intermediate_size
    k  = config.num_experts_per_tok   # experts actually called per token

    # Baseline: k active experts per layer (existing top-k gating)
    flops_baseline = 0.0
    flops_pruned   = 0.0

    for i, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, MOEFeedForward):
            continue
        # 2 matmuls per expert × 2 (gate + up projections feed into down)
        flops_per_expert = 2 * (h * im + im * h)
        flops_baseline  += k * flops_per_expert

        # After pruning: effective pool shrinks, but top-k still fires
        # The saving comes from the gate never selecting disabled experts
        active = config.num_experts - n_disabled_per_layer.get(i, 0)
        effective_k = min(k, active)
        flops_pruned += effective_k * flops_per_expert

    return {
        "baseline_GFLOPs_per_token": flops_baseline / 1e9,
        "pruned_GFLOPs_per_token":   flops_pruned   / 1e9,
        "flop_reduction_pct":        100 * (1 - flops_pruned / max(flops_baseline, 1)),
    }


# 8. Main experiment

if __name__ == "__main__":

    # ── Load and split prompts ─────────────────────────────────────────────
    all_prompts = extract_user_prompts(JSONL_PATH)
    print(f"Loaded {len(all_prompts):,} user prompts from {JSONL_PATH}")

    # ── Step 1: Profile ────────────────────────────────────────────────────
    print("\n[Step 1] Profiling expert activation across validation set...")
    profile_experts(model, all_prompts)
    all_scores = print_importance_table(model)

    # ── Step 2: Baseline metrics (before any pruning) ─────────────────────
    print("\n[Step 2] Computing BASELINE metrics...")
    model.enable_all_experts()

    ppl_baseline  = compute_perplexity(model, all_prompts)
    tps_baseline  = benchmark_tps(model)
    params_before = count_parameters(model)

    print(f"  Perplexity : {ppl_baseline:.4f}")
    print(f"  Throughput : {tps_baseline:.2f} tokens/s")
    print(f"  Parameters : {params_before['total']:,} total  |  {params_before['active']:,} active")

    # ── Step 3: Build and apply pruning plan ──────────────────────────────
    print("\n[Step 3] Building pruning plan...")
    plan = build_pruning_plan(model, threshold=PRUNE_THRESHOLD)
    apply_pruning_plan(model, plan)

    # ── Step 4: Post-pruning metrics ──────────────────────────────────────
    print("\n[Step 4] Computing POST-PRUNING metrics...")

    ppl_pruned   = compute_perplexity(model, all_prompts)
    tps_pruned   = benchmark_tps(model)
    params_after = count_parameters(model)
    expert_stats = count_disabled_experts(model)

    n_disabled_per_layer = {
        i: info["disabled"]
        for i, info in expert_stats["per_layer"].items()
    }
    flop_stats = estimate_moe_flops_per_token(model.config, n_disabled_per_layer)

    print(f"  Perplexity : {ppl_pruned:.4f}")
    print(f"  Throughput : {tps_pruned:.2f} tokens/s")
    print(f"  Parameters : {params_after['total']:,} total  |  {params_after['active']:,} active")

    # ── Step 5: Final summary ─────────────────────────────────────────────
    ppl_delta     = ppl_pruned - ppl_baseline
    ppl_delta_pct = 100 * ppl_delta / max(ppl_baseline, 1e-9)
    speedup       = tps_pruned / max(tps_baseline, 1e-9)
    param_saved   = params_before["active"] - params_after["active"]

    print("\n" + "=" * 62)
    print(f"{'Pruning Experiment — Final Report':^62}")
    print("=" * 62)

    print(f"\n  Threshold used          : {PRUNE_THRESHOLD}")
    print(f"  Experts disabled        : {expert_stats['total_disabled']} / {expert_stats['total_experts']}")
    print(f"  Expert sparsity         : {expert_stats['sparsity']:.1%}")

    print(f"\n  --- Accuracy ---")
    print(f"  Perplexity (baseline)   : {ppl_baseline:.4f}")
    print(f"  Perplexity (pruned)     : {ppl_pruned:.4f}")
    print(f"  PPL change              : {ppl_delta:+.4f}  ({ppl_delta_pct:+.2f}%)")

    print(f"\n  --- Speed ---")
    print(f"  Throughput (baseline)   : {tps_baseline:.2f} tokens/s")
    print(f"  Throughput (pruned)     : {tps_pruned:.2f} tokens/s")
    print(f"  Speedup                 : {speedup:.4f}x")

    print(f"\n  --- Parameters ---")
    print(f"  Total params            : {params_before['total']:,}")
    print(f"  Active params (baseline): {params_before['active']:,}")
    print(f"  Active params (pruned)  : {params_after['active']:,}")
    print(f"  Params saved (active)   : {param_saved:,}  ({100*param_saved/max(params_before['active'],1):.2f}%)")

    print(f"\n  --- Estimated FLOPs per token (MoE layers only) ---")
    print(f"  Baseline                : {flop_stats['baseline_GFLOPs_per_token']:.6f} GFLOPs")
    print(f"  Pruned                  : {flop_stats['pruned_GFLOPs_per_token']:.6f} GFLOPs")
    print(f"  FLOP reduction          : {flop_stats['flop_reduction_pct']:.2f}%")

    print("\n" + "=" * 62)
    

    # Save pruning plan to the same directory as the model weights
    import json

    pruning_plan = {
        str(layer_idx): list(layer.mlp.disabled_experts)
        for layer_idx, layer in enumerate(model.model.layers)
        if isinstance(layer.mlp, MOEFeedForward) and layer.mlp.disabled_experts
    }
    plan_path = "./out/pruning_plan.json"
    with open(plan_path, "w") as f:
        json.dump(pruning_plan, f, indent=2)
    print(f"\nPruning plan saved to {plan_path}")
    print(json.dumps(pruning_plan, indent=2))