import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


@dataclass
class GenResult:
    text: str
    gen_tokens: int
    latency_sec: float


def load_cases(path: str):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {i}: {e}") from e
            if "id" not in item or "prompt" not in item:
                raise ValueError(f"Missing id/prompt at line {i}")
            item.setdefault("match_type", "contains_any")
            item.setdefault("weight", 1.0)
            cases.append(item)
    if not cases:
        raise ValueError(f"No cases found: {path}")
    return cases


def extract_first_number(text: str):
    import re

    m = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def score_case(case: dict, response: str):
    import re

    mt = str(case.get("match_type", "contains_any")).lower()
    resp = response.strip()
    resp_lower = resp.lower()

    if mt == "exact":
        target = str(case.get("answer", "")).strip().lower()
        return resp_lower == target

    if mt == "numeric":
        pred = extract_first_number(resp)
        if pred is None:
            return False
        if case.get("answer") is None:
            return False
        answer = float(case.get("answer", 0.0))
        tol = float(case.get("tolerance", 0.0))
        return abs(pred - answer) <= tol

    if mt == "regex":
        pattern = case.get("pattern", "")
        return bool(pattern and re.search(pattern, resp, flags=re.IGNORECASE))

    expected = [str(x).lower() for x in case.get("expected", [])]
    if not expected:
        return False

    if mt == "contains_all":
        return all(k in resp_lower for k in expected)

    if mt == "contains_any":
        return any(k in resp_lower for k in expected)

    return False


def fidelity(a: str, b: str) -> float:
    a = a.strip().lower()
    b = b.strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def maybe_sync(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(checkpoint: str, device: str, hidden_size: int, num_hidden_layers: int):
    # Match load_model.py practice: create MiniMind model explicitly and load state_dict.
    model = MiniMindForCausalLM(
        MiniMindConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            use_moe=True,
            inference_rope_scaling=False,
        )
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.to(device).eval()
    return model


def apply_plan(model: MiniMindForCausalLM, plan_path: str):
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    for layer_idx_str, expert_ids in plan.items():
        for eid in expert_ids:
            model.disable_expert(eid, layer_idx=int(layer_idx_str))
    return plan


def build_inputs(tokenizer, prompt: str, device: str):
    # Match load_model.py practice: build chat template once and reuse same inputs for both models.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt").to(device)


def generate_one(model, tokenizer, inputs, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float, device: str):
    maybe_sync(device)
    start = time.time()
    with torch.no_grad():
        # Keep generate call style aligned with load_model.py.
        out = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    maybe_sync(device)
    elapsed = time.time() - start

    prompt_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return GenResult(text=text, gen_tokens=int(out[0].shape[0] - prompt_len), latency_sec=float(elapsed))


def main():
    parser = argparse.ArgumentParser(description="Lightweight MiniMind pruning benchmark")
    parser.add_argument("--benchmark_file", default="./benchmark/fixed_prompts.jsonl", type=str)
    parser.add_argument("--report_dir", default="./benchmark/reports", type=str)
    parser.add_argument("--model_dir", default="model", type=str)
    parser.add_argument("--checkpoint", default="./out/full_sft_768_moe.pth", type=str)
    parser.add_argument("--pruning_plan", default="./out/pruning_plan.json", type=str)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--do_sample", default=1, type=int, choices=[0, 1])
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--shuffle_cases", default=1, type=int, choices=[0, 1])
    parser.add_argument("--num_runs", default=1, type=int,
                        help="Number of runs per case (>1 enables multi-run averaging for do_sample=1)")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--verbose", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cases = load_cases(args.benchmark_file)
    if bool(args.shuffle_cases):
        random.shuffle(cases)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base_model = load_model(args.checkpoint, args.device, args.hidden_size, args.num_hidden_layers)
    pruned_model = load_model(args.checkpoint, args.device, args.hidden_size, args.num_hidden_layers)
    plan = apply_plan(pruned_model, args.pruning_plan)

    os.makedirs(args.report_dir, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = os.path.join(args.report_dir, f"benchmark_{run_name}.json")
    report_csv = os.path.join(args.report_dir, f"benchmark_{run_name}.csv")

    results = []
    total_weight = 0.0
    base_pass_weight = 0.0
    pruned_pass_weight = 0.0
    f_sum = 0.0
    speedups = []

    num_runs = max(1, args.num_runs)

    for i, case in enumerate(cases, start=1):
        prompt = case["prompt"]
        weight = float(case.get("weight", 1.0))
        total_weight += weight

        inputs = build_inputs(tokenizer, prompt, args.device)

        # -- Multi-run loop: generate num_runs times, aggregate results ----------
        base_passes = 0
        pruned_passes = 0
        sim_sum = 0.0
        speedup_sum = 0.0
        base_latency_sum = 0.0
        pruned_latency_sum = 0.0
        last_base_text = ""
        last_pruned_text = ""

        for run_idx in range(num_runs):
            # Give each run a distinct but reproducible torch seed
            # run_seed = args.seed + i * 1000 + run_idx
            # torch.manual_seed(run_seed)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed_all(run_seed)

            base = generate_one(
                base_model, tokenizer, inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )

            # torch.manual_seed(run_seed)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed_all(run_seed)

            pruned = generate_one(
                pruned_model, tokenizer, inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )

            base_passes += int(score_case(case, base.text))
            pruned_passes += int(score_case(case, pruned.text))
            sim_sum += fidelity(base.text, pruned.text)
            sp = (base.latency_sec / pruned.latency_sec) if pruned.latency_sec > 0 else 0.0
            speedup_sum += sp
            base_latency_sum += base.latency_sec
            pruned_latency_sum += pruned.latency_sec
            last_base_text = base.text
            last_pruned_text = pruned.text

        # -- Aggregate across runs -----------------------------------------------
        base_pass_rate = base_passes / num_runs
        pruned_pass_rate = pruned_passes / num_runs
        sim = sim_sum / num_runs
        speedup = speedup_sum / num_runs

        base_pass_weight += weight * base_pass_rate
        pruned_pass_weight += weight * pruned_pass_rate
        f_sum += sim
        speedups.append(speedup)

        row = {
            "index": i,
            "id": case.get("id", f"case_{i}"),
            "match_type": case.get("match_type", "contains_any"),
            "weight": weight,
            "base_correct": round(base_pass_rate, 4),
            "pruned_correct": round(pruned_pass_rate, 4),
            "fidelity": sim,
            "speedup": speedup,
            "base_latency_sec": base_latency_sum / num_runs,
            "pruned_latency_sec": pruned_latency_sum / num_runs,
            "base_gen_tokens": base.gen_tokens,
            "pruned_gen_tokens": pruned.gen_tokens,
            "prompt": prompt,
            "base_response": last_base_text,
            "pruned_response": last_pruned_text,
        }
        results.append(row)

        if args.verbose:
            tag = f"(avg of {num_runs} runs) " if num_runs > 1 else ""
            print(
                f"[{i}/{len(cases)}] {row['id']} {tag}"
                f"| base={base_pass_rate:.0%} "
                f"| pruned={pruned_pass_rate:.0%} "
                f"| fidelity={sim:.3f} | speedup={speedup:.2f}x"
            )

    base_acc = (sum(r["base_correct"] for r in results) / len(results)) if results else 0.0
    pruned_acc = (sum(r["pruned_correct"] for r in results) / len(results)) if results else 0.0
    base_wacc = (base_pass_weight / total_weight) if total_weight > 0 else 0.0
    pruned_wacc = (pruned_pass_weight / total_weight) if total_weight > 0 else 0.0
    avg_f = (f_sum / len(results)) if results else 0.0
    avg_speedup = (sum(speedups) / len(speedups)) if speedups else 0.0

    summary = {
        "mode": "base_vs_pruned_same_checkpoint",
        "num_cases": len(results),
        "num_runs_per_case": num_runs,
        "benchmark_file": args.benchmark_file,
        "checkpoint": args.checkpoint,
        "pruning_plan": args.pruning_plan,
        "disabled_layers": [str(x) for x in sorted(int(k) for k in plan.keys())],
        "base_accuracy": round(base_acc, 4),
        "pruned_accuracy": round(pruned_acc, 4),
        "base_weighted_accuracy": round(base_wacc, 4),
        "pruned_weighted_accuracy": round(pruned_wacc, 4),
        "accuracy_delta": round(pruned_acc - base_acc, 4),
        "weighted_accuracy_delta": round(pruned_wacc - base_wacc, 4),
        "avg_fidelity": round(avg_f, 4),
        "avg_speedup": round(avg_speedup, 4),
        "args": vars(args),
        "results": results,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_fields = [
        "index",
        "id",
        "match_type",
        "weight",
        "base_correct",
        "pruned_correct",
        "fidelity",
        "speedup",
        "base_latency_sec",
        "pruned_latency_sec",
        "base_gen_tokens",
        "pruned_gen_tokens",
        "prompt",
        "base_response",
        "pruned_response",
    ]
    with open(report_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in csv_fields})

    print("\n========== Benchmark Summary ==========")
    print(f"Cases: {len(results)}")
    print(f"Base Accuracy: {base_acc * 100:.2f}%")
    print(f"Pruned Accuracy: {pruned_acc * 100:.2f}%")
    print(f"Accuracy Delta (pruned-base): {(pruned_acc - base_acc) * 100:.2f}%")
    print(f"Average Fidelity: {avg_f:.4f}")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Report JSON: {report_json}")
    print(f"Report CSV: {report_csv}")


if __name__ == "__main__":
    main()
