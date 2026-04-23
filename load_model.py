import json
import torch
from transformers import AutoTokenizer, TextStreamer
from model.model_minimind import MOEFeedForward, MiniMindConfig, MiniMindForCausalLM


DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT  = "./out/full_sft_768_moe.pth"
MODEL_DIR   = "model"
PLAN_PATH   = "./out/pruning_plan.json"


def load_model(checkpoint: str) -> MiniMindForCausalLM:
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=True,
        inference_rope_scaling=False,
    ))
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE), strict=True)
    model.to(DEVICE).eval()
    return model


def apply_plan(model: MiniMindForCausalLM, plan_path: str) -> dict:
    """
    Reads the JSON pruning plan and disables the listed experts.
    Returns the plan dict so the caller can inspect it.
    """
    with open(plan_path) as f:
        plan = json.load(f)

    # JSON keys are always strings — convert back to int
    for layer_idx_str, expert_ids in plan.items():
        for eid in expert_ids:
            model.disable_expert(eid, layer_idx=int(layer_idx_str))

    return plan


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # ── Load base model (no pruning)
    print("Loading base model...")
    base_model = load_model(CHECKPOINT)
    print("  Base model ready — all experts active.\n")

    # ── Load pruned model
    print("Loading pruned model...")
    pruned_model = load_model(CHECKPOINT)
    plan = apply_plan(pruned_model, PLAN_PATH)

    # Print which experts were disabled
    print("  Pruning plan applied:")
    for layer_idx, expert_ids in sorted(plan.items(), key=lambda x: int(x[0])):
        print(f"    Layer {layer_idx}: experts {expert_ids} disabled")

    # Quick sanity check — confirm disabled sets are non-empty
    total_disabled = sum(
        len(layer.mlp.disabled_experts)
        for layer in pruned_model.model.layers
        if isinstance(layer.mlp, MOEFeedForward)
    )
    print(f"  Total experts disabled: {total_disabled}\n")

    # ── Side-by-side generation
    test_prompts = [
        "解释一下机器学习的基本原理",
        "请用Python写一个计算斐波那契数列的函数",
        "为什么天空是蓝色的",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            return_tensors="pt",
        ).to(DEVICE)

        print(f"{'='*60}")
        print(f"Prompt: {prompt}")

        for label, mdl in [("Base", base_model), ("Pruned", pruned_model)]:
            print(f"\n[{label}]")
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            mdl.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50, #200
                do_sample=True,
                temperature=0.85,
                top_p=0.95,
                streamer=streamer,
            )

        print("Done")