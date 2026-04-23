"""
train_expert_distillation.py
============================
Teacher-student distillation trainer with two novel extensions:

1. **Hidden-state distillation** (`--hidden_distill_weight > 0`):
   Transfers the aggregated MoE expert knowledge that lives in the teacher's
   final hidden representations into the denser student, by minimising an
   MSE loss between their L2-normalised hidden states.  A learnable linear
   projection is created automatically when the two hidden sizes differ.

   Total loss = α · CE  +  (1-α) · KL(logits)  +  λ · MSE(hidden states)

2. **Dynamic expert pruning** at inference (`--enable_dynamic_pruning`):
   Activates `MiniMindConfig.enable_dynamic_pruning` for the student model.
   During inference each token whose top-1 gate score exceeds
   `--pruning_threshold` is routed to only its single best expert, skipping
   all weaker expert forward passes for a real latency reduction.
   The threshold is tunable; a value of 0.5 is a safe starting point.
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
)

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl


def hidden_state_distillation_loss(student_hs, teacher_hs, projection=None):
    """MSE loss between the final hidden states of student and teacher.

    Transfers aggregated MoE expert knowledge into the student's representation
    space.  A linear projection is applied when the two models have different
    hidden sizes.  Both representations are L2-normalised before comparison so
    the loss measures directional alignment independently of magnitude.
    """
    s_h = student_hs                    # (bsz, seq_len, student_hidden)
    t_h = teacher_hs.detach()           # (bsz, seq_len, teacher_hidden)
    if projection is not None:
        s_h = projection(s_h.to(dtype=projection.weight.dtype))
    s_h = F.normalize(s_h.float(), p=2, dim=-1)
    t_h = F.normalize(t_h.float(), p=2, dim=-1)
    return F.mse_loss(s_h, t_h)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    epoch, loader, iters, teacher_model, lm_config_student,
    start_step=0, wandb=None,
    alpha=0.0, temperature=1.0,
    hidden_proj=None, hidden_distill_weight=0.0,
):
    start_time = time.time()

    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        loss_mask = (labels[..., 1:] != -100).float()
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ── Student forward ──────────────────────────────────────────────────
        with autocast_ctx:
            res = model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()

        # ── Teacher forward (eval / no_grad) ─────────────────────────────────
        teacher_out = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_out = teacher_model(input_ids)
                teacher_logits = teacher_out.logits[..., :-1, :].contiguous()
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ── Loss computation ─────────────────────────────────────────────────
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)

        # 1) Ground-truth CE loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none',
        )
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        ce_loss = (ce_loss_raw + res.aux_loss) if lm_config_student.use_moe else ce_loss_raw

        # 2) Logit-level distillation loss (KL divergence)
        if teacher_out is not None:
            kl_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature,
            )
        else:
            kl_loss = torch.tensor(0.0, device=args.device)

        # 3) Hidden-state distillation loss (transfers MoE expert knowledge)
        if hidden_distill_weight > 0.0 and teacher_out is not None:
            hd_loss = hidden_state_distillation_loss(
                res.hidden_states, teacher_out.hidden_states, projection=hidden_proj,
            )
        else:
            hd_loss = torch.tensor(0.0, device=args.device)

        # 4) Total loss
        loss = (alpha * ce_loss + (1 - alpha) * kl_loss + hidden_distill_weight * hd_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, kl: {kl_loss.item():.4f}, '
                f'hd: {hd_loss.item():.4f}, '
                f'learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min'
            )

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "kl_loss": kl_loss.item(),
                    "hd_loss": hd_loss.item(),
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config_student, weight=args.save_weight, model=model,
                optimizer=optimizer, scaler=scaler, epoch=epoch, step=step,
                wandb=wandb, save_dir='../checkpoints',
            )
            model.train()
            del state_dict

        del input_ids, labels, loss_mask, res, student_logits, teacher_out, \
            ce_loss, kl_loss, hd_loss, loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Expert Knowledge Distillation")
    # ── General ──────────────────────────────────────────────────────────────
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="expert_dist", type=str)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--data_path", type=str, default="../dataset/data/sft_en.jsonl")
    # ── Model architecture ────────────────────────────────────────────────────
    parser.add_argument("--student_hidden_size", default=512, type=int)
    parser.add_argument("--student_num_layers", default=8, type=int)
    parser.add_argument("--teacher_hidden_size", default=768, type=int)
    parser.add_argument("--teacher_num_layers", default=16, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--from_student_weight", default="full_sft", type=str)
    parser.add_argument("--from_teacher_weight", default="full_sft", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    # ── Distillation ──────────────────────────────────────────────────────────
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="CE weight; total = α·CE + (1-α)·KL + λ·HD")
    parser.add_argument("--temperature", default=1.5, type=float,
                        help="Distillation temperature (1.0–2.0 recommended)")
    parser.add_argument("--hidden_distill_weight", default=0.0, type=float,
                        help="λ for hidden-state distillation loss (0 = disabled); "
                             "transfers MoE expert knowledge into the dense student")
    # ── Dynamic expert pruning ────────────────────────────────────────────────
    parser.add_argument("--enable_dynamic_pruning", default=0, type=int, choices=[0, 1],
                        help="Enable dynamic expert pruning for the student at inference: "
                             "tokens with high gate confidence are routed to only 1 expert")
    parser.add_argument("--pruning_threshold", default=0.5, type=float,
                        help="Gate-score threshold above which a token is pruned to 1 expert "
                             "(e.g. 0.5 = moderate, 0.7 = aggressive)")
    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-ExpertDistillation")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Dataset file not found: {args.data_path}. "
            f"Try --data_path ../dataset/data/sft_en.jsonl"
        )

    # ── 1. Environment & seed ─────────────────────────────────────────────────
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ── 2. Directories & model configs ───────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config_student = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        use_moe=bool(args.use_moe),
        enable_dynamic_pruning=bool(args.enable_dynamic_pruning),
        pruning_threshold=args.pruning_threshold,
    )
    lm_config_teacher = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints')
        if args.from_resume == 1 else None
    )

    # ── 3. Mixed precision ────────────────────────────────────────────────────
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ── 4. Logging / wandb ────────────────────────────────────────────────────
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (
            f"ExpertDistill-S{args.student_hidden_size}T{args.teacher_hidden_size}"
            f"-Epoch{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ── 5. Models ─────────────────────────────────────────────────────────────
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'Student params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M  '
           f'(dynamic_pruning={lm_config_student.enable_dynamic_pruning}, '
           f'threshold={lm_config_student.pruning_threshold})')

    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'Teacher params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')

    # ── 6. Data ───────────────────────────────────────────────────────────────
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # ── 7. Optimizer (with optional hidden-state projection) ──────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    hidden_proj = None
    if args.hidden_distill_weight > 0.0 and args.student_hidden_size != args.teacher_hidden_size:
        hidden_proj = nn.Linear(args.student_hidden_size, args.teacher_hidden_size, bias=False)
        hidden_proj = hidden_proj.to(args.device).to(dtype)
        Logger(f'Hidden-state projection: {args.student_hidden_size} -> {args.teacher_hidden_size}')
    trainable_params = list(model.parameters()) + (
        list(hidden_proj.parameters()) if hidden_proj is not None else []
    )
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    # ── 8. Resume from checkpoint ─────────────────────────────────────────────
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ── 9. DDP wrap ───────────────────────────────────────────────────────────
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ── 10. Train ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: resuming from step {start_step + 1}')
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student,
                        start_step, wandb, args.alpha, args.temperature,
                        hidden_proj, args.hidden_distill_weight)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student,
                        0, wandb, args.alpha, args.temperature,
                        hidden_proj, args.hidden_distill_weight)

    # ── 11. Cleanup ───────────────────────────────────────────────────────────
    if dist.is_initialized():
        dist.destroy_process_group()
