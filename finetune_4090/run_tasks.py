"""Standalone single-GPU trainer for the finetune_4090 tasks.

Why not run.py?  run.py hardcodes torch.distributed.launch + NCCL + wandb and its
data class filters annotations by `audio/<id>.mp3` existence.  This script keeps
the same GRAM model, the same mappers and the same `evaluate_ret` evaluator, but
uses a plain PyTorch DataLoader (no DDP), works without wandb, and reports clear
loss/eval lines.

Run from the repo root after `python finetune_4090/make_cfgs.py ...`:

    python finetune_4090/run_tasks.py --task t1 \
        --config finetune_4090/cfgs/t1_didemo.json \
        --pretrain_dir /path/GRAM_pretrained_4modalities \
        --output_dir runs/t1_didemo \
        --train_batch_size 2 --epochs 3 --lr 3e-5 --eval_steps 500
"""

import argparse
import json
import math
import os
import random
import time


def _load_model_cfg(task_cfg_path, pretrain_dir):
    """Replicate utils/args.py merge rules for the *model* config."""
    from easydict import EasyDict as edict

    file_cfg = edict(json.load(open(task_cfg_path, encoding="utf-8")))
    model_cfg = edict(json.load(open(file_cfg.model_cfg.default, encoding="utf-8")))
    model_cfg.update(file_cfg.get("model_cfg", {}))
    if pretrain_dir:
        hps = json.load(open(os.path.join(pretrain_dir, "log", "hps.json"), encoding="utf-8"))
        pmc = edict(hps["model_cfg"])
        inherit = set(["vision_encoder_type", "pool_video"]) | set(model_cfg.inherit_keys)
        model_cfg.update({k: v for k, v in pmc.items() if k in inherit})
    return model_cfg


class _MapperArgs:
    def __init__(self, model_cfg, training):
        self.model_cfg = model_cfg
        self.training = training


class RetVideoAudioTxtDataset:
    """video(+audio)+text rows for GRAM retrieval.

    Mirrors data/IndexAnno.py's row semantics without the mp3-only hard filter:
    a row is kept when the video file exists; audio is loaded if present and
    AudioMapper already returns an all-zero spectrogram otherwise.
    """

    def __init__(self, d_cfg, model_cfg):
        from easydict import EasyDict as edict
        from data.vision_mapper import VisionMapper
        from data.audio_mapper import AudioMapper

        d = edict(dict(d_cfg))
        self.vision_mapper = VisionMapper(d, _MapperArgs(model_cfg, bool(d.training)))
        self.audio_mapper = AudioMapper(d, _MapperArgs(model_cfg, bool(d.training)))
        self.vision_dir = d_cfg["vision"]
        self.use_subtitles = "s" in str(d_cfg.get("task", "")).split("%")[-1]

        annos = json.load(open(d_cfg["txt"], encoding="utf-8"))
        self.rows = []
        for a in annos:
            vid = a.get("video_id") or a.get("id")
            if vid is None:
                continue
            if not os.path.exists(self._video_path(vid)):
                continue
            desc = a.get("desc") or a.get("caption")
            if isinstance(desc, (list, tuple)):
                # Official multi-caption test rows: one row per caption, so the
                # same video appears once per description (repo loader does the
                # same expansion for `raw_captions` during evaluation).
                for cap in desc:
                    self.rows.append({**a, "desc": cap})
            else:
                self.rows.append(a)
        print(f"[data] {d_cfg['name']}: kept {len(self.rows)}/{len(annos)} rows")
        if not self.rows:
            raise RuntimeError(
                f"{d_cfg['name']}: 0 rows with videos under '{self.vision_dir}'. "
                "Check --video-dir contents / annotation video_id names.")

    def _video_path(self, vid):
        # One scan per dataset, then membership checks (repo's
        # VisionMapper.check_extension globs the whole directory per frame-read,
        # which is far too slow for >1k videos).
        if not hasattr(self, "_file_idx"):
            self._file_idx = set(os.listdir(self.vision_dir))
        for ext in (".mp4", ".avi", ".webm", ".mkv", ".mov"):
            if str(vid) + ext in self._file_idx:
                return os.path.join(self.vision_dir, str(vid) + ext)
        return os.path.join(self.vision_dir, str(vid) + ".mp4")

    def _read_row(self, anno, i):
        vid = anno.get("video_id") or anno["id"]
        desc = anno.get("desc") or anno.get("caption")
        video, _ = self.vision_mapper.read(vid)
        if video is None:
            return None
        audio = self.audio_mapper.read(vid)
        if audio is None:
            import torch
            audio = torch.zeros(
                (int(self.audio_mapper.sample_num),
                 int(self.audio_mapper.target_length),
                 int(self.audio_mapper.melbins)), dtype=torch.float32)
        row = {"id": vid, "caption": desc, "video": video, "audio": audio}
        if "subtitle" in anno and self.use_subtitles:
            row["subtitle"] = anno.get("subtitle", "")
        return row

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        anno = self.rows[i]
        out = self._read_row(anno, i)
        if out is None:
            return self.__getitem__(random.randrange(len(self.rows)))
        return out


def _collate(items):
    import torch

    batch = {}
    first = items[0]
    for key in first:
        vals = [it[key] for it in items]
        if key in ("id", "caption", "subtitle"):
            batch[key] = vals
        else:
            batch[key] = torch.stack(vals, dim=0).float()
    return batch


def _forward_ret_volume_only(self, batch, task, compute_loss=True):
    """Single-GPU-friendly forward_ret used during *training*.

    Identical objective to `GRAM.forward_ret` (volume contrastive loss over
    text/video/audio) but without the ITM hard-negative branch, which costs an
    extra BERT cross-attention forward and makes 24 GB cards OOM.  Evaluation
    still uses the repository's native `forward_ret(compute_loss=False)`.
    """
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from easydict import EasyDict as edict
    from utils.distributed import concat_all_gather
    from utils.volume import volume_computation3

    batch = edict(batch)
    if not compute_loss:
        orig = getattr(self, "_original_forward_ret", None)
        if orig is None:
            orig = self._original_forward_ret = self.__class__.forward_ret
        return orig(self, batch, task, compute_loss=False)

    feat_t = self.batch_get(batch, "feat_t")
    feat_v = self.batch_get(batch, "feat_v")
    feat_a = self.batch_get(batch, "feat_a")
    feat_t_all = concat_all_gather(feat_t)
    feat_v_all = concat_all_gather(feat_v)
    feat_a_all = concat_all_gather(feat_a)

    volume = volume_computation3(feat_t, feat_v_all, feat_a_all) / self.contra_temp
    volumeT = volume_computation3(feat_t_all, feat_v, feat_a).T / self.contra_temp
    rank = dist.get_rank()
    bs = feat_t.size(0)
    targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs,
                             dtype=int, device=volume.device)
    loss_area = (
        F.cross_entropy(-volume, targets, label_smoothing=0.1)
        + F.cross_entropy(-volumeT, targets, label_smoothing=0.1)
    ) / 2
    return {"loss_area": loss_area,
            "loss_itc": volume.sum() * 0.0,
            "loss_itm": volume.sum() * 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["t1", "t2", "t3"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--pretrain_dir", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_batch_size", type=int, default=2,
                    help="volume contrastive needs >=2 samples for gradients")
    ap.add_argument("--val_batch_size", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_train_steps", type=int, default=0,
                    help="debug/smoke: hard cap the number of gradient steps "
                         "(0 = auto from epochs)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.add_argument("--first_eval", action="store_true")
    ap.add_argument("--eval_steps", type=int, default=2000)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--log_steps", type=int, default=20)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32.0)
    ap.add_argument("--unfreeze_vision_blocks", type=int, default=4)
    ap.add_argument("--unfreeze_audio_layers", type=int, default=2)
    args = ap.parse_args()
    if args.train_batch_size < 2:
        raise SystemExit("volume-contrastive loss needs --train_batch_size >= 2 "
                         "(batch=1 gives no negatives / ~zero gradient)")

    import torch
    from torch.utils.data import DataLoader
    import torch.distributed as dist

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    if not dist.is_initialized():
        # world_size=1: init only so the repo's all_gather helpers no-op.
        # gloo keeps this working on Windows dev boxes; NCCL users can switch.
        if device.type == "cuda":
            try:
                dist.init_process_group(backend="nccl", world_size=1, rank=0)
            except Exception as e:
                print(f"[warn] NCCL init failed ({e}); using gloo. "
                      "eval all-gather may need NCCL.")
                dist.init_process_group(backend="gloo", world_size=1, rank=0)
        else:
            dist.init_process_group(backend="gloo", world_size=1, rank=0)

    # ---------- model config + pretrained weights ---------------------------
    model_cfg = _load_model_cfg(args.config, args.pretrain_dir)
    file_cfg = json.load(open(args.config, encoding="utf-8"))
    d_train = file_cfg["data_cfg"]["train"][0]
    d_val = file_cfg["data_cfg"]["val"][0]

    from model import model_registry
    model = model_registry[model_cfg.model_type](model_cfg).to(device)
    if args.pretrain_dir:
        ckpt_file = _latest_ckpt(args.pretrain_dir)
        print("[ckpt]", ckpt_file)
        ckpt = torch.load(ckpt_file, map_location="cpu")
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        ckpt = model.modify_checkpoint(ckpt)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print("[ckpt] missing", len(missing), "unexpected", len(unexpected))
        del ckpt

    from finetune_4090.peft_utils import apply_peft
    apply_peft(
        model,
        task=args.task,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        unfreeze_vision_blocks=args.unfreeze_vision_blocks,
        unfreeze_audio_layers=args.unfreeze_audio_layers,
    )
    # Training uses volume-only forward_ret (no ITM branch); evaluation still
    # uses the repository's native compute_loss=False forward_ret.
    import model.gram as gram_mod
    model._original_forward_ret = gram_mod.GRAM.forward_ret
    gram_mod.GRAM.forward_ret = _forward_ret_volume_only

    # ---------- datasets -----------------------------------------------------
    train_ds = RetVideoAudioTxtDataset(d_train, model_cfg)
    val_ds = RetVideoAudioTxtDataset(d_val, model_cfg)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size,
                              shuffle=True, num_workers=d_train.get("n_workers", 2),
                              collate_fn=_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=d_val.get("n_workers", 2),
                            collate_fn=_collate, drop_last=False)

    # ---------- optimizer ----------------------------------------------------
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    decay_p, nodecay_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay_p if any(s in n for s in no_decay) else decay_p).append(p)
    opt = torch.optim.AdamW([
        {"params": decay_p, "weight_decay": args.weight_decay},
        {"params": nodecay_p, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.98))
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "log"), exist_ok=True)

    # ---------- helpers ------------------------------------------------------
    def to_device(batch):
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        return out

    from evaluation.evaluation_mm import evaluate_ret

    def run_eval(tag):
        model.eval()
        with torch.no_grad():
            try:
                log = evaluate_ret(model, d_val["task"], val_loader, tag)
            except Exception as e:
                print(f"[warn] evaluate_ret failed at step {tag}: {e}")
                log = {}
        model.train()
        rep = None
        for k, v in log.items():
            if isinstance(v, dict) and "forward_r1" in v:
                rep = (k, v["forward_r1"])
                break
        print(f"\n[EVAL {tag}] " + json.dumps(log, default=str)[:1800] + "\n")
        return rep

    if args.first_eval:
        run_eval(0)

    # ---------- training loop -----------------------------------------------
    gstep = 0
    best = -1.0
    t0 = time.time()
    for epoch in range(max(1, math.ceil(args.epochs))):
        frac = min(1.0, max(0.0, args.epochs - epoch))
        if frac <= 0:
            break
        epoch_steps = min(steps_per_epoch, max(1, int(math.ceil(frac * steps_per_epoch))))
        for bi, batch in enumerate(train_loader):
            if bi >= epoch_steps:
                break
            if "video" not in batch or "audio" not in batch:
                continue
            batch = to_device(batch)
            model_batch = {
                "raw_captions": batch["caption"],
                "vision_pixels": batch["video"],
                "audio_spectrograms": batch["audio"],
                "ids": batch["id"],
                "ids_txt": batch["id"],
            }
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                out = model(model_batch, task=d_train["task"], compute_loss=True)
                loss = sum(out.values())

            scaler.scale(loss).backward()
            gstep += 1
            progress = min(1.0, gstep / total_steps)
            if progress < warmup_steps / total_steps:
                lr_ratio = progress * total_steps / max(1, warmup_steps)
            else:
                lr_ratio = max(
                    0.0, (1.0 - progress) / (1.0 - warmup_steps / total_steps))
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_ratio
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None], 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if gstep % args.log_steps == 0 or bi == 0:
                lv = {k: round(float(v.detach().cpu()), 4) for k, v in out.items()}
                print(f"[{gstep}/{total_steps}] ep{epoch} {lv} "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"{(time.time()-t0)/60:.1f}min")

            if gstep % args.eval_steps == 0:
                rep = run_eval(gstep)
                if rep and rep[1] > best:
                    best = rep[1]
                    torch.save(model.state_dict(),
                               os.path.join(args.output_dir, "ckpt", "best_ret.pt"))
            if gstep % args.save_steps == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "ckpt", f"model_step_{gstep}.pt"))
            if gstep >= total_steps:
                break
        if gstep >= total_steps:
            break

    torch.save(model.state_dict(), os.path.join(args.output_dir, "ckpt", "final.pt"))
    print("done:", os.path.join(args.output_dir, "ckpt", "final.pt"))


def _latest_ckpt(pretrain_dir):
    ckpt_dir = os.path.join(pretrain_dir, "ckpt")
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("model_step") and f.endswith(".pt")]
    files.sort(key=lambda f: int(f.split("_")[2].split(".")[0]))
    return os.path.join(ckpt_dir, files[-1])


if __name__ == "__main__":
    main()
