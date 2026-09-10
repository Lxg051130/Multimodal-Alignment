"""Small, dependency-free LoRA / freeze / unfreeze helpers for GRAM on a single GPU.

No repo code is modified: these helpers are applied after `build_model` returns
the (already pretrained) GRAM model, so the native state_dict keys stay
compatible with `utils/save.py` / checkpointing used by the training loop.
"""

import torch
import torch.nn as nn


class _LoRALinear(nn.Module):
    """Wrap an nn.Linear: keep the original frozen projection and add a LoRA bypass.

    `self.original` keeps the *untouched* Linear (weight detached, requires_grad
    False). `self.lora_a/b` are the only trainable tensors of this module.
    State-dict keys of the parent model do not change (the sub-module is replaced
    in place), which keeps native load/save working.
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        in_f, out_f = linear.in_features, linear.out_features
        self.original = linear
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0
        self.lora_a = nn.Parameter(torch.zeros(self.r, in_f))
        self.lora_b = nn.Parameter(torch.zeros(out_f, self.r))
        # "zero init" (LoRA paper): B=0 -> output equals original projection.
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        out = self.original(x)
        if self.r > 0:
            out = out + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling
        return out


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


def set_trainable(module: nn.Module, trainable: bool = True):
    for p in module.parameters():
        p.requires_grad_(trainable)


def add_lora_to_module(module: nn.Module, r: int = 8, alpha: float = 16.0,
                       target_suffix: tuple = ("weight",),
                       include_names: tuple = ()) -> int:
    """Replace every nn.Linear in `module` with a LoRA bypass.

    `include_names` is a whitelist of sub-string matchers on dotted parameter
    names (e.g. ("attention.self", "intermediate", "output.dense")).
    Returns the number of injected adapters.
    """
    n_injected = 0

    def _dot_name(parent_name: str, child_name: str):
        return f"{parent_name}.{child_name}" if parent_name else child_name

    def _walk(mod: nn.Module, prefix: str = ""):
        nonlocal n_injected
        for child_name, child in list(mod.named_children()):
            full = _dot_name(prefix, child_name)
            if isinstance(child, nn.Linear):
                hit = (not include_names) or any(s in full for s in include_names)
                if hit:
                    mod._modules[child_name] = _LoRALinear(child, r=r, alpha=alpha)
                    n_injected += 1
                    continue
            if len(list(child.children())) > 0:
                _walk(child, full)

    _walk(module)
    return n_injected


def unfreeze_vision_tail(model, n_blocks: int = 4):
    """Unfreeze the last `n_blocks` EVAVisionTransformer blocks of GRAM's CLIP."""
    try:
        blocks = model.vision_encoder.visual.blocks
    except AttributeError:
        print("[peft_utils] vision encoder layout not recognised; vision kept frozen")
        return 0
    n = len(blocks)
    k = min(int(n_blocks), n)
    for i in range(n - k, n):
        set_trainable(blocks[i], True)
        blocks[i].train()  # unfreeze switches this tail back to train mode
    print(f"[peft_utils] unfroze {k}/{n} vision blocks")
    return k


def unfreeze_audio_tail(model, n_layers: int = 2):
    """Unfreeze the last `n_layers` BEATs transformer layers."""
    try:
        layers = model.audio_encoder.layers
    except AttributeError:
        try:
            layers = model.audio_encoder.encoder.layers
        except AttributeError:
            print("[peft_utils] audio encoder layout not recognised; audio kept frozen")
            return 0
    n = len(layers)
    k = min(int(n_layers), n)
    for i in range(n - k, n):
        set_trainable(layers[i], True)
        layers[i].train()
    print(f"[peft_utils] unfroze {k}/{n} audio layers")
    return k


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[peft_utils] trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M params")
    return trainable, total


def apply_peft(model, task: str, lora_r: int = 0, lora_alpha: float = 16.0,
               lora_targets: tuple = ("attention.self", "intermediate", "output.dense"),
               unfreeze_vision_blocks: int = 0, unfreeze_audio_layers: int = 0):
    """Route PEFT setup by task (see README_TASKS.md section 4).

    - Backbones (EVA-CLIP visual tower + BEATs) are always frozen first.
    - 't1': train the text/fusion BERT + all small heads on top of frozen features.
    - 't2': 't1' + LoRA adapters inside BERT's q/k/v/out & FFN.
    - 't3': 't2' + unfreeze the tail blocks/layers of vision & audio backbones.
    """
    from model.general_module import disabled_train

    # --- vision/audio backbones -------------------------------------------
    if hasattr(model, "vision_encoder"):
        freeze_module(model.vision_encoder)
        model.vision_encoder.train = disabled_train
    if hasattr(model, "audio_encoder"):
        freeze_module(model.audio_encoder)
        model.audio_encoder.train = disabled_train

    # Keep the backbone outputs from ever flowing into trainable modules, so
    # autograd does not have to retain backbone activations.  For T1/T2 the
    # contrastive video/audio projection heads + hidden transforms are frozen;
    # T3 re-enables them below together with the corresponding tail layers.
    # Exact module-name set (direct children of GRAM).
    va_names = {
        "contra_head_v", "contra_head_a", "contra_head_va", "contra_head_d",
        "contra_head_vas", "contra_head_vs",
        "hidden_trans_vision_multimodal", "hidden_trans_audio_multimodal",
    }
    for n, m in model.named_modules():
        if n in va_names:
            freeze_module(m)

    # T1: BERT(text) + text/projection heads are trainable; backbone-side
    # projections stay frozen -> no gradient path into EVA/BEATs.
    if task == "t1":
        count_trainable(model)
        return model

    # T2/T3: LoRA adapters inside the trainable BERT (q/k/v, intermediate, out).
    if task in ("t2", "t3"):
        n = add_lora_to_module(
            model.multimodal_encoder, r=lora_r, alpha=lora_alpha,
            include_names=lora_targets)
        print(f"[peft_utils] injected {n} LoRA adapters into multimodal_encoder")
        # Base BERT stays trainable; the adapters add capacity on top. Original
        # weights inside the wrappers were detached -> base part stays frozen
        # inside those layers, while embeddings/norms/heads keep training.
        count_trainable(model)
        if task == "t2":
            return model
        unfreeze_vision_tail(model, unfreeze_vision_blocks)
        unfreeze_audio_tail(model, unfreeze_audio_layers)
        # Re-enable the small projections that now have to route gradients
        # into the unfrozen tails (harmless even if tail=0).
        for n, m in model.named_modules():
            if n in va_names:
                set_trainable(m, True)
                m.train()
        count_trainable(model)
        return model

    raise ValueError(f"Unknown task {task!r}")
