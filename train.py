"""
train.py — LEAP-VO Training Loop
=================================
Trains LeapKernel on TAP-Vid-Kubric (MOVI-F), warm-started from pretrained
CoTracker v1 fnet per paper Sec. 4.2.
 
Loss structure (Eq. 8 in paper):
    L_total = w1 * L_main + w2 * L_vis + w3 * L_dyn
 
    L_main : Cauchy NLL on trajectory + covariance  (FIXED — never touch this)
    L_vis  : visibility classification              (bce | gce | trunc_gce)
    L_dyn  : dynamic object classification          (bce | gce | trunc_gce)
 
=============================================================================
PAPER vs CURRENT RUN — settings to change when moving to A100
=============================================================================
                       paper        current (RTX 4090)    notes
    GPU                4×A100       1×RTX 4090 16GB
    Batch (eff.)       4            1 (set accum=4 on A100)  B==1 hardcoded
    --grad_accum_steps  -           1      (set to 4 on A100)
    --num_tracks       256          128    (OOMs at 256 on 4090)
    --num_anchors      64           64                        ✓
    --num_steps        100000       50000  (set to 100k on A100)
    Resolution         unspecified* 256×256  (TAP-Vid eval resolution)
    Dataset size       5737 seqs    2000 seqs
    fnet init          CoTracker v1 CoTracker v1  ✓ (via --pretrained_fnet)
    Optimizer          AdamW        AdamW        ✓
    Scheduler          OneCycleLR   OneCycleLR   ✓
    LR                 5e-4         5e-4         ✓
    γ (iter decay)     0.8          0.8          ✓
    w1, w2, w3         1.0,0.5,0.5  1.0,0.5,0.5  ✓
    * paper says "follow CoTracker"; CoTracker defaults to [384, 512] crops.
 
=============================================================================
 
Usage:
    # Sanity check — no pretrained fnet (will diverge, for reference only)
    python train.py --use_dummy --run_name smoke_test --num_steps 10
 
    # Paper-faithful baseline (warm-started fnet)
    python train.py --loss_vis bce --loss_dyn bce --run_name baseline \\
        --pretrained_fnet ~/checkpoints/cotracker_stride_4_wind_8.pth
 
    # GCE on both classification heads
    python train.py --loss_vis gce --loss_dyn gce --run_name gce_both \\
        --pretrained_fnet ~/checkpoints/cotracker_stride_4_wind_8.pth
 
    # Truncated GCE on both
    python train.py --loss_vis trunc_gce --loss_dyn trunc_gce --run_name trunc_both \\
        --pretrained_fnet ~/checkpoints/cotracker_stride_4_wind_8.pth
 
    # Full paper setup on A100
    python train.py --grad_accum_steps 4 --num_tracks 256 --num_steps 100000 \\
        --run_name baseline_a100 \\
        --pretrained_fnet ~/checkpoints/cotracker_stride_4_wind_8.pth
 
Key facts confirmed from reading the actual repo:
    1. LeapKernel(cfg, stride=4) — cfg is OmegaConf, field names from demo.yaml
    2. B == 1 is hardcoded (assert B==1). Use --grad_accum_steps for larger eff. batch.
    3. forward() returns 6-tuple:
         (traj_e, feat_init, vis_e, (cov_x_e, cov_y_e), dynamic_e, train_data)
    4. train_data is a tuple of 6:
         (vis_predictions, coord_predictions, dynamic_predictions,
          cov_predictions, wind_inds, sort_inds)
    5. vis_predictions[w]  = [B, S_local, N_w]   — ALREADY sigmoid-ed
    6. dyn_predictions[w]  = [B, S_local, N_w]   — ALREADY sigmoid-ed
    7. coord_predictions[w] = list[K] of [B, S_local, N_w, 2]  — pixel coords
    8. cov_predictions[w]   = list[K] of [Sx, Sy], each [B, N_w, S_local, S_local]
    9. wind_inds[w]          = scalar tensor (number of active tracks in window w)
   10. sort_inds             = [N] — track permutation; MUST be applied to gt
   11. MotionLabelBlock cfg needs: mode, in_dim, hidden_dim, S (from demo.yaml)
   12. kernel_block composition in demo.yaml is "sum" (not "product")
   13. anchor_aug needs: num_anchors, anchor_mode, margin, frame fields
"""
 
import csv
import gc
import json
import logging
import random
import sys
import time
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
 
# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LEAP_SRC = SCRIPT_DIR / "main"
sys.path.insert(0, str(LEAP_SRC))
 
from leap.leap_kernel import LeapKernel  # noqa: E402
 
# ---------------------------------------------------------------------------
# Optional imports — losses.py and dataloader.py written separately
# ---------------------------------------------------------------------------
try:
    from losses import cauchy_nll_loss, bce_loss, gce_loss, truncated_gce_loss
    LOSSES_AVAILABLE = True
except ImportError:
    LOSSES_AVAILABLE = False
    logging.warning("losses.py not found — using stub losses.")
 
try:
    from dataloader import LeapKubricDataset
    DATALOADER_AVAILABLE = True
except ImportError:
    DATALOADER_AVAILABLE = False
    logging.warning("dataloader.py not found — using DummyDataset.")
 
 
# ===========================================================================
# Config builder — field names taken directly from demo.yaml
# ===========================================================================
 
def build_leap_cfg(args):
    """
    Build the OmegaConf config that LeapKernel.__init__ reads.
    Every field name here is verified against demo.yaml and leap_kernel.py.
    """
    cfg = OmegaConf.create({
        "model": {
            # Core dims (from demo.yaml)
            "sliding_window_len": args.window_size,   # S=8
            "hidden_dim":         256,
            "latent_dim":         128,
            "hidden_size":        384,
 
            # Transformer depth
            "space_depth":    6,
            "time_depth":     6,
            "num_heads":      8,
 
            # Correlation
            "corr_levels":    4,
            "corr_radius":    3,
 
            # Attention flag
            "add_space_attn": True,
 
            # Kernel block — composition="sum" matches demo.yaml
            "kernel_from_delta": True,
            "kernel_block": {
                "kernel_list":  ["linear"],
                "composition":  "sum",       # demo.yaml uses "sum"
                "add_time":     False,
            },
 
            # S for KernelBlock (RBFKernel uses cfg.S for input_dim)
            "S": args.window_size,
 
            # MotionLabelBlock — mode/in_dim/hidden_dim/S from demo.yaml
            "motion_label_block": {
                "mode":       "mlp_v1",   # matches demo.yaml
                "in_dim":     128,        # = latent_dim
                "hidden_dim": 256,
                "S":          args.window_size,
            },
        },
 
        # anchor_aug fields read by get_anchors() in anchor_sampler.py:
        #   cfg.anchor_mode, cfg.num_anchors, cfg.margin, cfg.frame
        "anchor_aug": OmegaConf.create({
            "anchor_mode": "uniform",   # uniform grid — stable for training
            "num_anchors":  args.num_anchors,
            "margin":       64,
            "frame":        0,
        }) if args.num_anchors > 0 else None,
    })
    return cfg
 
 
# ===========================================================================
# Stub losses — only used until losses.py is written
# ===========================================================================
 
def _stub_cauchy_nll(coord_pred, coord_gt, sigma_x, sigma_y, valid_mask):
    """
    Minimal stand-in for Cauchy NLL. Shapes:
        coord_pred / coord_gt : [B, S, N, 2]
        sigma_x / sigma_y     : [B, N, S, S]
        valid_mask            : [B, S, N]  bool
    """
    diff = (coord_pred - coord_gt) * valid_mask.float().unsqueeze(-1)
    return diff.abs().mean()
 
 
def _stub_bce_prob(probs, targets, valid_mask=None):
    """BCE on already-sigmoid-ed probabilities."""
    eps   = 1e-6
    p     = probs.clamp(eps, 1 - eps)
    t     = targets.float()
    loss  = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
    if valid_mask is not None:
        loss = loss * valid_mask.float()
    return loss.mean()
 
 
# ===========================================================================
# Loss dispatchers
# NOTE: vis and dyn predictions come out of LeapKernel already sigmoid-ed.
#       losses.py must handle probabilities, not logits (input_is_prob=True).
# ===========================================================================
 
def compute_vis_loss(probs, targets, valid_mask, loss_type, gce_q, trunc_frac):
    """
    probs      : [B, S, N]  already sigmoid-ed
    targets    : [B, S, N]  bool — True = visible
    valid_mask : [B, S, N]  bool
    """
    if not LOSSES_AVAILABLE:
        return _stub_bce_prob(probs, targets, valid_mask)
    if loss_type == "bce":
        return bce_loss(probs, targets, valid_mask, input_is_prob=True)
    elif loss_type == "gce":
        return gce_loss(probs, targets, valid_mask, q=gce_q, input_is_prob=True)
    elif loss_type == "trunc_gce":
        return truncated_gce_loss(probs, targets, valid_mask, q=gce_q,
                                  trunc_frac=trunc_frac, input_is_prob=True)
    raise ValueError(f"Unknown loss_type: {loss_type}")
 
 
def compute_dyn_loss(probs, targets, valid_mask, loss_type, gce_q, trunc_frac):
    """
    probs      : [B, S, N]  already sigmoid-ed, repeated across S frames
    targets    : [B, N]     bool — True = dynamic (per-track label)
    valid_mask : [B, S, N]  bool
    Dynamic label is per-track — we broadcast it to [B, S, N] to match probs.
    """
    targets_exp = targets.unsqueeze(1).expand_as(probs)   # [B, S, N]
    if not LOSSES_AVAILABLE:
        return _stub_bce_prob(probs, targets_exp, valid_mask)
    if loss_type == "bce":
        return bce_loss(probs, targets_exp, valid_mask, input_is_prob=True)
    elif loss_type == "gce":
        return gce_loss(probs, targets_exp, valid_mask, q=gce_q, input_is_prob=True)
    elif loss_type == "trunc_gce":
        return truncated_gce_loss(probs, targets_exp, valid_mask, q=gce_q,
                                  trunc_frac=trunc_frac, input_is_prob=True)
    raise ValueError(f"Unknown loss_type: {loss_type}")
 
 
# ===========================================================================
# Ground-truth window slicer
# ===========================================================================
 
def build_gt_for_window(coords_gt, vis_gt, is_dyn_gt,
                        ind, S_local, wind_idx, sort_inds, device):
    """
    Slice ground-truth to match one window's model predictions.
 
    LeapKernel internally sorts all N tracks by their first visible frame
    (sort_inds). Predictions are produced in that sorted order up to wind_idx
    active tracks. We must apply the same permutation to gt before comparing.
 
    Args:
        coords_gt  : [B, T, N, 2]
        vis_gt     : [B, T, N]  bool
        is_dyn_gt  : [B, N]     bool
        ind        : window start frame index
        S_local    : actual frames in this window (may be < S at seq end)
        wind_idx   : number of active tracks  (scalar int)
        sort_inds  : [N] permutation tensor from LeapKernel
        device     : torch device
 
    Returns:
        gt_coord   : [B, S_local, wind_idx, 2]
        gt_vis     : [B, S_local, wind_idx]  bool
        gt_dyn     : [B, wind_idx]           bool
        valid_mask : [B, S_local, wind_idx]  bool  (ones for now)
    """
    wi = int(wind_idx)
 
    # Apply the same sort permutation LeapKernel used
    c = coords_gt[:, :, sort_inds]   # [B, T, N, 2]
    v = vis_gt[:, :, sort_inds]      # [B, T, N]
    d = is_dyn_gt[:, sort_inds]      # [B, N]
 
    gt_coord = c[:, ind:ind + S_local, :wi].to(device)   # [B, S_local, Nw, 2]
    gt_vis   = v[:, ind:ind + S_local, :wi].to(device)   # [B, S_local, Nw]
    gt_dyn   = d[:, :wi].to(device)                       # [B, Nw]
 
    valid_mask = torch.ones_like(gt_vis, dtype=torch.bool)
 
    return gt_coord, gt_vis, gt_dyn, valid_mask
 
 
# ===========================================================================
# Core loss computation
# ===========================================================================
 
def compute_batch_loss(train_data, coords_gt, vis_gt, is_dyn_gt, args, device):
    """
    Unpack LeapKernel's train_data tuple and compute L_total.
 
    train_data layout (confirmed from leap_kernel.py lines 536-547):
        [0] vis_predictions    : list[W] — each [B, S_local, N_w]         sigmoid-ed
        [1] coord_predictions  : list[W] — each list[K] of [B, S_local, N_w, 2]
        [2] dynamic_predictions: list[W] — each [B, S_local, N_w]         sigmoid-ed
        [3] cov_predictions    : list[W] — each list[K] of [Sx, Sy]
                                 Sx/Sy = [B, N_w, S_local, S_local]
        [4] wind_inds          : list[W] — each scalar tensor
        [5] sort_inds          : [N] permutation
 
    Returns: (total, l_main, l_vis, l_dyn, diagnostics_dict)
      diagnostics_dict carries the log-det / Mahalanobis / sigma-diag / delta-abs
      breakdown of L_main averaged across windows and iterations.
    """
    vis_preds, coord_preds, dyn_preds, cov_preds, wind_inds, sort_inds = train_data
 
    S     = args.window_size
    K     = args.train_iters
    gamma = 0.8       # iteration decay from Eq. 5 (paper fixed value)
    W     = len(coord_preds)
 
    total_main = torch.tensor(0.0, device=device)
    total_vis  = torch.tensor(0.0, device=device)
    total_dyn  = torch.tensor(0.0, device=device)
 
    # Diagnostic accumulators (floats). Averaged across (window × iter) at the end.
    diag_sums = {
        "log_det_x": 0.0, "log_det_y": 0.0,
        "mahal_x":   0.0, "mahal_y":   0.0,
        "sigma_diag_x": 0.0, "sigma_diag_y": 0.0,
        "delta_abs":    0.0,
    }
    diag_count = 0
 
    ind = 0   # window start frame index; advances by S//2 per window
 
    for w in range(W):
        wind_idx = wind_inds[w]
        S_local  = coord_preds[w][0].shape[1]   # actual frames this window
 
        gt_coord, gt_vis, gt_dyn, valid_mask = build_gt_for_window(
            coords_gt, vis_gt, is_dyn_gt,
            ind, S_local, wind_idx, sort_inds, device
        )
 
        # ---- L_main (Eq. 5): all K iterations with exponential decay ----
        for k in range(len(coord_preds[w])):
            # Final iteration (k = K-1) gets weight gamma^0 = 1.0 (highest)
            weight  = gamma ** (K - 1 - k)
            pred_c  = coord_preds[w][k]     # [B, S_local, Nw, 2]
            sigma_x = cov_preds[w][k][0]    # [B, Nw, S_local, S_local]
            sigma_y = cov_preds[w][k][1]    # [B, Nw, S_local, S_local]
 
            if LOSSES_AVAILABLE:
                l_k, d_k = cauchy_nll_loss(
                    pred_c, gt_coord, sigma_x, sigma_y,
                    valid_mask, stride=4, return_diagnostics=True,
                )
                for key in diag_sums:
                    diag_sums[key] += d_k[key]
                diag_count += 1
            else:
                l_k = _stub_cauchy_nll(pred_c, gt_coord, sigma_x, sigma_y, valid_mask)
 
            total_main = total_main + weight * l_k
 
        # ---- L_vis: one sigmoid-ed prediction per window ----
        total_vis = total_vis + compute_vis_loss(
            vis_preds[w], gt_vis, valid_mask,
            args.loss_vis, args.gce_q, args.trunc_frac
        )
 
        # ---- L_dyn: sigmoid-ed, repeated across S frames ----
        total_dyn = total_dyn + compute_dyn_loss(
            dyn_preds[w], gt_dyn, valid_mask,
            args.loss_dyn, args.gce_q, args.trunc_frac
        )
 
        ind += S // 2   # stride = S // 2
 
    # Average across windows
    total_main = total_main / max(W, 1)
    total_vis  = total_vis  / max(W, 1)
    total_dyn  = total_dyn  / max(W, 1)
 
    # Average diagnostics across (windows × iterations)
    if diag_count > 0:
        diagnostics = {k: v / diag_count for k, v in diag_sums.items()}
    else:
        diagnostics = {k: 0.0 for k in diag_sums}
 
    # Eq. 8 weighted sum — paper values: w1=1.0, w2=0.5, w3=0.5
    total = args.w1 * total_main + args.w2 * total_vis + args.w3 * total_dyn
 
    return total, total_main, total_vis, total_dyn, diagnostics
 
 
# ===========================================================================
# Dummy dataset — smoke-test without real data
# ===========================================================================
 
class DummyDataset(torch.utils.data.Dataset):
    """
    Synthetic data with exact shapes LeapKernel expects.
    Use with --use_dummy to verify the full training loop runs end-to-end.
    """
    def __init__(self, length=100, T=24, N=256, H=256, W=256):
        self.length = length
        self.T, self.N, self.H, self.W = T, N, H, W
 
    def __len__(self):
        return self.length
 
    def __getitem__(self, idx):
        T, N, H, W = self.T, self.N, self.H, self.W
        rgbs       = torch.randint(0, 256, (T, 3, H, W), dtype=torch.float32)
        coords     = torch.rand(T, N, 2) * torch.tensor([W, H], dtype=torch.float32)
        visibility = torch.ones(T, N, dtype=torch.bool)
        is_dynamic = torch.zeros(N, dtype=torch.bool)
        # queries: [N, 3] = [frame_idx, x, y] — all queried from frame 0
        queries    = torch.cat([torch.zeros(N, 1), coords[0]], dim=1)
        return {
            "rgbs":       rgbs,        # [T, 3, H, W]
            "coords":     coords,      # [T, N, 2]
            "visibility": visibility,  # [T, N]
            "is_dynamic": is_dynamic,  # [N]
            "queries":    queries,     # [N, 3]
        }
 
 
def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
 
 
# ===========================================================================
# Optimizer & Scheduler
# ===========================================================================
 
def build_optimizer(model, args):
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="cos",
    )
    return optimizer, scheduler
 
 
# ===========================================================================
# Checkpoint helpers
# ===========================================================================
 
def save_checkpoint(model, optimizer, scheduler, step, run_dir, tag="latest"):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"leap_{tag}.pth"
    torch.save({
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "total_steps": step,
    }, path)
    logging.info(f"Saved → {path}")
 
 
def load_checkpoint(model, optimizer, scheduler, run_dir):
    run_dir = Path(run_dir)
    candidates = sorted(run_dir.glob("leap_*.pth"))
    if not candidates:
        logging.info("No checkpoint found — starting from scratch.")
        return 0
    # Prefer the latest numbered checkpoint over 'final'
    numbered = [c for c in candidates if c.stem != "leap_final"]
    path = numbered[-1] if numbered else candidates[-1]
    logging.info(f"Resuming from {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("total_steps", 0)
 
 
def load_pretrained_fnet(model, ckpt_path):
    """
    Warm-start the fnet (feature extractor) from a pretrained CoTracker v1
    or LEAP checkpoint. The paper (Sec. 4.2) does this, and it's essential for
    stable Cauchy NLL training — random fnet features cause Σ to oscillate
    and the loss to go wildly negative.
 
    Robust to multiple checkpoint formats:
      - fnet keys at root            (e.g. 'fnet.conv1.weight')
      - nested under 'model'         (e.g. 'model.fnet.conv1.weight')
      - nested under 'module.model'  (e.g. 'module.model.fnet.conv1.weight')
      - checkpoint root IS the state_dict (no wrapping)
 
    Args:
        model     : LeapKernel instance (on CPU or GPU).
        ckpt_path : path to checkpoint file.
 
    Returns:
        dict with:
          'loaded'       : list of matched key names in OUR model
          'missing'      : fnet keys in OUR model that were NOT in the checkpoint
          'skipped'      : keys in the checkpoint that did NOT match our fnet
          'shape_errors' : list of (key, ours_shape, ckpt_shape) for size mismatches
    """
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
 
    logging.info(f"Loading pretrained fnet from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
 
    # ---- Locate the state_dict inside the checkpoint ----
    state_dict = None
    for candidate in ("model", "state_dict", "model_state_dict"):
        if isinstance(ckpt, dict) and candidate in ckpt and isinstance(ckpt[candidate], dict):
            state_dict = ckpt[candidate]
            logging.info(f"  Found state_dict at ckpt[{candidate!r}] ({len(state_dict)} entries)")
            break
    if state_dict is None and isinstance(ckpt, dict):
        # Maybe the checkpoint IS the state_dict
        if all(hasattr(v, "shape") for v in ckpt.values()):
            state_dict = ckpt
            logging.info(f"  Using checkpoint root as state_dict ({len(state_dict)} entries)")
    if state_dict is None:
        raise ValueError(f"Could not find a state_dict in {ckpt_path}")
 
    # ---- Find all fnet keys in the checkpoint ----
    fnet_keys_ckpt = [k for k in state_dict.keys() if "fnet" in k]
    if not fnet_keys_ckpt:
        raise ValueError(f"No 'fnet' keys found in checkpoint {ckpt_path}")
 
    # ---- Determine prefix to strip ----
    # For each fnet key, everything before 'fnet.' is the prefix.
    prefixes_before_fnet = set()
    for k in fnet_keys_ckpt:
        idx = k.find("fnet.")
        if idx == -1:
            # Key contains 'fnet' but not 'fnet.' — skip (probably a 1D scalar like counter)
            continue
        prefixes_before_fnet.add(k[:idx])
 
    if len(prefixes_before_fnet) == 1:
        prefix = prefixes_before_fnet.pop()
        if prefix:
            logging.info(f"  Stripping common prefix: {prefix!r}")
        else:
            logging.info(f"  fnet keys at state_dict root — no prefix to strip")
    else:
        raise ValueError(
            f"Expected one consistent prefix before 'fnet.', got {prefixes_before_fnet}"
        )
 
    # ---- Build mapping: OUR key -> checkpoint tensor ----
    our_fnet_keys = {k for k in model.state_dict().keys() if k.startswith("fnet.")}
    ckpt_subdict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix) and k[len(prefix):].startswith("fnet."):
            new_key = k[len(prefix):]  # strip prefix, leaving 'fnet.xxx'
            ckpt_subdict[new_key] = v
 
    # ---- Match and shape-check ----
    our_sd = model.state_dict()
    loaded, missing, skipped, shape_errors = [], [], [], []
 
    for k in our_fnet_keys:
        if k not in ckpt_subdict:
            missing.append(k)
            continue
        ours_shape = tuple(our_sd[k].shape)
        ckpt_shape = tuple(ckpt_subdict[k].shape)
        if ours_shape != ckpt_shape:
            shape_errors.append((k, ours_shape, ckpt_shape))
            continue
        loaded.append(k)
 
    for k in ckpt_subdict:
        if k not in our_fnet_keys:
            skipped.append(k)
 
    # ---- Actually load ----
    load_dict = {k: ckpt_subdict[k] for k in loaded}
    incompat = model.load_state_dict(load_dict, strict=False)
    # incompat.missing_keys will be large (all non-fnet keys) — that's expected.
 
    # ---- Report ----
    logging.info(f"  fnet load summary:")
    logging.info(f"    loaded:       {len(loaded):4d} keys")
    logging.info(f"    missing:      {len(missing):4d} keys (in model, not in ckpt)")
    logging.info(f"    skipped:      {len(skipped):4d} keys (in ckpt, not in model)")
    logging.info(f"    shape errors: {len(shape_errors):4d}")
 
    if shape_errors:
        logging.error("  SHAPE MISMATCHES — fnet was NOT fully loaded:")
        for k, ours, ck in shape_errors[:10]:
            logging.error(f"    {k}: ours={ours}  ckpt={ck}")
        raise ValueError(f"{len(shape_errors)} shape mismatches — aborting")
 
    if missing:
        logging.warning(f"  {len(missing)} fnet keys were NOT in the checkpoint, e.g.:")
        for k in missing[:5]:
            logging.warning(f"    {k}")
 
    if loaded == 0 or len(loaded) < len(our_fnet_keys) * 0.5:
        raise ValueError(
            f"Loaded only {len(loaded)}/{len(our_fnet_keys)} fnet keys — "
            f"something is wrong. Check inspect_ckpt.py output."
        )
 
    logging.info(f"  ✓ fnet warm-start complete ({len(loaded)}/{len(our_fnet_keys)} keys)")
    return {"loaded": loaded, "missing": missing,
            "skipped": skipped, "shape_errors": shape_errors}
 
 
# ===========================================================================
# Main training loop
# ===========================================================================
 
def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
 
    # All outputs for this run go into checkpoints/{run_name}/
    # Structure:
    #   checkpoints/
    #   ├── baseline/
    #   │   ├── config.json
    #   │   ├── losses.csv
    #   │   ├── leap_002000.pth
    #   │   ├── leap_final.pth
    #   │   └── tb_logs/          ← TensorBoard events
    #   ├── gce_both/
    #   └── trunc_both/
    run_dir = Path(args.ckpt_path) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Run directory: {run_dir}")
 
    # Warn if old checkpoints exist and --resume not set
    existing = list(run_dir.glob("leap_*.pth"))
    if existing and not args.resume:
        logging.warning(
            f"{len(existing)} existing checkpoint(s) found in {run_dir} "
            f"but --resume not set — starting from scratch and overwriting. "
            f"Pass --resume to continue from latest checkpoint."
        )
 
    # ---- Model (random init — from scratch, NOT from leap_kernel.pth) ----
    cfg   = build_leap_cfg(args)
    model = LeapKernel(cfg=cfg, stride=4).to(device)
 
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {total_params:,}")
 
    # ---- Warm-start fnet from a pretrained CoTracker v1 checkpoint ----
    # Paper (Sec. 4.2): "We load the pretrained CoTracker weights for the image
    # feature extractor and train LEAP for 100,000 steps". This is REQUIRED
    # for stable Cauchy NLL — random fnet features make Σ oscillate and
    # cause L_main to diverge. See diag_run logs for empirical confirmation.
    if args.pretrained_fnet:
        load_pretrained_fnet(model, args.pretrained_fnet)
 
    # B==1 is hardcoded in LeapKernel. No DataParallel.
    # Use --grad_accum_steps to simulate an effective batch size > 1.
    if args.grad_accum_steps > 1:
        logging.info(
            f"Gradient accumulation: {args.grad_accum_steps} steps "
            f"→ effective batch size = {args.grad_accum_steps}"
        )
 
    # ---- Dataset ----
    if DATALOADER_AVAILABLE and not args.use_dummy:
        train_dataset = LeapKubricDataset(
            data_root=args.dataset_root,
            num_tracks=args.num_tracks,
        )
        logging.info(f"Real dataset: {len(train_dataset)} sequences")
    else:
        logging.warning("Using DummyDataset — no real data.")
        train_dataset = DummyDataset(length=args.dummy_length, N=args.num_tracks)
 
    # batch_size=1 required — LeapKernel asserts B==1
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    logging.info(f"DataLoader: {len(train_loader)} batches/epoch")
 
    # ---- Optimizer ----
    optimizer, scheduler = build_optimizer(model, args)
 
    # ---- TensorBoard ----
    log_dir = run_dir / "tb_logs"
    writer  = SummaryWriter(log_dir=str(log_dir))
    logging.info(f"TensorBoard → {log_dir}")
 
    # ---- CSV loss log (for matplotlib plots without TensorBoard) ----
    csv_path = run_dir / "losses.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss_total", "loss_main", "loss_vis",
                         "loss_dyn", "grad_norm", "lr",
                         "log_det_x", "log_det_y", "mahal_x", "mahal_y",
                         "sigma_diag_x", "sigma_diag_y", "delta_abs"])
 
    # Save full config for reproducibility
    cfg_path = run_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
 
    # ---- Resume ----
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, scheduler, run_dir)
 
    # ---- Training ----
    model.train()
    total_steps  = start_step
    should_train = True
    accum_count  = 0
    consecutive_ooms = 0   # reset to 0 after each successful optimizer.step
    MAX_OOM_STREAK   = 20  # abort if we hit this many OOMs in a row
    running      = {"total": 0.0, "main": 0.0, "vis": 0.0, "dyn": 0.0,
                    "log_det_x": 0.0, "log_det_y": 0.0,
                    "mahal_x": 0.0, "mahal_y": 0.0,
                    "sigma_diag_x": 0.0, "sigma_diag_y": 0.0,
                    "delta_abs": 0.0}
 
    logging.info(
        f"Starting: run={args.run_name} | "
        f"L_vis={args.loss_vis} | L_dyn={args.loss_dyn} | "
        f"steps={args.num_steps} | accum={args.grad_accum_steps}"
    )
 
    epoch = 0
    while should_train:
        epoch += 1
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
 
            # Move only what we need to GPU — skip 'valid' (used by dataloader
            # smoke test only, not needed during training)
            rgbs      = batch["rgbs"].to(device)          # [1, T, 3, H, W]
            coords_gt = batch["coords"].to(device)        # [1, T, N, 2]
            vis_gt    = batch["visibility"].to(device)    # [1, T, N]
            dyn_gt    = batch["is_dynamic"].to(device)    # [1, N]
            queries   = batch["queries"].to(device)       # [1, N, 3]
 
            # ---- Forward + Loss ----
            try:
                _, _, _, _, _, train_data = model(
                    rgbs=rgbs,
                    queries=queries,
                    iters=args.train_iters,
                    is_train=True,
                )
 
                if train_data is None:
                    logging.warning("train_data is None — skipping batch.")
                    torch.cuda.empty_cache()
                    continue
 
                loss, l_main, l_vis, l_dyn, diag = compute_batch_loss(
                    train_data, coords_gt, vis_gt, dyn_gt, args, device
                )
 
            except RuntimeError as e:
                if "out of memory" in str(e):
                    consecutive_ooms += 1
                    logging.warning(
                        f"OOM at step {total_steps} "
                        f"(streak={consecutive_ooms}/{MAX_OOM_STREAK}) — "
                        f"skipping batch and clearing cache."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    # Aggressive cleanup to stop OOM cascade.
                    # Re-binding to None is the reliable way to drop tensor refs;
                    # del on locals() returned from a dict copy doesn't actually
                    # free anything in CPython.
                    rgbs = queries = coords_gt = vis_gt = dyn_gt = None
                    train_data = loss = l_main = l_vis = l_dyn = diag = None
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    gc.collect()
                    # Pause to let DataLoader workers drain their pinned-memory
                    # queue — without this the next pre-fetched batch hits the
                    # same ceiling and OOMs again.
                    time.sleep(0.5)
 
                    if consecutive_ooms >= MAX_OOM_STREAK:
                        logging.error(
                            f"Got {consecutive_ooms} consecutive OOMs — aborting. "
                            f"Consider reducing --num_tracks or --num_anchors, "
                            f"or restart with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
                        )
                        raise RuntimeError(
                            f"OOM streak exceeded {MAX_OOM_STREAK} — aborting run."
                        )
                    continue
                else:
                    logging.error(f"RuntimeError at step {total_steps}: {e}", exc_info=True)
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    torch.cuda.empty_cache()
                    continue
            except Exception as e:
                logging.error(f"Forward/loss failed at step {total_steps}: {e}", exc_info=True)
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                torch.cuda.empty_cache()
                continue
 
            if not torch.isfinite(loss):
                logging.warning(
                    f"Non-finite loss={loss.item():.4f} at step {total_steps} — skipping."
                )
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                torch.cuda.empty_cache()
                continue
 
            # ---- Backward ----
            (loss / args.grad_accum_steps).backward()
            accum_count += 1
 
            # ---- Optimizer step every grad_accum_steps samples ----
            if accum_count == args.grad_accum_steps:
                # Compute gradient norm before clipping (for diagnostics)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                ).item()
 
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count  = 0
                total_steps += 1
                consecutive_ooms = 0   # healthy step — reset OOM streak
 
                # Free cache after every optimizer step
                torch.cuda.empty_cache()
 
                running["total"] += loss.item()
                running["main"]  += l_main.item()
                running["vis"]   += l_vis.item()
                running["dyn"]   += l_dyn.item()
                running["gnorm"] = running.get("gnorm", 0.0) + grad_norm
                # Diagnostics (per-batch means, already averaged across windows × iters)
                for key in ("log_det_x", "log_det_y", "mahal_x", "mahal_y",
                            "sigma_diag_x", "sigma_diag_y", "delta_abs"):
                    running[key] += diag.get(key, 0.0)
 
                # ---- Log ----
                if total_steps % args.log_every == 0:
                    avg_total = running["total"] / args.log_every
                    avg_main  = running["main"]  / args.log_every
                    avg_vis   = running["vis"]   / args.log_every
                    avg_dyn   = running["dyn"]   / args.log_every
                    avg_gnorm = running["gnorm"] / args.log_every
                    cur_lr    = optimizer.param_groups[0]["lr"]
 
                    avg_ldx   = running["log_det_x"]    / args.log_every
                    avg_ldy   = running["log_det_y"]    / args.log_every
                    avg_mhx   = running["mahal_x"]      / args.log_every
                    avg_mhy   = running["mahal_y"]      / args.log_every
                    avg_sdx   = running["sigma_diag_x"] / args.log_every
                    avg_sdy   = running["sigma_diag_y"] / args.log_every
                    avg_dab   = running["delta_abs"]    / args.log_every
 
                    # TensorBoard
                    writer.add_scalar("loss/total", avg_total, total_steps)
                    writer.add_scalar("loss/main",  avg_main,  total_steps)
                    writer.add_scalar("loss/vis",   avg_vis,   total_steps)
                    writer.add_scalar("loss/dyn",   avg_dyn,   total_steps)
                    writer.add_scalar("train/grad_norm", avg_gnorm, total_steps)
                    writer.add_scalar("train/lr",   cur_lr,    total_steps)
                    writer.add_scalar("diag/log_det_x",    avg_ldx, total_steps)
                    writer.add_scalar("diag/log_det_y",    avg_ldy, total_steps)
                    writer.add_scalar("diag/mahal_x",      avg_mhx, total_steps)
                    writer.add_scalar("diag/mahal_y",      avg_mhy, total_steps)
                    writer.add_scalar("diag/sigma_diag_x", avg_sdx, total_steps)
                    writer.add_scalar("diag/sigma_diag_y", avg_sdy, total_steps)
                    writer.add_scalar("diag/delta_abs",    avg_dab, total_steps)
 
                    # CSV — one row per log interval
                    csv_writer.writerow([
                        total_steps, avg_total, avg_main,
                        avg_vis, avg_dyn, avg_gnorm, cur_lr,
                        avg_ldx, avg_ldy, avg_mhx, avg_mhy,
                        avg_sdx, avg_sdy, avg_dab,
                    ])
                    csv_file.flush()
 
                    # Reset running stats
                    running = {"total": 0.0, "main": 0.0,
                               "vis": 0.0, "dyn": 0.0, "gnorm": 0.0,
                               "log_det_x": 0.0, "log_det_y": 0.0,
                               "mahal_x": 0.0, "mahal_y": 0.0,
                               "sigma_diag_x": 0.0, "sigma_diag_y": 0.0,
                               "delta_abs": 0.0}
 
                    logging.info(
                        f"Step {total_steps:6d} | "
                        f"total={avg_total:.4f}  main={avg_main:.4f}  "
                        f"vis={avg_vis:.4f}  dyn={avg_dyn:.4f}  "
                        f"gnorm={avg_gnorm:.3f}  lr={cur_lr:.2e}"
                    )
                    logging.info(
                        f"       DIAG | "
                        f"logdet(x,y)=({avg_ldx:+.3f},{avg_ldy:+.3f})  "
                        f"mahal(x,y)=({avg_mhx:.3f},{avg_mhy:.3f})  "
                        f"sigdiag(x,y)=({avg_sdx:.3e},{avg_sdy:.3e})  "
                        f"|delta|={avg_dab:.3f}"
                    )
 
                # ---- Checkpoint ----
                if total_steps % args.save_every == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, total_steps, run_dir,
                        tag=str(total_steps).zfill(6)
                    )
 
                if total_steps >= args.num_steps:
                    should_train = False
                    break
 
    save_checkpoint(model, optimizer, scheduler, total_steps, run_dir, tag="final")
    writer.close()
    csv_file.close()
    logging.info("Training complete.")
 
 
# ===========================================================================
# Argument parsing
# ===========================================================================
 
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="LEAP-VO Training")
 
    # Paths
    p.add_argument("--dataset_root", default="~/ASU/PIR/leapvo_training/training_dataset")
    p.add_argument("--ckpt_path",    default="~/ASU/PIR/leapvo_training/checkpoints")
    p.add_argument("--run_name",     default="baseline",
                   help="baseline | gce_vis | gce_dyn | gce_both | trunc_vis | trunc_both")
    p.add_argument("--pretrained_fnet", type=str, default=None,
                   help="Path to CoTracker v1 checkpoint (.pth) to warm-start fnet. "
                        "Paper Sec 4.2 requires this. Use empty string or omit to skip.")
 
    # Loss config
    p.add_argument("--loss_vis",   default="bce", choices=["bce", "gce", "trunc_gce"])
    p.add_argument("--loss_dyn",   default="bce", choices=["bce", "gce", "trunc_gce"])
    p.add_argument("--gce_q",      type=float, default=0.7,
                   help="GCE exponent q. 0→CE, 1→MAE. Paper recommends 0.7")
    p.add_argument("--trunc_frac", type=float, default=0.7,
                   help="Fraction of samples retained in Truncated GCE")
 
    # Loss weights (paper Eq. 8)
    p.add_argument("--w1", type=float, default=1.0, help="L_main weight")
    p.add_argument("--w2", type=float, default=0.5, help="L_vis weight")
    p.add_argument("--w3", type=float, default=0.5, help="L_dyn weight")
 
    # Training schedule
    p.add_argument("--num_steps",        type=int,   default=50000,
                   help="Paper: 100k on 4xA100. 50k on 1x4090 is a good start.")
    p.add_argument("--lr",               type=float, default=5e-4)
    p.add_argument("--wdecay",           type=float, default=1e-5)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int,   default=1,
                   help="Gradient accumulation. B==1 is hardcoded in LeapKernel.")
 
    # Model / data
    p.add_argument("--window_size",  type=int, default=8,
                   help="Sliding window S. Paper: 8.")
    p.add_argument("--train_iters",  type=int, default=4,
                   help="Refinement iterations K. Paper: 4.")
    p.add_argument("--num_tracks",   type=int, default=256,
                   help="Query tracks N per sequence. Paper: 256.")
    p.add_argument("--num_anchors",  type=int, default=64,
                   help="Anchor points for inter-track attention. Paper: 64. 0=disable.")
    p.add_argument("--num_workers",  type=int, default=4)
 
    # Logging / checkpointing
    p.add_argument("--log_every",  type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume",     action="store_true")
 
    # Debug
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--use_dummy",    action="store_true",
                   help="Use DummyDataset for smoke testing")
    p.add_argument("--dummy_length", type=int, default=200)
 
    return p.parse_args()
 
 
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = parse_args()
    args.dataset_root = str(Path(args.dataset_root).expanduser())
    args.ckpt_path    = str(Path(args.ckpt_path).expanduser())
    if args.pretrained_fnet:
        args.pretrained_fnet = str(Path(args.pretrained_fnet).expanduser())
    train(args)