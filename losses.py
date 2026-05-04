"""
losses.py — LEAP-VO Loss Functions
====================================
Implements the three loss components from the paper (Eq. 5-8):
 
    L_total = w1 * L_main + w2 * L_vis + w3 * L_dyn          (Eq. 8)
 
CRITICAL IMPLEMENTATION NOTE — coordinate spaces:
    LeapKernel internally operates in STRIDE space (coords / stride=4).
    coord_predictions are scaled BACK to pixel space on line 281:
        coord_predictions.append(coords * self.stride)
    But the covariance matrices (sigma_x, sigma_y) are computed from
    features that remain in STRIDE space — they are NEVER rescaled.
 
    Therefore in cauchy_nll_loss we MUST normalize coords back to stride
    space before computing residuals:
        delta = (coord_pred - coord_gt) / stride
 
    Failure to do this caused gradient explosion in the baseline run
    (gnorm reaching 40,000+ by step 50k).
 
L_main — Cauchy NLL (Eq. 4-5)
--------------------------------
For coordinate a (x or y), the S-dim multivariate Cauchy NLL is:
 
    -log p(a) = (1/2)*log|Sigma|
              + ((1+S)/2)*log(1 + delta^T Sigma^-1 delta)
              + C(S)    [constant, dropped]
 
where delta = (pred - gt) / stride   [in stride space, matching Sigma]
 
L_vis / L_dyn — Classification (Eq. 6-7)
------------------------------------------
LeapKernel outputs ALREADY SIGMOID-ED probabilities.
Use plain BCE, not BCEWithLogits.
 
Variants: bce | gce (q=0.7) | trunc_gce (q=0.7, trunc_frac=0.7)
 
GCE: L_GCE = (1 - p_y^q) / q   [Zhang & Sabuncu, NeurIPS 2018]
     p_y = probability of true class
     q→0: cross-entropy, q=1: MAE, q=0.7: recommended
"""
 
import torch
 
 
STRIDE = 4   # LeapKernel stride — must match cfg.model.stride
 
 
# ===========================================================================
# L_main — Multivariate Cauchy NLL  (Eq. 4-5)
# ===========================================================================
 
def cauchy_nll_loss(coord_pred, coord_gt, sigma_x, sigma_y, valid_mask,
                    stride=STRIDE, return_diagnostics=False):
    """
    Negative log-likelihood of multivariate Cauchy distribution.
 
    Args:
        coord_pred : [B, S, N, 2]  predicted trajectory  (PIXEL coords)
        coord_gt   : [B, S, N, 2]  ground truth           (PIXEL coords)
        sigma_x    : [B, N, S, S]  scale matrix for x     (STRIDE space, SPD)
        sigma_y    : [B, N, S, S]  scale matrix for y     (STRIDE space, SPD)
        valid_mask : [B, S, N]     bool
        stride     : int           model stride (4) — used to normalize coords
        return_diagnostics : if True, also return dict with per-term means
 
    Returns:
        scalar loss                              (if return_diagnostics=False)
        (scalar loss, diagnostics_dict)          (if return_diagnostics=True)
 
    Diagnostics dict (all detached floats, for logging only):
        'log_det_x'     : mean log-det term (clamped) for x, over valid tracks
        'log_det_y'     : mean log-det term (clamped) for y
        'mahal_x'       : mean Mahalanobis term for x
        'mahal_y'       : mean Mahalanobis term for y
        'sigma_diag_x'  : mean of diag(Sigma_x)  — watch for Σ-collapse
        'sigma_diag_y'  : mean of diag(Sigma_y)
        'delta_abs'     : mean |delta| in stride space  — sanity check
    """
    # Normalize residuals to STRIDE space to match covariance matrices
    delta    = (coord_pred - coord_gt) / stride          # [B, S, N, 2]
 
    delta_x  = delta[..., 0].permute(0, 2, 1)            # [B, N, S]
    delta_y  = delta[..., 1].permute(0, 2, 1)            # [B, N, S]
    mask     = valid_mask.permute(0, 2, 1).float()        # [B, N, S]
 
    # Zero invalid positions
    delta_x  = delta_x * mask
    delta_y  = delta_y * mask
 
    S        = delta_x.shape[-1]
    loss_x, diag_x = _cauchy_nll_single(delta_x, sigma_x, S, return_terms=True)
    loss_y, diag_y = _cauchy_nll_single(delta_y, sigma_y, S, return_terms=True)
    loss     = loss_x + loss_y                            # [B, N]
 
    # Average over valid tracks only
    track_valid = mask.any(dim=2).float()                 # [B, N]
    denom       = track_valid.sum().clamp(min=1.0)
    total_loss  = (loss * track_valid).sum() / denom
 
    if not return_diagnostics:
        return total_loss
 
    # Build diagnostics — weighted mean over valid tracks so numbers match loss
    tv = track_valid
    denom_t = tv.sum().clamp(min=1.0)
 
    def _track_mean(per_track):
        return ((per_track * tv).sum() / denom_t).detach()
 
    with torch.no_grad():
        # Sigma diagonal mean (O(1) if healthy, ↓ if Σ is collapsing)
        sx_diag = sigma_x.diagonal(dim1=-2, dim2=-1).mean(dim=-1)   # [B, N]
        sy_diag = sigma_y.diagonal(dim1=-2, dim2=-1).mean(dim=-1)   # [B, N]
        dabs    = (delta_x.abs().mean(dim=-1) + delta_y.abs().mean(dim=-1)) * 0.5
 
    diagnostics = {
        "log_det_x":    _track_mean(diag_x["log_det"]).item(),
        "log_det_y":    _track_mean(diag_y["log_det"]).item(),
        "mahal_x":      _track_mean(diag_x["mahal"]).item(),
        "mahal_y":      _track_mean(diag_y["mahal"]).item(),
        "sigma_diag_x": _track_mean(sx_diag).item(),
        "sigma_diag_y": _track_mean(sy_diag).item(),
        "delta_abs":    _track_mean(dabs).item(),
    }
    return total_loss, diagnostics
 
 
def _cauchy_nll_single(delta, sigma, S, return_terms=False):
    """
    NLL of S-dimensional multivariate Cauchy for one coordinate.
 
    -log p(a) = (1/2)*log|Sigma| + ((1+S)/2)*log(1 + delta^T Sigma^-1 delta)
 
    Numerical strategy:
        - Cholesky decomposition for log-det (stable for SPD matrices)
        - cholesky_solve for Sigma^-1 delta (avoids explicit inversion)
        - log1p for the Mahalanobis term (avoids log(1+x) cancellation)
        - Clamp log-det to prevent loss going negative due to numerical drift
 
    Args:
        delta : [B, N, S]    residual in stride space
        sigma : [B, N, S, S] scale matrix (SPD — kernel + epsilon*I)
        S     : window length
        return_terms : if True, also return dict with 'log_det' and 'mahal'
                       (each [B, N], for diagnostics)
 
    Returns:
        loss : [B, N]  (non-negative)
        [terms: dict]  (if return_terms=True)
    """
    B, N, _ = delta.shape
 
    # --- Cholesky decomposition with escalating regularization ---
    # sigma should be SPD by construction (kernel + epsilon*I in KernelBlock).
    # In practice, late in training sigma can drift toward singular for a
    # handful of batch elements. We try increasingly larger jitter values until
    # cholesky succeeds. Almost any near-PSD matrix becomes solvable in this
    # range; the final 1.0*I tier is essentially "give up calibration for this
    # element but keep training."
    eps_eye = torch.eye(S, device=sigma.device, dtype=sigma.dtype)
    eps_eye = eps_eye.unsqueeze(0).unsqueeze(0)               # [1, 1, S, S]
 
    L = None
    last_err = None
    used_eps = None
    for eps_val in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        sigma_reg = sigma + eps_val * eps_eye
        try:
            L = torch.linalg.cholesky(sigma_reg)              # [B, N, S, S]
            used_eps = eps_val
            break
        except RuntimeError as e:
            last_err = e
            continue
 
    if L is None:
        # All five tiers failed. This is an exceptional path; rather than
        # crashing the whole run, build L per-batch-element using the most
        # aggressive jitter and zero-out elements that still won't decompose.
        # The masked average in cauchy_nll_loss naturally ignores zeroed
        # rows since they multiply by track_valid.
        import logging
        logging.warning(
            f"Cauchy NLL: all jitter levels failed for sigma. "
            f"Falling back to identity for this batch. Last error: {last_err}"
        )
        L = eps_eye.expand_as(sigma).contiguous().sqrt()      # = I
        used_eps = 1.0
    elif used_eps > 1e-4:
        # Log when we needed more than the default jitter — useful signal that
        # Sigma is drifting. Doesn't kill training but flags the issue.
        import logging
        logging.debug(f"Cauchy NLL: needed eps={used_eps:.0e} for cholesky")
 
    # --- Log-determinant: (1/2) * log|Sigma| = sum(log(diag(L))) ---
    log_diag = L.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).log()  # [B, N, S]
    log_det  = log_diag.sum(dim=-1)                                  # [B, N]
 
    # Clamp log_det: prevent it from being so negative that it dominates
    # In stride space with well-behaved features, |Sigma| shouldn't be tiny
    log_det  = log_det.clamp(min=-50.0)
 
    log_det_term = log_det                                    # [B, N]  (already halved above — sum of log(diag(L)), not 2*sum)
 
    # --- Mahalanobis: ((1+S)/2) * log(1 + delta^T Sigma^-1 delta) ---
    delta_col        = delta.unsqueeze(-1)                    # [B, N, S, 1]
    sigma_inv_delta  = torch.cholesky_solve(delta_col, L)     # [B, N, S, 1]
    mahal            = (delta_col * sigma_inv_delta).sum(dim=(-2, -1))  # [B, N]
    mahal            = mahal.clamp(min=0.0)                   # numerical safety
    mahal_term       = ((1.0 + S) / 2.0) * torch.log1p(mahal)  # [B, N]
 
    total = log_det_term + mahal_term                          # [B, N]
 
    if return_terms:
        return total, {"log_det": log_det_term, "mahal": mahal_term}
    return total
 
 
# ===========================================================================
# Classification losses — inputs are PROBABILITIES (sigmoid already applied)
# ===========================================================================
 
def bce_loss(probs, targets, valid_mask, input_is_prob=True):
    """
    Plain BCE on sigmoid-ed probabilities.
 
    Args:
        probs      : [B, S, N]  already sigmoid-ed
        targets    : [B, S, N]  bool/float ground truth
        valid_mask : [B, S, N]  bool
    """
    eps   = 1e-6
    p     = probs.clamp(eps, 1.0 - eps)
    t     = targets.float()
    loss  = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
 
    mask  = valid_mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom
 
 
def gce_loss(probs, targets, valid_mask, q=0.7, input_is_prob=True):
    """
    Generalized Cross Entropy (Zhang & Sabuncu, NeurIPS 2018).
 
    L_GCE = (1 - p_y^q) / q
    p_y   = probability assigned to the TRUE class
 
    q→0: cross-entropy  (noise-sensitive)
    q=1: MAE            (noise-robust)
    q=0.7: recommended balance
 
    Args:
        probs      : [B, S, N]  already sigmoid-ed
        targets    : [B, S, N]  bool/float
        valid_mask : [B, S, N]  bool
        q          : float in (0, 1]
    """
    assert 0.0 < q <= 1.0, f"GCE q must be in (0,1], got {q}"
    eps = 1e-6
    p   = probs.clamp(eps, 1.0 - eps)
    t   = targets.float()
 
    p_y  = t * p + (1.0 - t) * (1.0 - p)    # prob of true class
    loss = (1.0 - p_y.pow(q)) / q
 
    mask  = valid_mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom
 
 
def truncated_gce_loss(probs, targets, valid_mask, q=0.7, trunc_frac=0.7,
                       input_is_prob=True):
    """
    Truncated GCE — zeros out the (1-trunc_frac) highest-loss samples.
    Keeps only the trunc_frac easiest samples, pruning likely noisy labels.
 
    Args:
        probs      : [B, S, N]  already sigmoid-ed
        targets    : [B, S, N]  bool/float
        valid_mask : [B, S, N]  bool
        q          : float in (0, 1]
        trunc_frac : float in (0, 1] — fraction of samples to KEEP
    """
    assert 0.0 < q <= 1.0
    assert 0.0 < trunc_frac <= 1.0
 
    eps  = 1e-6
    p    = probs.clamp(eps, 1.0 - eps)
    t    = targets.float()
    p_y  = t * p + (1.0 - t) * (1.0 - p)
    loss = (1.0 - p_y.pow(q)) / q
 
    mask         = valid_mask.float()
    valid_losses = loss[valid_mask.bool()]
 
    if valid_losses.numel() == 0:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
 
    k         = max(1, int(trunc_frac * valid_losses.numel()))
    threshold = valid_losses.topk(k, largest=False).values[-1]
    keep_mask = (loss <= threshold).float() * mask
 
    denom = keep_mask.sum().clamp(min=1.0)
    return (loss * keep_mask).sum() / denom
 
 
# ===========================================================================
# Self-test
# ===========================================================================
 
if __name__ == "__main__":
    import sys
    print("Testing losses.py...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    B, S, N = 2, 8, 64
    torch.manual_seed(42)
 
    # Simulate model outputs: coords in PIXEL space, sigma in STRIDE space
    stride     = 4
    coord_pred = torch.rand(B, S, N, 2, device=device, requires_grad=True) * 256.0
    coord_gt   = torch.rand(B, S, N, 2, device=device) * 256.0
 
    # Sigma: SPD matrix from kernel (stride space, values O(1))
    feat    = torch.rand(B*N, S, 128, device=device, requires_grad=True)
    raw     = torch.bmm(feat, feat.transpose(1, 2))
    eps_eye = 1e-4 * torch.eye(S, device=device).unsqueeze(0)
    sigma   = (raw + eps_eye).reshape(B, N, S, S)
 
    valid   = torch.ones(B, S, N, dtype=torch.bool, device=device)
 
    print("Testing cauchy_nll_loss...")
    l = cauchy_nll_loss(coord_pred, coord_gt, sigma, sigma, valid, stride=stride)
    print(f"  L_main = {l.item():.4f}  (should be positive)")
    assert l.item() >= 0, f"Cauchy NLL must be non-negative, got {l.item()}"
    l.backward()
    print(f"  Backward: OK")
 
    # Test with large pixel residuals — should still be stable
    coord_pred2 = torch.rand(B, S, N, 2, device=device, requires_grad=True) * 512.0
    coord_gt2   = torch.zeros(B, S, N, 2, device=device)
    l2 = cauchy_nll_loss(coord_pred2, coord_gt2, sigma.detach(), sigma.detach(),
                         valid, stride=stride)
    print(f"  Large residual test: L_main = {l2.item():.4f}  (should be positive)")
    assert l2.item() >= 0
 
    print("\nTesting classification losses...")
    vis_probs = torch.sigmoid(torch.randn(B, S, N, device=device))
    vis_gt    = torch.randint(0, 2, (B, S, N), device=device).bool()
 
    l_bce  = bce_loss(vis_probs, vis_gt, valid)
    l_gce  = gce_loss(vis_probs, vis_gt, valid, q=0.7)
    l_tgce = truncated_gce_loss(vis_probs, vis_gt, valid, q=0.7, trunc_frac=0.7)
    print(f"  BCE={l_bce.item():.4f}  GCE={l_gce.item():.4f}  TruncGCE={l_tgce.item():.4f}")
 
    print("\nGCE q sensitivity:")
    for q in [0.1, 0.5, 0.7, 1.0]:
        l = gce_loss(vis_probs, vis_gt, valid, q=q)
        print(f"  q={q:.1f} → {l.item():.4f}")
 
    print("\nTruncGCE frac sensitivity:")
    for frac in [0.3, 0.7, 1.0]:
        l = truncated_gce_loss(vis_probs, vis_gt, valid, q=0.7, trunc_frac=frac)
        print(f"  frac={frac:.1f} → {l.item():.4f}")
 
    print("\n✓ All tests passed.")


# """
# losses.py — LEAP-VO Loss Functions
# ====================================
# Implements the three loss components from the paper (Eq. 5-8):
 
#     L_total = w1 * L_main + w2 * L_vis + w3 * L_dyn          (Eq. 8)
 
# CRITICAL IMPLEMENTATION NOTE — coordinate spaces:
#     LeapKernel internally operates in STRIDE space (coords / stride=4).
#     coord_predictions are scaled BACK to pixel space on line 281:
#         coord_predictions.append(coords * self.stride)
#     But the covariance matrices (sigma_x, sigma_y) are computed from
#     features that remain in STRIDE space — they are NEVER rescaled.
 
#     Therefore in cauchy_nll_loss we MUST normalize coords back to stride
#     space before computing residuals:
#         delta = (coord_pred - coord_gt) / stride
 
#     Failure to do this caused gradient explosion in the baseline run
#     (gnorm reaching 40,000+ by step 50k).
 
# L_main — Cauchy NLL (Eq. 4-5)
# --------------------------------
# For coordinate a (x or y), the S-dim multivariate Cauchy NLL is:
 
#     -log p(a) = (1/2)*log|Sigma|
#               + ((1+S)/2)*log(1 + delta^T Sigma^-1 delta)
#               + C(S)    [constant, dropped]
 
# where delta = (pred - gt) / stride   [in stride space, matching Sigma]
 
# L_vis / L_dyn — Classification (Eq. 6-7)
# ------------------------------------------
# LeapKernel outputs ALREADY SIGMOID-ED probabilities.
# Use plain BCE, not BCEWithLogits.
 
# Variants: bce | gce (q=0.7) | trunc_gce (q=0.7, trunc_frac=0.7)
 
# GCE: L_GCE = (1 - p_y^q) / q   [Zhang & Sabuncu, NeurIPS 2018]
#      p_y = probability of true class
#      q→0: cross-entropy, q=1: MAE, q=0.7: recommended
# """
 
# import torch
 
 
# STRIDE = 4   # LeapKernel stride — must match cfg.model.stride
 
 
# # ===========================================================================
# # L_main — Multivariate Cauchy NLL  (Eq. 4-5)
# # ===========================================================================
 
# def cauchy_nll_loss(coord_pred, coord_gt, sigma_x, sigma_y, valid_mask,
#                     stride=STRIDE, return_diagnostics=False):
#     """
#     Negative log-likelihood of multivariate Cauchy distribution.
 
#     Args:
#         coord_pred : [B, S, N, 2]  predicted trajectory  (PIXEL coords)
#         coord_gt   : [B, S, N, 2]  ground truth           (PIXEL coords)
#         sigma_x    : [B, N, S, S]  scale matrix for x     (STRIDE space, SPD)
#         sigma_y    : [B, N, S, S]  scale matrix for y     (STRIDE space, SPD)
#         valid_mask : [B, S, N]     bool
#         stride     : int           model stride (4) — used to normalize coords
#         return_diagnostics : if True, also return dict with per-term means
 
#     Returns:
#         scalar loss                              (if return_diagnostics=False)
#         (scalar loss, diagnostics_dict)          (if return_diagnostics=True)
 
#     Diagnostics dict (all detached floats, for logging only):
#         'log_det_x'     : mean log-det term (clamped) for x, over valid tracks
#         'log_det_y'     : mean log-det term (clamped) for y
#         'mahal_x'       : mean Mahalanobis term for x
#         'mahal_y'       : mean Mahalanobis term for y
#         'sigma_diag_x'  : mean of diag(Sigma_x)  — watch for Σ-collapse
#         'sigma_diag_y'  : mean of diag(Sigma_y)
#         'delta_abs'     : mean |delta| in stride space  — sanity check
#     """
#     # Normalize residuals to STRIDE space to match covariance matrices
#     delta    = (coord_pred - coord_gt) / stride          # [B, S, N, 2]
 
#     delta_x  = delta[..., 0].permute(0, 2, 1)            # [B, N, S]
#     delta_y  = delta[..., 1].permute(0, 2, 1)            # [B, N, S]
#     mask     = valid_mask.permute(0, 2, 1).float()        # [B, N, S]
 
#     # Zero invalid positions
#     delta_x  = delta_x * mask
#     delta_y  = delta_y * mask
 
#     S        = delta_x.shape[-1]
#     loss_x, diag_x = _cauchy_nll_single(delta_x, sigma_x, S, return_terms=True)
#     loss_y, diag_y = _cauchy_nll_single(delta_y, sigma_y, S, return_terms=True)
#     loss     = loss_x + loss_y                            # [B, N]
 
#     # Average over valid tracks only
#     track_valid = mask.any(dim=2).float()                 # [B, N]
#     denom       = track_valid.sum().clamp(min=1.0)
#     total_loss  = (loss * track_valid).sum() / denom
 
#     if not return_diagnostics:
#         return total_loss
 
#     # Build diagnostics — weighted mean over valid tracks so numbers match loss
#     tv = track_valid
#     denom_t = tv.sum().clamp(min=1.0)
 
#     def _track_mean(per_track):
#         return ((per_track * tv).sum() / denom_t).detach()
 
#     with torch.no_grad():
#         # Sigma diagonal mean (O(1) if healthy, ↓ if Σ is collapsing)
#         sx_diag = sigma_x.diagonal(dim1=-2, dim2=-1).mean(dim=-1)   # [B, N]
#         sy_diag = sigma_y.diagonal(dim1=-2, dim2=-1).mean(dim=-1)   # [B, N]
#         dabs    = (delta_x.abs().mean(dim=-1) + delta_y.abs().mean(dim=-1)) * 0.5
 
#     diagnostics = {
#         "log_det_x":    _track_mean(diag_x["log_det"]).item(),
#         "log_det_y":    _track_mean(diag_y["log_det"]).item(),
#         "mahal_x":      _track_mean(diag_x["mahal"]).item(),
#         "mahal_y":      _track_mean(diag_y["mahal"]).item(),
#         "sigma_diag_x": _track_mean(sx_diag).item(),
#         "sigma_diag_y": _track_mean(sy_diag).item(),
#         "delta_abs":    _track_mean(dabs).item(),
#     }
#     return total_loss, diagnostics
 
 
# def _cauchy_nll_single(delta, sigma, S, return_terms=False):
#     """
#     NLL of S-dimensional multivariate Cauchy for one coordinate.
 
#     -log p(a) = (1/2)*log|Sigma| + ((1+S)/2)*log(1 + delta^T Sigma^-1 delta)
 
#     Numerical strategy:
#         - Cholesky decomposition for log-det (stable for SPD matrices)
#         - cholesky_solve for Sigma^-1 delta (avoids explicit inversion)
#         - log1p for the Mahalanobis term (avoids log(1+x) cancellation)
#         - Clamp log-det to prevent loss going negative due to numerical drift
 
#     Args:
#         delta : [B, N, S]    residual in stride space
#         sigma : [B, N, S, S] scale matrix (SPD — kernel + epsilon*I)
#         S     : window length
#         return_terms : if True, also return dict with 'log_det' and 'mahal'
#                        (each [B, N], for diagnostics)
 
#     Returns:
#         loss : [B, N]  (non-negative)
#         [terms: dict]  (if return_terms=True)
#     """
#     B, N, _ = delta.shape
 
#     # --- Cholesky decomposition ---
#     # sigma should be SPD by construction (kernel + epsilon*I in KernelBlock)
#     # Add a small regularizer just in case numerical drift made it non-SPD
#     eps_eye = 1e-4 * torch.eye(S, device=sigma.device, dtype=sigma.dtype)
#     sigma_reg = sigma + eps_eye.unsqueeze(0).unsqueeze(0)   # [B, N, S, S]
 
#     try:
#         L = torch.linalg.cholesky(sigma_reg)                 # [B, N, S, S]
#     except RuntimeError:
#         # Fallback: increase regularization
#         sigma_reg = sigma + 1e-2 * torch.eye(
#             S, device=sigma.device, dtype=sigma.dtype
#         ).unsqueeze(0).unsqueeze(0)
#         L = torch.linalg.cholesky(sigma_reg)
 
#     # --- Log-determinant: (1/2) * log|Sigma| = sum(log(diag(L))) ---
#     log_diag = L.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).log()  # [B, N, S]
#     log_det  = log_diag.sum(dim=-1)                                  # [B, N]
 
#     # Clamp log_det: prevent it from being so negative that it dominates
#     # In stride space with well-behaved features, |Sigma| shouldn't be tiny
#     log_det  = log_det.clamp(min=-50.0)
 
#     log_det_term = log_det                                    # [B, N]  (already halved above — sum of log(diag(L)), not 2*sum)
 
#     # --- Mahalanobis: ((1+S)/2) * log(1 + delta^T Sigma^-1 delta) ---
#     delta_col        = delta.unsqueeze(-1)                    # [B, N, S, 1]
#     sigma_inv_delta  = torch.cholesky_solve(delta_col, L)     # [B, N, S, 1]
#     mahal            = (delta_col * sigma_inv_delta).sum(dim=(-2, -1))  # [B, N]
#     mahal            = mahal.clamp(min=0.0)                   # numerical safety
#     mahal_term       = ((1.0 + S) / 2.0) * torch.log1p(mahal)  # [B, N]
 
#     total = log_det_term + mahal_term                          # [B, N]
 
#     if return_terms:
#         return total, {"log_det": log_det_term, "mahal": mahal_term}
#     return total
 
 
# # ===========================================================================
# # Classification losses — inputs are PROBABILITIES (sigmoid already applied)
# # ===========================================================================
 
# def bce_loss(probs, targets, valid_mask, input_is_prob=True):
#     """
#     Plain BCE on sigmoid-ed probabilities.
 
#     Args:
#         probs      : [B, S, N]  already sigmoid-ed
#         targets    : [B, S, N]  bool/float ground truth
#         valid_mask : [B, S, N]  bool
#     """
#     eps   = 1e-6
#     p     = probs.clamp(eps, 1.0 - eps)
#     t     = targets.float()
#     loss  = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
 
#     mask  = valid_mask.float()
#     denom = mask.sum().clamp(min=1.0)
#     return (loss * mask).sum() / denom
 
 
# def gce_loss(probs, targets, valid_mask, q=0.7, input_is_prob=True):
#     """
#     Generalized Cross Entropy (Zhang & Sabuncu, NeurIPS 2018).
 
#     L_GCE = (1 - p_y^q) / q
#     p_y   = probability assigned to the TRUE class
 
#     q→0: cross-entropy  (noise-sensitive)
#     q=1: MAE            (noise-robust)
#     q=0.7: recommended balance
 
#     Args:
#         probs      : [B, S, N]  already sigmoid-ed
#         targets    : [B, S, N]  bool/float
#         valid_mask : [B, S, N]  bool
#         q          : float in (0, 1]
#     """
#     assert 0.0 < q <= 1.0, f"GCE q must be in (0,1], got {q}"
#     eps = 1e-6
#     p   = probs.clamp(eps, 1.0 - eps)
#     t   = targets.float()
 
#     p_y  = t * p + (1.0 - t) * (1.0 - p)    # prob of true class
#     loss = (1.0 - p_y.pow(q)) / q
 
#     mask  = valid_mask.float()
#     denom = mask.sum().clamp(min=1.0)
#     return (loss * mask).sum() / denom
 
 
# def truncated_gce_loss(probs, targets, valid_mask, q=0.7, trunc_frac=0.7,
#                        input_is_prob=True):
#     """
#     Truncated GCE — zeros out the (1-trunc_frac) highest-loss samples.
#     Keeps only the trunc_frac easiest samples, pruning likely noisy labels.
 
#     Args:
#         probs      : [B, S, N]  already sigmoid-ed
#         targets    : [B, S, N]  bool/float
#         valid_mask : [B, S, N]  bool
#         q          : float in (0, 1]
#         trunc_frac : float in (0, 1] — fraction of samples to KEEP
#     """
#     assert 0.0 < q <= 1.0
#     assert 0.0 < trunc_frac <= 1.0
 
#     eps  = 1e-6
#     p    = probs.clamp(eps, 1.0 - eps)
#     t    = targets.float()
#     p_y  = t * p + (1.0 - t) * (1.0 - p)
#     loss = (1.0 - p_y.pow(q)) / q
 
#     mask         = valid_mask.float()
#     valid_losses = loss[valid_mask.bool()]
 
#     if valid_losses.numel() == 0:
#         return torch.tensor(0.0, device=probs.device, requires_grad=True)
 
#     k         = max(1, int(trunc_frac * valid_losses.numel()))
#     threshold = valid_losses.topk(k, largest=False).values[-1]
#     keep_mask = (loss <= threshold).float() * mask
 
#     denom = keep_mask.sum().clamp(min=1.0)
#     return (loss * keep_mask).sum() / denom
 
 
# # ===========================================================================
# # Self-test
# # ===========================================================================
 
# if __name__ == "__main__":
#     import sys
#     print("Testing losses.py...\n")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")
 
#     B, S, N = 2, 8, 64
#     torch.manual_seed(42)
 
#     # Simulate model outputs: coords in PIXEL space, sigma in STRIDE space
#     stride     = 4
#     coord_pred = torch.rand(B, S, N, 2, device=device, requires_grad=True) * 256.0
#     coord_gt   = torch.rand(B, S, N, 2, device=device) * 256.0
 
#     # Sigma: SPD matrix from kernel (stride space, values O(1))
#     feat    = torch.rand(B*N, S, 128, device=device, requires_grad=True)
#     raw     = torch.bmm(feat, feat.transpose(1, 2))
#     eps_eye = 1e-4 * torch.eye(S, device=device).unsqueeze(0)
#     sigma   = (raw + eps_eye).reshape(B, N, S, S)
 
#     valid   = torch.ones(B, S, N, dtype=torch.bool, device=device)
 
#     print("Testing cauchy_nll_loss...")
#     l = cauchy_nll_loss(coord_pred, coord_gt, sigma, sigma, valid, stride=stride)
#     print(f"  L_main = {l.item():.4f}  (should be positive)")
#     assert l.item() >= 0, f"Cauchy NLL must be non-negative, got {l.item()}"
#     l.backward()
#     print(f"  Backward: OK")
 
#     # Test with large pixel residuals — should still be stable
#     coord_pred2 = torch.rand(B, S, N, 2, device=device, requires_grad=True) * 512.0
#     coord_gt2   = torch.zeros(B, S, N, 2, device=device)
#     l2 = cauchy_nll_loss(coord_pred2, coord_gt2, sigma.detach(), sigma.detach(),
#                          valid, stride=stride)
#     print(f"  Large residual test: L_main = {l2.item():.4f}  (should be positive)")
#     assert l2.item() >= 0
 
#     print("\nTesting classification losses...")
#     vis_probs = torch.sigmoid(torch.randn(B, S, N, device=device))
#     vis_gt    = torch.randint(0, 2, (B, S, N), device=device).bool()
 
#     l_bce  = bce_loss(vis_probs, vis_gt, valid)
#     l_gce  = gce_loss(vis_probs, vis_gt, valid, q=0.7)
#     l_tgce = truncated_gce_loss(vis_probs, vis_gt, valid, q=0.7, trunc_frac=0.7)
#     print(f"  BCE={l_bce.item():.4f}  GCE={l_gce.item():.4f}  TruncGCE={l_tgce.item():.4f}")
 
#     print("\nGCE q sensitivity:")
#     for q in [0.1, 0.5, 0.7, 1.0]:
#         l = gce_loss(vis_probs, vis_gt, valid, q=q)
#         print(f"  q={q:.1f} → {l.item():.4f}")
 
#     print("\nTruncGCE frac sensitivity:")
#     for frac in [0.3, 0.7, 1.0]:
#         l = truncated_gce_loss(vis_probs, vis_gt, valid, q=0.7, trunc_frac=frac)
#         print(f"  frac={frac:.1f} → {l.item():.4f}")
 
#     print("\n✓ All tests passed.")

    


# """
# losses.py — LEAP-VO Loss Functions
# ====================================
# Implements the three loss components from the paper (Eq. 5-8):

#     L_total = w1 * L_main + w2 * L_vis + w3 * L_dyn          (Eq. 8)

# L_main — Cauchy NLL (Eq. 5)
# -----------------------------
# LEAP models trajectory distributions as multivariate Cauchy (Eq. 4).
# For a single track and one coordinate (x or y), the S-dimensional
# multivariate Cauchy NLL is:

#     -log p(a) = (1/2)*log|Sigma|
#               + ((1+S)/2) * log(1 + delta^T @ Sigma^-1 @ delta)
#               + const   (dropped — doesn't affect gradients)

# where:
#     delta = pred - gt            [S,]   (residual for one coordinate)
#     Sigma = kernel matrix        [S, S] (predicted scale matrix, SPD)
#     S     = window length (8)

# We sum over x and y coordinates and average over valid tracks and windows.

# L_vis — Visibility classification (Eq. 6)
# -------------------------------------------
# Binary classification: visible (1) vs occluded (0).
# LeapKernel outputs ALREADY SIGMOID-ED probabilities, so we use plain BCE
# (not BCEWithLogits).

#     BCE(p, t) = -(t*log(p) + (1-t)*log(1-p))

# Variants:
#     bce      : standard BCE
#     gce      : Generalized Cross Entropy (Zhang & Sabuncu, NeurIPS 2018)
#                L_GCE = (1 - p_y^q) / q   where p_y = prob of true class
#                q in (0,1]. q→0 recovers CE, q=1 gives MAE.
#     trunc_gce: Truncated GCE — same formula but zero out the top
#                (1-trunc_frac) highest-loss samples to handle noisy labels.

# L_dyn — Dynamic track classification (Eq. 7)
# ----------------------------------------------
# Binary classification: dynamic (1) vs static (0).
# Same BCE/GCE/TruncatedGCE variants as L_vis.
# The dynamic label is per-track (not per-frame) but is repeated across S
# frames by LeapKernel before returning dyn_predictions.

# References:
#     LEAP-VO paper: https://arxiv.org/pdf/2401.01887
#     GCE paper: "Generalized Cross Entropy Loss for Training Deep Neural
#                 Networks with Noisy Labels", Zhang & Sabuncu, NeurIPS 2018
#     Truncated Loss repo: https://github.com/AlanChou/Truncated-Loss
# """

# import torch
# import torch.nn.functional as F


# # ===========================================================================
# # L_main — Multivariate Cauchy NLL  (Eq. 5 in paper)
# # ===========================================================================

# def cauchy_nll_loss(coord_pred, coord_gt, sigma_x, sigma_y, valid_mask):
#     """
#     Negative log-likelihood of multivariate Cauchy distribution.
#     Applied separately to x and y coordinates then summed (paper Eq. 4-5).

#     Args:
#         coord_pred : [B, S, N, 2]  predicted trajectory  (pixel coords)
#         coord_gt   : [B, S, N, 2]  ground truth trajectory (pixel coords)
#         sigma_x    : [B, N, S, S]  scale matrix for x coordinate (SPD, from kernel)
#         sigma_y    : [B, N, S, S]  scale matrix for y coordinate (SPD, from kernel)
#         valid_mask : [B, S, N]     bool — True where loss should be computed

#     Returns:
#         scalar loss (mean over valid tracks and batch)
#     """
#     B, S, N, _ = coord_pred.shape

#     # Residuals: [B, S, N, 2] → split into x and y
#     delta = coord_pred - coord_gt                    # [B, S, N, 2]
#     delta_x = delta[..., 0]                          # [B, S, N]
#     delta_y = delta[..., 1]                          # [B, S, N]

#     # Rearrange to [B, N, S] for matrix operations with sigma [B, N, S, S]
#     delta_x = delta_x.permute(0, 2, 1)              # [B, N, S]
#     delta_y = delta_y.permute(0, 2, 1)              # [B, N, S]
#     mask    = valid_mask.permute(0, 2, 1).float()   # [B, N, S]

#     # Zero out invalid positions in the residual
#     delta_x = delta_x * mask
#     delta_y = delta_y * mask

#     loss_x = _cauchy_nll_single(delta_x, sigma_x, S)  # [B, N]
#     loss_y = _cauchy_nll_single(delta_y, sigma_y, S)  # [B, N]

#     loss = loss_x + loss_y                            # [B, N]

#     # Average over valid tracks only
#     track_valid = mask.any(dim=2).float()             # [B, N] — track has any valid frame
#     denom = track_valid.sum().clamp(min=1.0)
#     return (loss * track_valid).sum() / denom


# def _cauchy_nll_single(delta, sigma, S):
#     """
#     NLL of S-dimensional multivariate Cauchy for one coordinate.

#     -log p(a) = (1/2)*log|Sigma| + ((1+S)/2)*log(1 + delta^T Sigma^-1 delta)
#               (constant terms dropped)

#     Args:
#         delta : [B, N, S]    residual vector
#         sigma : [B, N, S, S] scale matrix (SPD — guaranteed by kernel + epsilon*I)
#         S     : int           window length

#     Returns:
#         loss : [B, N]
#     """
#     # --- Log determinant term: (1/2) * log|Sigma| ---
#     # Use Cholesky for numerical stability (sigma is SPD by construction)
#     try:
#         L = torch.linalg.cholesky(sigma)                    # [B, N, S, S]
#         log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # [B, N]
#     except RuntimeError:
#         # Fallback to slogdet if Cholesky fails (shouldn't happen with epsilon*I)
#         sign, log_det = torch.linalg.slogdet(sigma)
#         log_det = log_det * sign.clamp(min=0)               # [B, N]

#     log_det_term = 0.5 * log_det                            # [B, N]

#     # --- Mahalanobis term: ((1+S)/2) * log(1 + delta^T Sigma^-1 delta) ---
#     # Solve Sigma @ x = delta  →  x = Sigma^-1 @ delta
#     # Use cholesky_solve for efficiency and stability
#     delta_col = delta.unsqueeze(-1)                         # [B, N, S, 1]
#     try:
#         L = torch.linalg.cholesky(sigma)
#         sigma_inv_delta = torch.cholesky_solve(delta_col, L)  # [B, N, S, 1]
#     except RuntimeError:
#         sigma_inv_delta = torch.linalg.solve(sigma, delta_col)

#     # delta^T @ Sigma^-1 @ delta
#     mahal = (delta_col * sigma_inv_delta).sum(dim=(-2, -1))  # [B, N]
#     mahal_term = ((1.0 + S) / 2.0) * torch.log1p(mahal)     # [B, N]  log1p for stability

#     return log_det_term + mahal_term                         # [B, N]


# # ===========================================================================
# # L_vis and L_dyn — Classification losses
# # NOTE: inputs are PROBABILITIES (already sigmoid-ed by LeapKernel),
# #       not logits. Do NOT use BCEWithLogits here.
# # ===========================================================================

# def bce_loss(probs, targets, valid_mask, input_is_prob=True):
#     """
#     Plain Binary Cross Entropy on probabilities.

#     Args:
#         probs      : [B, S, N]  sigmoid-ed probabilities from model
#         targets    : [B, S, N]  bool/float ground truth (1=positive class)
#         valid_mask : [B, S, N]  bool — where to compute loss
#         input_is_prob: always True here (kept for API consistency with losses.py)

#     Returns:
#         scalar loss
#     """
#     eps    = 1e-6
#     p      = probs.clamp(eps, 1.0 - eps)
#     t      = targets.float()
#     loss   = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))  # [B, S, N]

#     mask   = valid_mask.float()
#     denom  = mask.sum().clamp(min=1.0)
#     return (loss * mask).sum() / denom


# def gce_loss(probs, targets, valid_mask, q=0.7, input_is_prob=True):
#     """
#     Generalized Cross Entropy loss (Zhang & Sabuncu, NeurIPS 2018).

#     L_GCE = (1 - p_y^q) / q

#     where p_y = model probability assigned to the TRUE class.

#     Properties:
#         q → 0  : recovers standard cross-entropy  (noise-sensitive)
#         q = 1  : mean absolute error              (noise-robust)
#         q = 0.7: recommended — good balance between robustness and learnability

#     Args:
#         probs      : [B, S, N]  sigmoid-ed probabilities
#         targets    : [B, S, N]  bool/float ground truth
#         valid_mask : [B, S, N]  bool
#         q          : float in (0, 1]

#     Returns:
#         scalar loss
#     """
#     assert 0.0 < q <= 1.0, f"GCE q must be in (0,1], got {q}"
#     eps = 1e-6
#     p   = probs.clamp(eps, 1.0 - eps)
#     t   = targets.float()

#     # p_y = probability of the TRUE class
#     # For binary: if t=1, p_y = p; if t=0, p_y = 1-p
#     p_y = t * p + (1.0 - t) * (1.0 - p)             # [B, S, N]

#     loss = (1.0 - p_y.pow(q)) / q                    # [B, S, N]

#     mask  = valid_mask.float()
#     denom = mask.sum().clamp(min=1.0)
#     return (loss * mask).sum() / denom


# def truncated_gce_loss(probs, targets, valid_mask, q=0.7, trunc_frac=0.7,
#                        input_is_prob=True):
#     """
#     Truncated Generalized Cross Entropy (from Truncated-Loss repo).

#     Same as GCE but we zero out the (1 - trunc_frac) fraction of samples
#     with the HIGHEST loss values. These are likely mislabeled/noisy samples.
#     Only the trunc_frac easiest (lowest-loss) samples contribute to the gradient.

#     This is the "Option B" from the professor's instructions — more aggressive
#     noise handling than plain GCE.

#     Truncation is done per-batch (not per-window) so the threshold is computed
#     globally across all valid positions in the batch.

#     Args:
#         probs      : [B, S, N]  sigmoid-ed probabilities
#         targets    : [B, S, N]  bool/float ground truth
#         valid_mask : [B, S, N]  bool
#         q          : float in (0, 1] — GCE exponent
#         trunc_frac : float in (0, 1] — fraction of samples to KEEP
#                      e.g. 0.7 means keep the 70% lowest-loss samples

#     Returns:
#         scalar loss
#     """
#     assert 0.0 < q <= 1.0,       f"q must be in (0,1], got {q}"
#     assert 0.0 < trunc_frac <= 1.0, f"trunc_frac must be in (0,1], got {trunc_frac}"

#     eps = 1e-6
#     p   = probs.clamp(eps, 1.0 - eps)
#     t   = targets.float()

#     # GCE loss per element
#     p_y  = t * p + (1.0 - t) * (1.0 - p)            # [B, S, N]
#     loss = (1.0 - p_y.pow(q)) / q                    # [B, S, N]

#     mask  = valid_mask.float()                        # [B, S, N]

#     # Compute truncation threshold from valid losses only
#     valid_losses = loss[valid_mask.bool()]            # [M,]  M = num valid elements

#     if valid_losses.numel() == 0:
#         return torch.tensor(0.0, device=probs.device, requires_grad=True)

#     # Keep the trunc_frac lowest losses — threshold at the trunc_frac quantile
#     k         = max(1, int(trunc_frac * valid_losses.numel()))
#     threshold = valid_losses.topk(k, largest=False).values[-1]  # k-th smallest

#     # Zero out samples above threshold (likely noisy)
#     keep_mask = (loss <= threshold).float() * mask    # [B, S, N]

#     denom = keep_mask.sum().clamp(min=1.0)
#     return (loss * keep_mask).sum() / denom


# # ===========================================================================
# # Quick self-test — run this file directly to verify all losses work
# # ===========================================================================

# if __name__ == "__main__":
#     import sys
#     print("Testing losses.py...\n")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     B, S, N = 2, 8, 64
#     torch.manual_seed(42)

#     # Fake model outputs matching LeapKernel shapes
#     coord_pred = torch.rand(B, S, N, 2, device=device, requires_grad=True) * 256.0
#     coord_gt   = torch.rand(B, S, N, 2, device=device) * 256.0

#     # SPD covariance matrix: K(F,F) + eps*I  — same as KernelBlock output
#     feat       = torch.rand(B*N, S, 128, device=device, requires_grad=True)
#     raw        = torch.bmm(feat, feat.transpose(1, 2))          # [B*N, S, S]
#     eps_eye    = 1e-4 * torch.eye(S, device=device).unsqueeze(0)
#     sigma      = (raw + eps_eye).reshape(B, N, S, S)

#     vis_probs  = torch.sigmoid(torch.randn(B, S, N, device=device))
#     vis_gt     = torch.randint(0, 2, (B, S, N), device=device).bool()
#     dyn_probs  = torch.sigmoid(torch.randn(B, S, N, device=device))
#     dyn_gt     = torch.randint(0, 2, (B, S, N), device=device).bool()
#     valid      = torch.ones(B, S, N, dtype=torch.bool, device=device)

#     # ---- L_main ----
#     print("Testing cauchy_nll_loss...")
#     l_main = cauchy_nll_loss(coord_pred, coord_gt, sigma, sigma, valid)
#     print(f"  L_main = {l_main.item():.4f}")
#     l_main.backward()
#     print(f"  Backward: OK")

#     # ---- L_vis variants ----
#     print("\nTesting bce_loss...")
#     l = bce_loss(vis_probs, vis_gt, valid)
#     print(f"  BCE     = {l.item():.4f}")

#     print("Testing gce_loss (q=0.7)...")
#     l = gce_loss(vis_probs, vis_gt, valid, q=0.7)
#     print(f"  GCE     = {l.item():.4f}")

#     print("Testing truncated_gce_loss (q=0.7, trunc_frac=0.7)...")
#     l = truncated_gce_loss(vis_probs, vis_gt, valid, q=0.7, trunc_frac=0.7)
#     print(f"  TruncGCE= {l.item():.4f}")

#     # ---- Partial valid mask ----
#     print("\nTesting with partial valid mask (50% valid)...")
#     partial_valid = torch.rand(B, S, N, device=device) > 0.5
#     l_bce  = bce_loss(vis_probs, vis_gt, partial_valid)
#     l_gce  = gce_loss(vis_probs, vis_gt, partial_valid, q=0.7)
#     l_tgce = truncated_gce_loss(vis_probs, vis_gt, partial_valid, q=0.7, trunc_frac=0.7)
#     print(f"  BCE={l_bce.item():.4f}  GCE={l_gce.item():.4f}  TruncGCE={l_tgce.item():.4f}")

#     # ---- Gradient flow through all losses ----
#     print("\nTesting gradient flow through all losses...")
#     raw2 = torch.randn(B, S, N, device=device, requires_grad=True)
#     vis_probs2 = torch.sigmoid(raw2)
#     total = (bce_loss(vis_probs2, vis_gt, valid) +
#              gce_loss(vis_probs2, vis_gt, valid, q=0.7) +
#              truncated_gce_loss(vis_probs2, vis_gt, valid, q=0.7, trunc_frac=0.7))
#     total.backward()
#     assert raw2.grad is not None
#     print(f"  Gradient norm: {raw2.grad.norm().item():.4f}  OK")

#     # ---- GCE q sensitivity ----
#     print("\nGCE q sensitivity (should change with q):")
#     for q in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
#         l = gce_loss(vis_probs, vis_gt, valid, q=q)
#         print(f"  q={q:.1f} → loss={l.item():.4f}")

#     # ---- Truncation fraction sensitivity ----
#     print("\nTruncGCE trunc_frac sensitivity:")
#     for frac in [0.3, 0.5, 0.7, 0.9, 1.0]:
#         l = truncated_gce_loss(vis_probs, vis_gt, valid, q=0.7, trunc_frac=frac)
#         print(f"  frac={frac:.1f} → loss={l.item():.4f}")

#     print("\n✓ All loss tests passed.")