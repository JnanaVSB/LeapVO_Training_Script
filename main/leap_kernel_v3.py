"""
leap_kernel_v3.py — LEAP-VO with CoTracker v3 backbone
========================================================
Drop-in replacement for leap_kernel.py (LeapKernel → LeapKernelV3).

Architecture:
    - fnet: BasicEncoder (identical to v1, output_dim=128, stride=4)
    - Correlation: v3's EfficientCorrBlock + corr_mlp (replaces v1's CorrBlock)
    - Transformer: v3's EfficientUpdateFormer with virtual-token cross-attention
      (depth 3+3, replaces v1's UpdateFormer depth 6+6)
    - feat_head: Linear(384 → 128) — projects v3 hidden tokens to feature space
      so LEAP's heads can consume them

    Everything below is unchanged from LeapKernel:
    - kernel_block + var_predictors → Cauchy covariance
    - vis_predictor → visibility
    - motion_label_block → dynamic track detection
    - ffeat_updater + norm → feature accumulation
    - forward() outer loop — windowing, anchors, sort_inds

The forward_iteration() inner loop follows v3's pattern for:
    - L2-normalized feature maps
    - Feature pyramid + track_feat_support extraction
    - On-the-fly correlation via einsum + corr_mlp
    - Transformer input: [vis, confidence, corr_embs, rel_pos_emb] (1110-dim)
    - Time embedding via interpolation

But at each iteration it also:
    - Projects hidden tokens (384-dim) → 128-dim features via feat_head
    - Accumulates features via ffeat_updater (same as v1 LeapKernel)
    - Computes covariance via kernel_block + var_predictors (same as v1)

After the loop:
    - vis_predictor on accumulated features (same as v1)
    - motion_label_block on accumulated features (same as v1)

Output signature is IDENTICAL to LeapKernel.forward_iteration() and
LeapKernel.forward(), so train.py works with no changes beyond
instantiating LeapKernelV3 instead of LeapKernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

# --- v3 backbone imports ---
# These come from the CoTracker3 repo (cotracker/models/core/...)
# We import them from the LEAP-bundled copy which will be updated to include v3 blocks.
# For now, we import from cotracker3's source directly and also from LEAP's existing code.
from leap.core.cotracker.blocks import (
    BasicEncoder,
    CorrBlock,
    MotionLabelBlock,
)
from leap.core.embeddings import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_embedding,
    get_2d_sincos_pos_embed,
)
from leap.core.model_utils import bilinear_sample2d, meshgrid2d, smart_cat

torch.manual_seed(0)


# ============================================================================
# V3 components — inlined here to avoid import path issues
# ============================================================================

def posenc(x, min_deg, max_deg):
    """Positional encoding from CoTracker3 (sinusoidal, multi-scale).

    Args:
        x: [..., D] tensor to encode.
        min_deg, max_deg: frequency range.
    Returns:
        [..., D + D * 2 * (max_deg - min_deg)] encoded tensor.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )
    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def bilinear_sampler_5d(input, coords, align_corners=True, padding_mode="border"):
    """Sample from 5D tensor [B, C, D, H, W] using 3D coordinates.

    Adapted from CoTracker3's model_utils.bilinear_sampler for 5D inputs.
    coords: [B*T, N, 1, 1, 3] or [B, N, H', W', 3] — last dim is (d, h, w).
    """
    # Use grid_sample with 5D input
    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode, mode="bilinear"
    )


def sample_features5d(fmaps, points):
    """Sample features from 5D feature maps at given 3D points.

    Args:
        fmaps: [B, T, C, H, W] feature maps
        points: [B, P, N, 3] — each point is (frame_idx, x, y) in pixel coords

    Returns:
        [B, P, N, C] sampled features
    """
    B, T, C, H, W = fmaps.shape
    P = points.shape[1]
    N = points.shape[2]

    # Normalize coordinates to [-1, 1] for grid_sample
    points_norm = points.clone()
    points_norm[..., 0] = 2.0 * points_norm[..., 0] / max(T - 1, 1) - 1.0  # frame
    points_norm[..., 1] = 2.0 * points_norm[..., 1] / max(H - 1, 1) - 1.0  # y
    points_norm[..., 2] = 2.0 * points_norm[..., 2] / max(W - 1, 1) - 1.0  # x

    # fmaps: [B, T, C, H, W] → [B, C, T, H, W] for grid_sample 5D
    fmaps_5d = fmaps.permute(0, 2, 1, 3, 4)

    # points_norm: [B, P, N, 3] → [B, P, N, 1, 3] for grid_sample
    grid = points_norm.unsqueeze(3)  # [B, P, N, 1, 3]

    sampled = F.grid_sample(
        fmaps_5d, grid, align_corners=True, padding_mode="border", mode="bilinear"
    )
    # sampled: [B, C, P, N, 1] → [B, P, N, C]
    sampled = sampled.squeeze(-1).permute(0, 2, 3, 1)
    return sampled


class V3Attention(nn.Module):
    """Attention module from CoTracker3 blocks.py."""
    def __init__(self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = num_heads
        self.num_heads = num_heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape
        h = self.heads
        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)
        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        sim = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            sim = sim + attn_bias
        attn = sim.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return self.to_out(x)


class V3Mlp(nn.Module):
    """MLP from CoTracker3 blocks.py."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class V3AttnBlock(nn.Module):
    """AttnBlock from CoTracker3 blocks.py."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = V3Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = V3Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                         act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

    def forward(self, x, mask=None):
        attn_bias = mask
        if mask is not None:
            mask = (
                (mask[:, None] * mask[:, :, None])
                .unsqueeze(1)
                .expand(-1, self.attn.num_heads, -1, -1)
            )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class V3CrossAttnBlock(nn.Module):
    """CrossAttnBlock from CoTracker3 cotracker.py."""
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = V3Attention(
            hidden_size, context_dim=context_dim, num_heads=num_heads, qkv_bias=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = V3Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                         act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

    def forward(self, x, context, mask=None):
        attn_bias = None
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x


class EfficientUpdateFormerV3(nn.Module):
    """
    CoTracker v3's EfficientUpdateFormer, modified to also return
    the 384-dim hidden tokens (before flow_head projection) so that
    LEAP's heads can extract features from them.

    Changes from original:
        - forward() returns (flow, hidden_tokens) instead of just flow
        - hidden_tokens: [B, N_real, T, hidden_size] — the tokens after
          all attention blocks, with virtual tokens stripped
    """

    def __init__(
        self,
        space_depth=3,
        time_depth=3,
        input_dim=1110,
        hidden_size=384,
        num_heads=8,
        output_dim=4,
        mlp_ratio=4.0,
        num_virtual_tracks=64,
        add_space_attn=True,
        linear_layer_for_vis_conf=True,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf

        if linear_layer_for_vis_conf:
            self.flow_head = nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)

        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)
        )
        self.add_space_attn = add_space_attn

        self.time_blocks = nn.ModuleList([
            V3AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(time_depth)
        ])

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList([
                V3AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(space_depth)
            ])
            self.space_point2virtual_blocks = nn.ModuleList([
                V3CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(space_depth)
            ])
            self.space_virtual2point_blocks = nn.ModuleList([
                V3CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(space_depth)
            ])
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None, add_space_attn=True):
        """
        Args:
            input_tensor: [B, N, T, input_dim]
            mask: optional attention mask
            add_space_attn: whether to use spatial attention

        Returns:
            flow: [B, N, T, output_dim] — coordinate deltas + vis/conf
            hidden_tokens: [B, N, T, hidden_size] — raw transformer tokens
                           for feat_head projection (LEAP addition)
        """
        tokens = self.input_transform(input_tensor)

        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape
        j = 0
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)
            if (
                add_space_attn
                and hasattr(self, "space_virtual_blocks")
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            ):
                space_tokens = tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)

                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)
                j += 1

        # Strip virtual tokens — keep only real point tokens
        tokens = tokens[:, : N - self.num_virtual_tracks]

        # Project to flow output
        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        # Return both flow AND raw hidden tokens
        return flow, tokens


# ============================================================================
# Helper functions (reused from leap_kernel.py)
# ============================================================================

def get_points_on_a_grid(grid_size, interp_shape, grid_center=(0, 0), device="cpu"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, interp_shape[0] / 2], device=device)[
            None, None
        ]
    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )
    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy


# ============================================================================
# Kernel blocks (identical to leap_kernel.py)
# ============================================================================

class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()
        self.const = nn.Parameter(torch.ones(1))

    def forward(self, x1, x2, sigma=1e-4):
        numerator = torch.sum(x1.unsqueeze(2) * x2.unsqueeze(1), dim=-1)
        return numerator + self.const


class RBFKernel(nn.Module):
    def __init__(self, input_dim):
        super(RBFKernel, self).__init__()
        self.scale = nn.Parameter(torch.ones(input_dim))

    def forward(self, x1, x2):
        distance = torch.sum((x1.unsqueeze(2) - x2.unsqueeze(1)) ** 2, dim=-1)
        return torch.exp(-distance / (2 * self.scale ** 2))


class KernelBlock(nn.Module):
    def __init__(self, cfg):
        super(KernelBlock, self).__init__()
        self.kernel_list = cfg.kernel_block.kernel_list
        self.composition = cfg.kernel_block.composition
        assert self.composition in ["sum", "product"]
        kernel_nets = []
        for kernel in self.kernel_list:
            if kernel == "linear":
                kernel_nets.append(LinearKernel())
            elif kernel == "rbf":
                kernel_nets.append(RBFKernel(input_dim=cfg.S))
        self.kernels = nn.ModuleList(kernel_nets)

    def forward(self, features, epsilon=1e-5):
        if self.composition == "sum":
            K = 0
            for kernel in self.kernels:
                K = K + kernel(features, features)
        elif self.composition == "product":
            K = 1
            for kernel in self.kernels:
                K = K * kernel(features, features)
        K = K + epsilon * torch.eye(K.size(-1)).unsqueeze(0).to(K.device)
        return K


# ============================================================================
# LeapKernelV3 — the main model
# ============================================================================

class LeapKernelV3(nn.Module):
    def __init__(self, cfg, stride=4):
        super(LeapKernelV3, self).__init__()
        self.cfg = cfg.model
        self.S = self.cfg.sliding_window_len
        self.stride = stride

        if "anchor_aug" in cfg:
            self.anchor_aug = cfg.anchor_aug
        else:
            self.anchor_aug = None

        self.kernel_from_delta = True
        if "kernel_from_delta" in self.cfg:
            self.kernel_from_delta = self.cfg.kernel_from_delta

        self.interp_shape = (384, 512)
        self.hidden_dim = self.cfg.hidden_dim if "hidden_dim" in self.cfg else 256
        self.latent_dim = self.cfg.latent_dim if "latent_dim" in self.cfg else 128
        self.corr_levels = self.cfg.corr_levels if "corr_levels" in self.cfg else 4
        self.corr_radius = self.cfg.corr_radius if "corr_radius" in self.cfg else 3

        self.add_space_attn = self.cfg.add_space_attn

        # --- v3 model resolution (needed for rel_pos_emb scaling) ---
        self.model_resolution = (384, 512)

        # --- v3 backbone components ---
        # fnet: identical to v1
        self.fnet = BasicEncoder(
            output_dim=self.latent_dim, norm_fn="instance", dropout=0, stride=stride
        )

        # v3 transformer config
        v3_cfg = self.cfg.get("v3", {})
        v3_space_depth = v3_cfg.get("space_depth", 3)
        v3_time_depth = v3_cfg.get("time_depth", 3)
        v3_hidden_size = v3_cfg.get("hidden_size", 384)
        v3_num_heads = v3_cfg.get("num_heads", 8)
        v3_num_virtual_tracks = v3_cfg.get("num_virtual_tracks", 64)
        v3_input_dim = 1110  # fixed by v3's input construction

        self.updateformer = EfficientUpdateFormerV3(
            space_depth=v3_space_depth,
            time_depth=v3_time_depth,
            input_dim=v3_input_dim,
            hidden_size=v3_hidden_size,
            output_dim=4,  # 2 coords + 2 vis/conf (v3 standard)
            mlp_ratio=4.0,
            num_virtual_tracks=v3_num_virtual_tracks,
            add_space_attn=self.add_space_attn,
            linear_layer_for_vis_conf=True,
        )

        # v3 correlation MLP: 49*49 → 384 → 256
        r = 2 * self.corr_radius + 1  # 7
        self.corr_mlp = V3Mlp(
            in_features=r * r * r * r,  # 49 * 49 = 2401
            hidden_features=v3_hidden_size,
            out_features=256,
        )

        # v3 time embedding (registered as buffer)
        time_grid = torch.linspace(0, self.S - 1, self.S).reshape(1, self.S, 1)
        self.register_buffer(
            "time_emb",
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(v3_input_dim, time_grid[0].numpy())
            ).float().unsqueeze(0)
        )

        # --- LEAP addition: project v3 hidden tokens → feature space ---
        self.feat_head = nn.Sequential(
            nn.Linear(v3_hidden_size, self.latent_dim),
            nn.GELU(),
        )

        # --- LEAP heads (identical to LeapKernel) ---
        self.kernel_block = KernelBlock(cfg=self.cfg)

        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

        self.var_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Softplus(),
            )
            for _ in range(2)  # x and y
        ])

        if "motion_label_block" in self.cfg:
            self.motion_label_block = MotionLabelBlock(cfg=self.cfg, S=self.S)
        else:
            self.motion_label_block = None

    # ------------------------------------------------------------------
    # V3 helper methods (from CoTrackerThreeBase)
    # ------------------------------------------------------------------

    def get_support_points(self, coords, r, reshape_back=True):
        """Get support point grid around each query."""
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)
        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)
        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
        delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl
        if reshape_back:
            return coords_lvl.reshape(B, N, (2 * r + 1) ** 2, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl

    def get_track_feat(self, fmaps, queried_frames, queried_coords, support_radius=0):
        """Extract track features and support features from feature maps."""
        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat([sample_frames, queried_coords[:, None]], dim=-1)
        support_points = self.get_support_points(sample_coords, support_radius)
        support_track_feats = sample_features5d(fmaps, support_points)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],
            support_track_feats,
        )

    def get_correlation_feat(self, fmaps, queried_coords):
        """Get correlation feature patches around current coordinates.

        Args:
            fmaps: [B, T, D, H, W] feature maps
            queried_coords: [B*T, N, 2] current coordinates in stride space

        Returns:
            [B, T, N, r', r', D] correlation features
        """
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius

        # Build support point grid: prepend zero for depth dim
        # queried_coords: [B*T, N, 2] → [B*T, 1, N, 3] with t=0
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]  # [B*T, 1, N, 3]

        support_points = self.get_support_points(sample_coords, r, reshape_back=False)
        # support_points: [B*T, N, 2r+1, 2r+1, 3]

        # Use v3's bilinear_sampler convention: coords in pixel space,
        # normalized internally by the sampler
        fmaps_5d = fmaps.reshape(B * T, D, 1, H_, W_)

        # Normalize coords for grid_sample: (t,x,y) → grid_sample expects (x,y,t) in [-1,1]
        # t coord: depth is 1, so normalize to [-1,1] → always 0 maps to 0
        # x,y coords: in stride-space pixel coords
        sizes = fmaps_5d.shape[2:]  # (1, H_, W_)
        sp = support_points.clone()
        # Reorder from (t, x, y) to (x, y, t) for grid_sample
        sp = sp[..., [1, 2, 0]]
        # Normalize to [-1, 1]
        sp[..., 0] = sp[..., 0] * 2.0 / max(W_ - 1, 1) - 1.0   # x
        sp[..., 1] = sp[..., 1] * 2.0 / max(H_ - 1, 1) - 1.0   # y
        sp[..., 2] = sp[..., 2] * 2.0 / max(sizes[0] - 1, 1) - 1.0  # t (depth=1)

        sampled = F.grid_sample(
            fmaps_5d, sp, align_corners=True, padding_mode="border", mode="bilinear"
        )
        # sampled: [B*T, D, N, 2r+1, 2r+1]
        return sampled.view(B, T, D, N, 2 * r + 1, 2 * r + 1).permute(0, 1, 3, 4, 5, 2)

    def interpolate_time_embed(self, x, t):
        """Interpolate time embedding to match sequence length."""
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]
        if t == T:
            return self.time_emb
        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb.to(previous_dtype)

    # ------------------------------------------------------------------
    # forward_iteration — v3 backbone loop + LEAP heads
    # ------------------------------------------------------------------

    def forward_iteration(
        self,
        fmaps,
        coords_init,
        feat_init=None,
        vis_init=None,
        track_mask=None,
        iters=4,
        queried_frames=None,
        queried_coords=None,
    ):
        """
        Inner iterative refinement loop.

        Uses v3's correlation + transformer architecture but extracts
        128-dim features for LEAP's heads at each iteration.

        Args:
            fmaps:          [B, S, C, H8, W8] feature maps (L2-normalized)
            coords_init:    [B, S, N, 2] initial coordinates (stride space)
            feat_init:      [B, S, N, latent_dim] initial features
            vis_init:       [B, S, N, 1] initial visibility logits
            track_mask:     [B, S, N, 1] track validity mask
            iters:          number of refinement iterations
            queried_frames: [B, N] frame indices for each query (v3 needs this)
            queried_coords: [B, N, 2] query coordinates in stride space

        Returns:
            coord_predictions: list[K] of [B, S, N, 2] in pixel space
            vars_predictions:  list[K] of [sigma_x, sigma_y], each [B, N, S, S]
            vis_e:             [B, S, N] visibility logits
            dynamic_e:         [B, N] dynamic track logits
            feat_init:         [B, S, N, latent_dim] (unchanged, passed through)
        """
        B, S_init, N, D = coords_init.shape
        assert D == 2
        assert B == 1

        B, S, __, H8, W8 = fmaps.shape
        device = fmaps.device

        if S_init < S:
            coords = torch.cat(
                [coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
        else:
            coords = coords_init.clone()

        # --- Build feature pyramid (v3 style) ---
        fmaps_pyramid = [fmaps]
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps_pyramid[-1].reshape(B * S, self.latent_dim, fmaps_pyramid[-1].shape[-2], fmaps_pyramid[-1].shape[-1])
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps_pyramid.append(fmaps_.reshape(B, S, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]))

        # --- Extract track support features (v3 style, done once) ---
        track_feat_support_pyramid = []
        if queried_frames is not None and queried_coords is not None:
            for i in range(self.corr_levels):
                _, track_feat_support = self.get_track_feat(
                    fmaps_pyramid[i],
                    queried_frames,
                    queried_coords / 2 ** i,
                    support_radius=self.corr_radius,
                )
                track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))
        else:
            # Fallback: use coords_init[:, 0] as queries
            q_frames = torch.zeros(B, N, device=device, dtype=torch.long)
            q_coords = coords_init[:, 0]  # [B, N, 2]
            for i in range(self.corr_levels):
                _, track_feat_support = self.get_track_feat(
                    fmaps_pyramid[i],
                    q_frames,
                    q_coords / 2 ** i,
                    support_radius=self.corr_radius,
                )
                track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))

        r = 2 * self.corr_radius + 1

        # --- Initialize accumulators ---
        vis = torch.zeros((B, S, N), device=device).float()
        confidence = torch.zeros((B, S, N), device=device).float()
        ffeats = feat_init.clone() if feat_init is not None else torch.zeros(B, S, N, self.latent_dim, device=device)

        coord_predictions = []
        vars_predictions = []

        # --- Iterative refinement (v3 backbone + LEAP heads) ---
        for it in range(iters):
            coords = coords.detach()  # B, S, N, 2

            # -- v3 correlation (process one level at a time, free intermediates) --
            coords_flat = coords.reshape(B * S, N, 2)
            corr_embs = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_flat / 2 ** i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                # Use float16 for the heavy einsum to halve memory
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    corr_volume = torch.einsum(
                        "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                    )
                del corr_feat  # free immediately
                corr_emb = self.corr_mlp(corr_volume.float().reshape(B * S * N, r * r * r * r))
                del corr_volume  # free immediately
                corr_embs.append(corr_emb)
            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.view(B, S, N, corr_embs.shape[-1])

            # -- v3 transformer input construction --
            transformer_input = [vis[..., None], confidence[..., None], corr_embs]

            # Relative position encoding
            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]
            rel_coords_forward = F.pad(rel_coords_forward, (0, 0, 0, 0, 0, 1))
            rel_coords_backward = F.pad(rel_coords_backward, (0, 0, 0, 0, 1, 0))

            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=device,
                ) / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )
            transformer_input.append(rel_pos_emb_input)

            # Assemble input: [B, S, N, 1110]
            x = torch.cat(transformer_input, dim=-1)
            x = x.permute(0, 2, 1, 3).reshape(B * N, S, -1)  # [B*N, S, 1110]
            x = x + self.interpolate_time_embed(x, S)
            x = x.view(B, N, S, -1)  # [B, N, S, 1110]

            # -- v3 transformer forward --
            delta, hidden_tokens = self.updateformer(x, add_space_attn=self.add_space_attn)
            # delta: [B, N, S, 4] — coord(2) + vis(1) + conf(1)
            # hidden_tokens: [B, N, S, 384]

            delta_coords = delta[..., :2].permute(0, 2, 1, 3)  # [B, S, N, 2]
            delta_vis = delta[..., 2].permute(0, 2, 1)          # [B, S, N]
            delta_confidence = delta[..., 3].permute(0, 2, 1)   # [B, S, N]

            vis = vis + delta_vis
            confidence = confidence + delta_confidence
            coords = coords + delta_coords

            coord_predictions.append(coords * self.stride)

            # -- LEAP: extract features from hidden tokens --
            # hidden_tokens: [B, N, S, 384] → project to [B, N, S, 128]
            delta_feats = self.feat_head(hidden_tokens)  # [B, N, S, 128]

            # Accumulate features (same as v1 LeapKernel)
            delta_feats_flat = delta_feats.reshape(B * N * S, self.latent_dim)
            ffeats_flat = ffeats.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)
            ffeats_flat = self.ffeat_updater(self.norm(delta_feats_flat)) + ffeats_flat
            ffeats = ffeats_flat.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)

            # -- LEAP: compute covariance (kernel) matrices --
            if self.kernel_from_delta:
                kernel_feat = delta_feats.reshape(B * N, S, self.latent_dim)
            else:
                kernel_feat = rearrange(ffeats, "b s n c -> (b n) s c")

            kernel_feat_x = self.var_predictors[0](kernel_feat)
            kernel_mat_x = self.kernel_block(kernel_feat_x).reshape(B, N, S, S)
            kernel_feat_y = self.var_predictors[1](kernel_feat)
            kernel_mat_y = self.kernel_block(kernel_feat_y).reshape(B, N, S, S)
            vars_predictions.append([kernel_mat_x, kernel_mat_y])

        # -- LEAP: visibility + dynamic track prediction --
        vis_e = self.vis_predictor(
            ffeats.reshape(B * S * N, self.latent_dim)
        ).reshape(B, S, N)

        if self.motion_label_block is not None:
            dynamic_e = self.motion_label_block(ffeats, coords).squeeze(2)
        else:
            dynamic_e = torch.ones(B, N).to(device)

        return coord_predictions, vars_predictions, vis_e, dynamic_e, feat_init

    # ------------------------------------------------------------------
    # forward — outer loop (windowing, anchors, sort_inds)
    # Adapted from LeapKernel.forward() but uses v3's fnet processing
    # ------------------------------------------------------------------

    def forward(self, rgbs, queries, iters=4, feat_init=None, is_train=False):
        B, T, C, H, W = rgbs.shape
        B, N, __ = queries.shape
        device = rgbs.device
        assert B == 1

        # Sort queries by first visible frame
        first_positive_inds = queries[:, :, 0].long()
        __, sort_inds = torch.sort(first_positive_inds[0], dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[0][sort_inds]

        assert torch.allclose(
            first_positive_inds[0], first_positive_inds[0][sort_inds][inv_sort_inds]
        )

        coords_init = queries[:, :, 1:].reshape(B, 1, N, 2).repeat(
            1, self.S, 1, 1
        ) / float(self.stride)

        rgbs = 2 * (rgbs / 255.0) - 1.0

        traj_e = torch.zeros((B, T, N, 2), device=device)
        vis_e = torch.zeros((B, T, N), device=device)
        cov_x_e = torch.zeros((B, T, N), device=device)
        cov_y_e = torch.zeros((B, T, N), device=device)
        dynamic_e = torch.zeros((B, T, N), device=device)

        ind_array = torch.arange(T, device=device)
        ind_array = ind_array[None, :, None].repeat(B, 1, N)
        track_mask = (ind_array >= first_positive_inds[:, None, :]).unsqueeze(-1)
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10

        ind = 0
        track_mask_ = track_mask[:, :, sort_inds].clone()
        coords_init_ = coords_init[:, :, sort_inds].clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()

        prev_wind_idx = 0
        fmaps_ = None
        vis_predictions = []
        coord_predictions = []
        dynamic_predictions = []
        cov_predictions = []
        wind_inds = []

        while ind < T - self.S // 2:
            rgbs_seq = rgbs[:, ind: ind + self.S]
            S = S_local = rgbs_seq.shape[1]
            if S < self.S:
                rgbs_seq = torch.cat(
                    [rgbs_seq, rgbs_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                    dim=1,
                )
                S = rgbs_seq.shape[1]
            rgbs_ = rgbs_seq.reshape(B * S, C, H, W)

            # --- v3 feature extraction with L2 normalization ---
            if fmaps_ is None:
                fmaps_raw = self.fnet(rgbs_)
            else:
                fmaps_raw = torch.cat(
                    [fmaps_[self.S // 2:], self.fnet(rgbs_[self.S // 2:])], dim=0
                )

            # L2 normalize (v3 style)
            fmaps_norm = fmaps_raw.permute(0, 2, 3, 1)  # [B*S, H8, W8, C]
            fmaps_norm = fmaps_norm / torch.sqrt(
                torch.maximum(
                    torch.sum(torch.square(fmaps_norm), axis=-1, keepdims=True),
                    torch.tensor(1e-12, device=device),
                )
            )
            fmaps_ = fmaps_norm.permute(0, 3, 1, 2)  # [B*S, C, H8, W8] — keep unnormalized for next window

            fmaps = fmaps_.reshape(B, S, self.latent_dim, H // self.stride, W // self.stride)

            curr_wind_points = torch.nonzero(first_positive_sorted_inds < ind + self.S)
            if curr_wind_points.shape[0] == 0:
                ind = ind + self.S // 2
                continue
            wind_idx = curr_wind_points[-1] + 1

            # Initialize features for new points
            if wind_idx - prev_wind_idx > 0:
                fmaps_sample = fmaps[
                    :, first_positive_sorted_inds[prev_wind_idx:wind_idx] - ind
                ]
                feat_init_ = bilinear_sample2d(
                    fmaps_sample,
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 0],
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 1],
                ).permute(0, 2, 1)
                feat_init_ = feat_init_.unsqueeze(1).repeat(1, self.S, 1, 1)
                feat_init = smart_cat(feat_init, feat_init_, dim=2)

            if prev_wind_idx > 0:
                new_coords = coords[-1][:, self.S // 2:] / float(self.stride)
                coords_init_[:, : self.S // 2, :prev_wind_idx] = new_coords
                coords_init_[:, self.S // 2:, :prev_wind_idx] = new_coords[
                    :, -1
                ].repeat(1, self.S // 2, 1, 1)

                new_vis = vis[:, self.S // 2:].unsqueeze(-1)
                vis_init_[:, : self.S // 2, :prev_wind_idx] = new_vis
                vis_init_[:, self.S // 2:, :prev_wind_idx] = new_vis[:, -1].repeat(
                    1, self.S // 2, 1, 1
                )

            # --- Build queried_frames and queried_coords for v3's correlation ---
            # These are the frame indices and coordinates (in stride space) for
            # the currently active tracks, needed by get_track_feat
            active_first_frames = first_positive_sorted_inds[:wind_idx] - ind
            active_first_frames = active_first_frames.clamp(min=0, max=S - 1)
            q_frames = active_first_frames.unsqueeze(0)  # [1, wind_idx]
            q_coords = coords_init_[:, 0, :wind_idx]     # [1, wind_idx, 2]

            if self.anchor_aug is not None and is_train:
                from leap.core.anchor_sampler import get_anchors

                N_anchor = self.anchor_aug.num_anchors
                anchor_queries = get_anchors(rgbs_seq, self.anchor_aug).float()

                coords_init_anchor = repeat(
                    anchor_queries[:, :, 1:], "b n c -> b s n c", s=S
                )
                vis_init_anchor = (
                    torch.ones((B, S, N_anchor, 1), device=device).float() * 10
                )

                anchor_frame_id = anchor_queries[..., 0].view(-1).long()
                anchor_fmaps_sample = fmaps[:, anchor_frame_id]
                feat_init_anchor = bilinear_sample2d(
                    anchor_fmaps_sample,
                    anchor_queries[:, :, 1],
                    anchor_queries[:, :, 2],
                ).permute(0, 2, 1)
                feat_init_anchor = feat_init_anchor.unsqueeze(1).repeat(1, self.S, 1, 1)

                anchor_ind_array = torch.arange(S, device=device)
                anchor_ind_array = anchor_ind_array[None, :, None].repeat(B, 1, N_anchor)
                anchor_task_mask = (
                    anchor_ind_array >= anchor_frame_id[None, None, :]
                ).unsqueeze(-1)

                coords_init_all = torch.cat(
                    [coords_init_[:, :, :wind_idx], coords_init_anchor], dim=2
                )
                feat_init_all = torch.cat(
                    [feat_init[:, :, :wind_idx], feat_init_anchor], dim=2
                )
                vis_init_all = torch.cat(
                    [vis_init_[:, :, :wind_idx], vis_init_anchor], dim=2
                )
                track_mask_all = torch.cat(
                    [track_mask_[:, ind: ind + self.S, :wind_idx], anchor_task_mask],
                    dim=2,
                )

                # Build combined queried_frames/coords including anchors
                q_frames_all = torch.cat([q_frames, anchor_frame_id.unsqueeze(0)], dim=1)
                q_coords_all = torch.cat([q_coords, anchor_queries[:, :, 1:]], dim=1)

                coords_all, covs_all, vis_all, dynamic_all, __ = self.forward_iteration(
                    fmaps=fmaps,
                    coords_init=coords_init_all,
                    feat_init=feat_init_all,
                    vis_init=vis_init_all,
                    track_mask=track_mask_all,
                    iters=iters,
                    queried_frames=q_frames_all,
                    queried_coords=q_coords_all,
                )
                # Remove anchors
                coords = [x[:, :, :wind_idx] for x in coords_all]
                covs = [[x[0][:, :wind_idx], x[1][:, :wind_idx]] for x in covs_all]
                vis = vis_all[:, :, :wind_idx]
                dynamic = dynamic_all[:, :wind_idx]
            else:
                coords, covs, vis, dynamic, __ = self.forward_iteration(
                    fmaps=fmaps,
                    coords_init=coords_init_[:, :, :wind_idx],
                    feat_init=feat_init[:, :, :wind_idx],
                    vis_init=vis_init_[:, :, :wind_idx],
                    track_mask=track_mask_[:, ind: ind + self.S, :wind_idx],
                    iters=iters,
                    queried_frames=q_frames,
                    queried_coords=q_coords,
                )

            if is_train:
                vis_predictions.append(torch.sigmoid(vis[:, :S_local]))
                dynamic_predictions.append(
                    torch.sigmoid(repeat(dynamic, "b n -> b s n", s=S_local))
                )
                cov_predictions_temp = []
                for cov_mat in covs:
                    cov_predictions_temp.append([
                        cov_mat[0][:, :, :S_local, :S_local],
                        cov_mat[1][:, :, :S_local, :S_local],
                    ])
                cov_predictions.append(cov_predictions_temp)
                coord_predictions.append([coord[:, :S_local] for coord in coords])
                wind_inds.append(wind_idx)

            traj_e[:, ind: ind + self.S, :wind_idx] = coords[-1][:, :S_local]
            vis_e[:, ind: ind + self.S, :wind_idx] = vis[:, :S_local]
            cov_x_e[:, ind: ind + self.S, :wind_idx] = torch.diagonal(
                covs[-1][0][:, :, :S_local, :S_local], dim1=-2, dim2=-1
            ).permute(0, 2, 1)
            cov_y_e[:, ind: ind + self.S, :wind_idx] = torch.diagonal(
                covs[-1][1][:, :, :S_local, :S_local], dim1=-2, dim2=-1
            ).permute(0, 2, 1)
            dynamic_e[:, ind: ind + self.S, :wind_idx] = repeat(
                dynamic, "b n -> b s n", s=S_local
            )

            track_mask_[:, : ind + self.S, :wind_idx] = 0.0
            ind = ind + self.S // 2
            prev_wind_idx = wind_idx

        traj_e = traj_e[:, :, inv_sort_inds]
        vis_e = vis_e[:, :, inv_sort_inds]
        cov_x_e = cov_x_e[:, :, inv_sort_inds]
        cov_y_e = cov_y_e[:, :, inv_sort_inds]
        vis_e = torch.sigmoid(vis_e)

        dynamic_e = dynamic_e[:, :, inv_sort_inds]
        dynamic_e = torch.sigmoid(dynamic_e)

        train_data = (
            (
                vis_predictions,
                coord_predictions,
                dynamic_predictions,
                cov_predictions,
                wind_inds,
                sort_inds,
            )
            if is_train
            else None
        )

        return traj_e, feat_init, vis_e, (cov_x_e, cov_y_e), dynamic_e, train_data



