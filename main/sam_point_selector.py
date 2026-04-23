import numpy as np
import torch
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SAMPointSelector:
    """
    Run SAM on a single RGB frame and convert masks -> sparse 2D points.

    Strategy:
    - generate masks
    - keep strongest / largest masks
    - sample:
        * centroid
        * a few high-gradient pixels inside each mask
    - enforce minimum distance between selected points
    - return exactly num_points points in (x, y)
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: str = "weights/sam_vit_b_01ec64.pth",
        device: str = "cuda",
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
        max_masks: int = 24,
        points_per_mask: int = 6,
        min_point_distance: int = 12,
    ):
        self.device = device
        self.max_masks = max_masks
        self.points_per_mask = points_per_mask
        self.min_point_distance = min_point_distance

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        sam.eval()

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )

    @torch.no_grad()
    def select_points(self, image_torch: torch.Tensor, num_points: int):
        """
        image_torch: (3, H, W), expected uint8 or float in [0,255]
        returns:
            coords: torch.FloatTensor of shape (1, num_points, 2) in (x, y)
        """
        if image_torch.dtype != torch.uint8:
            image_np = image_torch.detach().permute(1, 2, 0).cpu().numpy()
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        else:
            image_np = image_torch.permute(1, 2, 0).cpu().numpy()

        H, W = image_np.shape[:2]

        # grayscale gradient for scoring candidate pixels
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)

        masks = self.mask_generator.generate(image_np)

        # sort masks by score first, then area
        masks = sorted(
            masks,
            key=lambda m: (
                float(m.get("predicted_iou", 0.0)),
                float(m.get("stability_score", 0.0)),
                int(m.get("area", 0)),
            ),
            reverse=True,
        )[: self.max_masks]

        candidates = []

        for m in masks:
            seg = m["segmentation"].astype(np.uint8)
            area = int(m.get("area", seg.sum()))
            if area < 50:
                continue

            ys, xs = np.where(seg > 0)
            if len(xs) == 0:
                continue

            # centroid
            cx = float(xs.mean())
            cy = float(ys.mean())
            candidates.append((cx, cy, 1e6))  # force centroid to rank high

            # top gradient pixels inside mask
            scores = grad[ys, xs]
            if len(scores) > 0:
                k = min(self.points_per_mask * 4, len(scores))
                top_idx = np.argpartition(scores, -k)[-k:]
                top_points = list(zip(xs[top_idx], ys[top_idx], scores[top_idx]))

                # keep a few spaced-out points per mask
                chosen_local = []
                for x, y, s in sorted(top_points, key=lambda t: t[2], reverse=True):
                    good = True
                    for px, py, _ in chosen_local:
                        if (x - px) ** 2 + (y - py) ** 2 < (self.min_point_distance ** 2):
                            good = False
                            break
                    if good:
                        chosen_local.append((float(x), float(y), float(s)))
                    if len(chosen_local) >= self.points_per_mask:
                        break

                candidates.extend(chosen_local)

        # fallback if SAM gives too little
        if len(candidates) < num_points:
            flat_idx = np.argpartition(grad.reshape(-1), -max(num_points * 4, 200))[
                -max(num_points * 4, 200):
            ]
            ys, xs = np.unravel_index(flat_idx, grad.shape)
            for x, y in zip(xs, ys):
                candidates.append((float(x), float(y), float(grad[y, x])))

        # global non-max suppression on point positions
        selected = []
        for x, y, s in sorted(candidates, key=lambda t: t[2], reverse=True):
            if x < 1 or x >= W - 1 or y < 1 or y >= H - 1:
                continue
            ok = True
            for px, py, _ in selected:
                if (x - px) ** 2 + (y - py) ** 2 < (self.min_point_distance ** 2):
                    ok = False
                    break
            if ok:
                selected.append((x, y, s))
            if len(selected) >= num_points:
                break

        # final fallback: random fill
        rng = np.random.default_rng(0)
        while len(selected) < num_points:
            x = rng.integers(1, W - 1)
            y = rng.integers(1, H - 1)
            selected.append((float(x), float(y), 0.0))

        coords = np.array([[x, y] for x, y, _ in selected[:num_points]], dtype=np.float32)
        coords = torch.from_numpy(coords).unsqueeze(0).to(self.device)  # (1, M, 2)
        return coords