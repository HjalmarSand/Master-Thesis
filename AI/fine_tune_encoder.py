import contextlib
import gc
import os
import random
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from PIL import Image
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.utils import make_grid
from tqdm import tqdm

current_script_dir = Path(__file__).resolve().parent
parent_dir = current_script_dir.parent
sys.path.append(str(parent_dir))

import constants

# commit


@dataclass(frozen=True)
class LossSpec:
    name: str  # "infonce" | "triplet" | "smoothap"
    fn: Callable[..., torch.Tensor]
    needs_pairs: bool  # whether we need flips/iou/z_all etc


def _norm_loss_type(x: str) -> str:
    return x.lower().replace("-", "").replace("_", "")


def make_loss_spec(
    loss_type: str, *, tau=0.07, margin=0.2, gamma=0.0, r_exclude_m=50.0
) -> LossSpec:
    lt = _norm_loss_type(loss_type)
    if lt == "infonce":

        def _loss(
            z_all,
            labels_all,
            c1_all,
            c2_all,
            coord_space,
            iou_mat,
            orig_count,
            force_neg_mask,
        ):
            return infoNCE_masked_weighted(
                z_all,
                labels_all,
                c1_all,
                c2_all,
                coord_space=coord_space,
                tau=tau,
                radius_exclude=r_exclude_m,
                iou_mat=iou_mat,
                gamma=gamma,
                orig_count=orig_count,
                force_neg_mask=force_neg_mask,
            )

        return LossSpec(name="infonce", fn=_loss, needs_pairs=True)

    if lt == "triplet":

        def _loss(
            z_all,
            labels_all,
            c1_all,
            c2_all,
            coord_space,
            iou_mat,
            orig_count,
            force_neg_mask,
        ):
            return triplet_semihard_loss_cosine_coords_forced(
                z_all,
                labels_all,
                c1_all,
                c2_all,
                coord_space=coord_space,
                margin=margin,
                r_exclude_m=r_exclude_m,
                force_neg_mask=force_neg_mask,
                iou_mat=iou_mat,
                gamma=gamma,
            )

        return LossSpec(name="triplet", fn=_loss, needs_pairs=True)

    if lt == "smoothap":

        def _loss(z, labels, *_, **__):  # ignores pair stuff
            return smoothap_loss(z, labels, tau=tau)

        return LossSpec(name="smoothap", fn=_loss, needs_pairs=False)

    raise ValueError(f"Unknown loss_type={loss_type}. Use infonce|triplet|smoothap.")


@contextlib.contextmanager
def _rng_seed_scope(seed: int):
    """Temporarily set Python/Numpy/Torch RNGs, then restore."""
    import random as _random

    py_state = _random.getstate()
    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    try:
        s = int(seed) & 0xFFFFFFFF
        _random.seed(s)
        torch.manual_seed(s)
        np.random.seed(s)
        yield
    finally:
        _random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)


def _pattern_train_val_by_count(
    ids: list[str],
    order_map: dict[str, int],
    train_len: int = 200,  # number of TRAIN ids in a cycle
    val_len: int = 20,  # number of VAL ids right after each train chunk
    purge_f: int = 10,  # temporal buffer (frames) to drop from TRAIN near boundaries
):
    """
    Deterministic pattern split:
        [train_len TRAIN] -> [val_len VAL] -> repeat
    Then train-only purge: drop TRAIN ids within ±purge_f frames of any
    TRAIN|VAL or VAL|TRAIN boundary (VAL is kept intact).

    Returns (train_clean_ids, val_ids, stats)
    """
    # 1) sort usable ids by frame index
    ids_sorted = sorted(ids, key=lambda k: (order_map.get(k) is None, order_map.get(k)))

    # 2) assign ids by a fixed pattern: 200T -> 20V -> 200T -> 20V -> ...
    train_ids = []
    val_ids = []
    segments = []  # for debugging: list of tuples (kind, start_idx, end_idx_exclusive, ids_subset)
    i = 0
    n = len(ids_sorted)
    while i < n:
        # TRAIN chunk
        t_end = min(i + train_len, n)
        if i < t_end:
            seg_ids = ids_sorted[i:t_end]
            train_ids.extend(seg_ids)
            segments.append(("train", i, t_end, seg_ids))
        i = t_end

        # VAL chunk
        v_end = min(i + val_len, n)
        if i < v_end:
            seg_ids = ids_sorted[i:v_end]
            val_ids.extend(seg_ids)
            segments.append(("val", i, v_end, seg_ids))
        i = v_end

    train_ids = set(train_ids)
    val_ids = set(val_ids)

    # 3) compute boundaries between adjacent train/val segments
    def idx_of(k):
        return order_map.get(k, -(10**9))

    boundaries = []  # list of (L, R) using frame indices, for every TRAIN|VAL and VAL|TRAIN switch
    for s in range(len(segments) - 1):
        kind_a, _, _, ids_a = segments[s]
        kind_b, _, _, ids_b = segments[s + 1]
        if not ids_a or not ids_b:
            continue
        # Only boundaries where the kinds differ
        if kind_a != kind_b:
            L = max(idx_of(x) for x in ids_a)  # last frame of segment A
            R = min(idx_of(x) for x in ids_b)  # first frame of segment B
            boundaries.append((L, R))

    # 4) train-only purge around ALL boundaries
    def near_boundary_train(k):
        ik = idx_of(k)
        for L, R in boundaries:
            # inclusive: <= purge_f gives you a full (purge_f + 1) worst-case window
            if abs(ik - L) <= purge_f or abs(ik - R) <= purge_f:
                return True
        return False

    drop_train = {k for k in train_ids if near_boundary_train(k)}
    train_clean = train_ids - drop_train
    val_clean = val_ids  # keep all val; no drop from val

    # 5) stats for verification
    stats = dict(
        total=len(ids_sorted),
        train_len=train_len,
        val_len=val_len,
        purge_f=purge_f,
        val_segments=sum(1 for kind, *_ in segments if kind == "val"),
        boundaries=len(boundaries),
        purged_train=len(drop_train),
        kept_train=len(train_clean),
        kept_val=len(val_clean),
        # (Optional) first few boundaries for quick eyeballing
        boundary_samples=boundaries[: min(5, len(boundaries))],
    )
    return train_clean, val_clean, stats


def _extract_frame_index_map(csv_path: Path) -> dict[str, int | None]:
    """
    Build stem->frame_index map from CSV.
    Prefer column 'n' (or 'index'), else parse digits from filename stem,
    else fall back to row order.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    # pick filename column
    fname_col = None
    for k in ["filename", "file", "image", "image_name"]:
        if k in cols:
            fname_col = cols[k]
            break
    if fname_col is None:
        raise KeyError("No filename column found for frame ordering")

    # option 1: explicit index column
    n_col = cols.get("n") or cols.get("index")
    idx_map: dict[str, int | None] = {}

    if n_col is not None:
        for _, row in df.iterrows():
            stem = Path(str(row[fname_col])).stem
            try:
                idx_map[stem] = int(row[n_col])
            except Exception:
                idx_map[stem] = None  # will fill later
    else:
        # option 2: parse digits from filename stem
        for _, row in df.iterrows():
            stem = Path(str(row[fname_col])).stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            idx_map[stem] = int(digits) if digits else None

    # fill any None with row order to keep determinism
    order_ctr = 0
    for _, row in df.iterrows():
        stem = Path(str(row[fname_col])).stem
        if idx_map[stem] is None:
            idx_map[stem] = order_ctr
        order_ctr += 1

    return idx_map


class SkylineIdentityMultiDataset(torch.utils.data.Dataset):
    """
    Loads multiple dataset dirs; labels are global-unique across datasets.
    Exposes .labels and .coord_space.
    """

    def __init__(
        self,
        dataset_dirs: list[Path],
        mode: str,
        transform=None,
        val_split: float = 0.1,
        random_state: int = 42,
        min_iou: float = 0.97,  # Keep only data where the iou with dem and real aligned was above 0.97
        max_yaw_dev_deg: float = 15.0,  # keep only samples within ± this of dominant yaw span
        *,
        force_val_datasets: "Sequence[Union[str, Path]] | None" = None,
    ):
        super().__init__()
        assert mode in ("train", "val")
        self.transform = transform

        import pandas as pd

        def circ_abs_diff_deg(a, b):
            """minimum absolute angular difference in degrees, scalar or numpy arrays"""
            d = np.abs((a - b + 180.0) % 360.0 - 180.0)
            return d

        def find_dominant_yaw_center(
            angles_deg: np.ndarray, half_width_deg: float = 15.0
        ) -> float:
            """
            Find the center (0..360) of the densest ±half_width_deg span.
            Robust to wrap-around and works with real-valued angles.
            """
            if angles_deg.size == 0:
                return 0.0
            # Sample candidate centers each degree for simplicity
            centers = np.arange(360.0)
            # For each center, count how many angles fall within the window
            counts = np.array(
                [
                    np.sum(circ_abs_diff_deg(angles_deg, c) <= half_width_deg)
                    for c in centers
                ]
            )
            best_idx = int(np.argmax(counts))
            return float(centers[best_idx])

        per_ds_ids = []
        per_ds_id_to_paths = []
        per_ds_id_to_coord = []
        coord_spaces = []

        force_val_names = set()
        if force_val_datasets:
            for x in force_val_datasets:
                force_val_names.add(Path(x).name)

        for ds_dir in dataset_dirs:
            real_dir = ds_dir / "binary_panorama_aligned"
            dem_dir = ds_dir / "dem_skylines_paired"
            gps_csv = ds_dir / "gps_interpolated_with_orientation.csv"
            if not (real_dir.is_dir() and dem_dir.is_dir() and gps_csv.is_file()):
                raise FileNotFoundError(f"Required data not found in {ds_dir}")

            # read yaw_offsets.csv and build whitelist by IoU AND yaw proximity
            iou_yaw_whitelist: set[str] | None = None
            yaw_csv = ds_dir / "yaw_offsets.csv"
            try:
                if yaw_csv.exists() and yaw_csv.stat().st_size > 0:
                    yaw_df = pd.read_csv(yaw_csv)
                    cols_lower = {c.lower(): c for c in yaw_df.columns}

                    # flexible column picking
                    filename_col = cols_lower.get("filename") or cols_lower.get("file")
                    iou_col = (
                        cols_lower.get("iou")
                        or cols_lower.get("best_iou")
                        or cols_lower.get("post_alignment_iou")
                    )
                    yaw_col = (
                        cols_lower.get("yaw_offset_deg")
                        or cols_lower.get("yaw_offset")
                        or cols_lower.get("yaw_deg")
                        or cols_lower.get("yaw")
                        or cols_lower.get("offset_deg")
                    )

                    if filename_col is None or iou_col is None or yaw_col is None:
                        print(
                            f"[{ds_dir.name}] yaw_offsets.csv missing required columns; not filtering by yaw/IoU."
                        )
                    else:
                        tmp = yaw_df[[filename_col, iou_col, yaw_col]].dropna()
                        # stems without extension
                        stems = (
                            tmp[filename_col].astype(str).apply(lambda s: Path(s).stem)
                        )
                        ious = tmp[iou_col].astype(float).to_numpy()
                        yaws = tmp[yaw_col].astype(float).to_numpy() % 360.0

                        # 1) IoU gate
                        iou_mask = ious >= float(min_iou)

                        # 2) Find dominant yaw center on the IoU-passed subset (more robust)
                        yaws_for_mode = yaws[iou_mask]
                        if yaws_for_mode.size > 0:
                            center = find_dominant_yaw_center(
                                yaws_for_mode, half_width_deg=max_yaw_dev_deg
                            )
                            yaw_close_mask = (
                                circ_abs_diff_deg(yaws, center) <= max_yaw_dev_deg
                            )
                        else:
                            # If nothing passes IoU, we’ll end up with empty whitelist; still compute safely
                            center = 0.0
                            yaw_close_mask = np.zeros_like(yaws, dtype=bool)

                        final_mask = iou_mask & yaw_close_mask
                        iou_yaw_whitelist = set(stems[final_mask].astype(str))

                        kept = int(final_mask.sum())
                        print(
                            f"[{ds_dir.name}] IoU+Yaw filter: center≈{center:.1f}°, "
                            f"keeping {kept} with IoU≥{min_iou:.2f} and |Δyaw|≤{max_yaw_dev_deg:.1f}°"
                        )
                else:
                    print(
                        f"[{ds_dir.name}] No yaw_offsets.csv; falling back to IoU-only (or none)."
                    )
            except Exception as e:
                print(
                    f"[{ds_dir.name}] Failed to read yaw_offsets.csv ({e}); not filtering by yaw/IoU."
                )

            # GPS & coord space detection
            gps_df = pd.read_csv(gps_csv)
            cols_lower = {c.lower(): c for c in gps_df.columns}

            def pick_col(cands):
                for key in cands:
                    if key in cols_lower:
                        return cols_lower[key]
                return None

            fname_col = pick_col(["filename", "file", "image", "image_name"])
            if fname_col is None:
                raise KeyError(
                    f"Could not find filename column among {list(gps_df.columns)} in {ds_dir}"
                )

            lat_col = pick_col(["lat", "latitude", "gps_lat"])
            lon_col = pick_col(["lon", "lng", "longitude", "gps_lon"])
            if lat_col is not None and lon_col is not None:
                coord_space = "geo"
                c1_col, c2_col = lat_col, lon_col
            else:
                east_col = pick_col(["easting", "x", "utm_e"])
                north_col = pick_col(["northing", "y", "utm_n"])
                if east_col is None or north_col is None:
                    raise KeyError(
                        f"Could not find GPS (lat/lon) or planar (easting/northing) columns in {ds_dir}"
                    )
                coord_space = "utm"
                c1_col, c2_col = east_col, north_col
            coord_spaces.append(coord_space)

            # build per-dataset id maps with the combined filter
            id_to_paths = {}
            id_to_coord = {}
            kept, skipped = 0, 0

            for _, row in gps_df.iterrows():
                image_number = Path(row[fname_col]).stem

                # If whitelist exists, require membership
                if (
                    iou_yaw_whitelist is not None
                    and image_number not in iou_yaw_whitelist
                ):
                    skipped += 1
                    continue

                real_path = real_dir / f"binary_panorama_{image_number}.png"
                dem_path = dem_dir / f"{image_number}.npy"
                if real_path.exists() and dem_path.exists():
                    id_to_paths[image_number] = [(real_path, "real"), (dem_path, "dem")]
                    c1 = float(row[c1_col])
                    c2 = float(row[c2_col])
                    id_to_coord[image_number] = (c1, c2)
                    kept += 1

            if iou_yaw_whitelist is not None:
                print(f"[{ds_dir.name}] Filter skipped {skipped} pairs; kept {kept}.")

            ids = sorted(id_to_paths.keys())
            if not ids:
                raise ValueError(f"No valid pairs in {ds_dir} after yaw/IoU filtering")

            per_ds_ids.append(ids)
            per_ds_id_to_paths.append(id_to_paths)
            per_ds_id_to_coord.append(id_to_coord)

        if len(set(coord_spaces)) != 1:
            raise ValueError(
                f"Mixed coord spaces across datasets: {coord_spaces}. Select datasets with the same coordinate system."
            )
        self.coord_space = coord_spaces[0]

        # train_pairs, val_pairs = [], []

        # for ds_idx, ids in enumerate(per_ds_ids):
        #    ds_dir = dataset_dirs[ds_idx]
        #    id_to_paths = per_ds_id_to_paths[ds_idx]

        # frame order
        #    gps_csv = ds_dir / "gps_interpolated_with_orientation.csv"
        #    order_map = _extract_frame_index_map(gps_csv)

        #    usable_ids = list(ids)

        # Apply the pattern: 300 train, 30 val
        #    train_ids, val_ids, stats = _pattern_train_val_by_count(
        #        ids=usable_ids,
        #        order_map=order_map,
        #        train_len=200,
        #        val_len=20,
        #        purge_f=10,  # buffer on both sides of each boundary (train-only)
        #    )

        # print(f"Train IDs: {train_ids} \n Val IDs: {val_ids}")

        #    print(
        #        f"[{ds_dir.name}] usable={stats['total']}, val_segments={stats['val_segments']}, "
        #        f"boundaries={stats['boundaries']}, purged(train={stats['purged_train']}), "
        #        f"kept(train={stats['kept_train']}, val={stats['kept_val']}) "
        #        f"| boundary_samples={stats['boundary_samples']}"
        #    )

        #    key_fn = lambda k: order_map.get(k, 0)
        #    if mode == "train":
        #        train_pairs += [
        #            (ds_idx, img_id) for img_id in sorted(train_ids, key=key_fn)
        #        ]
        #    else:
        #        val_pairs += [
        #            (ds_idx, img_id) for img_id in sorted(val_ids, key=key_fn)
        #        ]

        train_pairs, val_pairs = [], []

        for ds_idx, ids in enumerate(per_ds_ids):
            ds_dir = dataset_dirs[ds_idx]
            id_to_paths = per_ds_id_to_paths[ds_idx]

            # deterministically sort by frame index for stable ordering
            gps_csv = ds_dir / "gps_interpolated_with_orientation.csv"
            order_map = _extract_frame_index_map(gps_csv)
            key_fn = lambda k: order_map.get(k, 0)
            usable_ids = list(ids)

            if ds_dir.name in force_val_names:
                # Entire dataset goes to VAL only
                if mode == "val":
                    val_pairs += [
                        (ds_idx, img_id) for img_id in sorted(usable_ids, key=key_fn)
                    ]
                    print(
                        f"[{ds_dir.name}] Using FULL dataset for VAL only; excluded from TRAIN."
                    )
                # if mode == "train": skip entirely
            else:
                # All other datasets go to TRAIN only
                if mode == "train":
                    train_pairs += [
                        (ds_idx, img_id) for img_id in sorted(usable_ids, key=key_fn)
                    ]
                    print(
                        f"[{ds_dir.name}] Using FULL dataset for TRAIN only; excluded from VAL."
                    )
                # if mode == "val": skip entirely

        chosen_pairs = train_pairs if mode == "train" else val_pairs
        unique_keys = sorted(set(chosen_pairs))
        self.key2label = {key: i for i, key in enumerate(unique_keys)}

        self.items = []
        for ds_idx, img_id in chosen_pairs:
            label = self.key2label[(ds_idx, img_id)]
            id_to_paths = per_ds_id_to_paths[ds_idx]
            id_to_coord = per_ds_id_to_coord[ds_idx]
            c1, c2 = id_to_coord[img_id]
            for p, kind in id_to_paths[img_id]:
                self.items.append((p, kind == "dem", label, c1, c2))

        self.labels = [label for (_, _, label, _, _) in self.items]
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: Path, is_dem: bool):
        if is_dem:
            arr = np.load(path)
            if arr.ndim == 2:
                mask = arr.astype(np.uint8)
            else:
                mask = dem_vector_to_mask(
                    elevations=arr,
                    output_width=getattr(constants, "PANORAMA_IMG_WIDTH", 5120),
                    output_height=getattr(constants, "PANORAMA_IMG_HEIGHT", 960),
                    v_fov_deg=73.74,
                    median_kernel=19,
                )
            return Image.fromarray(mask)
        return Image.open(path).convert("L")

    def __getitem__(self, idx):
        path, is_dem, label, c1, c2 = self.items[idx]
        img = self._load_image(path, is_dem)
        if self.transform is not None:
            seed = (int(label) * 73856093 + int(self._epoch) * 19349663) % (
                2**32 - 1
            ) or 1
            with _rng_seed_scope(seed):
                out = self.transform(img)
        else:
            # fallback if no transform
            t = transforms.ToTensor()(img)
            prof = skyline_profile_from_mask(t, 0.5)
            out = (t, prof)

        img_t, prof = out
        return (
            img_t,
            prof,  # NEW
            torch.tensor(label, dtype=torch.long),
            torch.tensor(c1, dtype=torch.float32),
            torch.tensor(c2, dtype=torch.float32),
        )


class PKSampler(Sampler):
    """Samples batches with P labels and K samples per label (batch size P*K)."""

    def __init__(self, labels, P, K, shuffle=True, drop_last=True):
        self.labels = list(labels)
        self.P = P
        self.K = K
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.label_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.label_to_indices[y].append(idx)

        self.labels_unique = list(self.label_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            for ys in self.label_to_indices.values():
                random.shuffle(ys)
            random.shuffle(self.labels_unique)

        ptr = {y: 0 for y in self.labels_unique}
        batch = []
        while True:
            candidates = [
                y
                for y in self.labels_unique
                if ptr[y] + self.K <= len(self.label_to_indices[y])
            ]
            if len(candidates) < self.P:
                break
            selected = random.sample(candidates, self.P)
            for y in selected:
                idxs = self.label_to_indices[y][ptr[y] : ptr[y] + self.K]
                batch.extend(idxs)
                ptr[y] += self.K
            yield batch
            batch = []

        if not self.drop_last and len(batch) > 0:
            yield batch

    def __len__(self):
        counts = [len(v) for v in self.label_to_indices.values()]
        return sum(c // self.K for c in counts) // self.P


class RandomMorphology:
    def __init__(self, p=0.5, max_kernel=5):
        self.p = p
        self.max_kernel = max(1, max_kernel)

    def _rand_odd(self, lo, hi):
        lo = max(1, lo + (lo % 2 == 0))  # make odd
        hi = hi - (hi % 2 == 0)
        if lo > hi:
            return 1
        n = ((hi - lo) // 2) + 1
        return lo + 2 * random.randrange(n)

    def __call__(self, x):
        if random.random() > self.p:
            return x
        C, H, W = x.shape
        k = self._rand_odd(1, self.max_kernel)
        if k == 1:
            return x
        pad = k // 2
        weight = torch.ones((C, 1, k, k), device=x.device, dtype=x.dtype)
        y = F.conv2d(x.unsqueeze(0), weight, padding=pad, groups=C)
        # random erode/dilate
        if random.random() < 0.5:
            y = (y > 0).to(x.dtype)  # dilate
        else:
            y = (y >= k * k).to(x.dtype)  # erode
        return y.squeeze(0)


class HorizontalDownsample:
    """Downsample width by pooling on tensors (C,H,W)."""

    def __init__(self, factor: int = 2, mode: str = "max"):
        assert factor >= 1
        assert mode in ("avg", "max")
        self.factor = factor
        self.mode = mode

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return x
        C, H, W = x.shape
        k = self.factor
        x_ = x.unsqueeze(0)
        if self.mode == "avg":
            y = F.avg_pool2d(x_, kernel_size=(1, k), stride=(1, k), ceil_mode=True)
        else:
            y = F.max_pool2d(x_, kernel_size=(1, k), stride=(1, k), ceil_mode=True)
        return y.squeeze(0)


class ResizeByHeightKeepAR:
    """Resize to target height while preserving aspect ratio."""

    def __init__(self, target_h: int):
        assert target_h > 0
        self.target_h = target_h

    def __call__(self, img):
        w, h = img.size
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image size {w}x{h}")
        scale = self.target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        # keep NEAREST to avoid gray levels in binary masks
        return img.resize((new_w, self.target_h), Image.NEAREST)


class CenterCropHeightPx:
    """Crop a fixed pixel height from the vertical center; keep full width."""

    def __init__(self, crop_h: int):
        assert crop_h > 0
        self.crop_h = crop_h

    def __call__(self, img):
        w, h = img.size
        ch = min(self.crop_h, h)
        top = max(0, (h - ch) // 2)
        return img.crop((0, top, w, top + ch))


class ToThreeChannels:
    def __call__(self, x):
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


def _denorm(x3):
    # x3: (3,H,W) normalized with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=x3.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x3.device).view(3, 1, 1)
    return (x3 * std + mean).clamp(0, 1)


def save_tensor_as_png(t, path):
    # t: (1,H,W) or (3,H,W) in [0,1]
    if t.dim() == 3 and t.size(0) == 1:
        arr = (t.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
    elif t.dim() == 3 and t.size(0) == 3:
        arr = (_denorm(t).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
    else:
        raise ValueError("Expect (1,H,W) or (3,H,W)")


def plot_profile(prof, path):
    plt.figure()
    plt.plot(prof.cpu().numpy())
    plt.title("Skyline profile (normalized height)")
    plt.xlabel("x (downsampled width)")
    plt.ylabel("h_norm")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def visualize_transform_pipeline(img_path, out_dir="debug_vis", train=True):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(img_path).convert("L")

    if train:
        T = make_train_transform()
    else:
        T = make_val_transform()

    # Re-run the inner steps manually to see each stage
    crop = CenterCropHeightPx(400)
    t1 = crop(img)

    # we’ll mimic the same randomness as in TrainBinaryTransform but fixed here
    roll_f = 0.0  # set non-zero to test yaw jitter
    t2 = (
        TrainBinaryTransform._circular_roll_pil(T, t1, roll_f)
        if hasattr(T, "_circular_roll_pil")
        else t1
    )

    ty_px = 0  # set non-zero to test vertical jitter
    t3 = vertical_translate_semantic(t2, ty_px)

    to_tensor = transforms.ToTensor()
    t = to_tensor(t3)  # (1,H,W) in [0,1]
    t_down = HorizontalDownsample(factor=2, mode="max")(t)

    prof = skyline_profile_from_mask(t_down.clone(), 0.5)  # (W',)

    # optional morph like in TrainBinaryTransform
    if train:
        t_m = RandomMorphology(p=0.0, max_kernel=5)(t_down)  # set p=0 for clarity first
    else:
        t_m = t_down

    t_3 = ToThreeChannels()(t_m)
    t_n = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        t_3
    )

    # Save previews
    t_orig = transforms.ToTensor()(t1)
    save_tensor_as_png(t_orig, f"{out_dir}/01_crop.png")
    save_tensor_as_png(transforms.ToTensor()(t2), f"{out_dir}/02_roll.png")
    save_tensor_as_png(transforms.ToTensor()(t3), f"{out_dir}/03_vtranslate.png")

    save_tensor_as_png(t_down, f"{out_dir}/04_downsample_1ch.png")
    plot_profile(prof, f"{out_dir}/05_profile.png")

    save_tensor_as_png(t_n, f"{out_dir}/06_final_3ch_normed.png")

    # flipped variant
    t_flip = torch.flip(t_n, dims=[-1])
    save_tensor_as_png(t_flip, f"{out_dir}/07_final_flip_3ch_normed.png")

    print(f"Wrote previews to: {out_dir}")


def visualize_batch(train_loader, out_dir="debug_batch", num=8):
    os.makedirs(out_dir, exist_ok=True)
    imgs, profs, labels, c1, c2 = next(iter(train_loader))  # (B,3,H,W), (B,W)
    B = min(num, imgs.size(0))
    # denorm for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    ims = (imgs[:B] * std + mean).clamp(0, 1)

    grid = make_grid(ims, nrow=min(4, B))
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(f"{out_dir}/images_grid.png")

    # plot a few profiles
    plt.figure()
    for i in range(min(6, B)):
        plt.plot(profs[i].cpu().numpy(), label=f"label={int(labels[i])}")
    plt.legend()
    plt.title("Batch skyline profiles")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/profiles.png")
    plt.close()

    print(f"Wrote: {out_dir}/images_grid.png and {out_dir}/profiles.png")


def despike_columns(
    rows: np.ndarray, spike_abs_px: int = 20, neighbor_agree_tol_px: int = 5
) -> np.ndarray:
    """
    rows: int array shape (W,), 'first ground row' per column (0=top).
    Replace 1-px spikes where the middle column differs from both neighbors
    by > spike_abs_px, while neighbors are close to each other (bridge).
    """
    r = rows.astype(np.int32, copy=True)
    if r.size < 3:
        return r

    left = r[:-2]
    mid = r[1:-1]
    right = r[2:]

    # neighbors agree?
    neighbors_close = np.abs(left - right) <= neighbor_agree_tol_px
    # middle far from BOTH neighbors?
    mid_far_left = np.abs(mid - left) > spike_abs_px
    mid_far_right = np.abs(mid - right) > spike_abs_px
    spike_mask_center = neighbors_close & mid_far_left & mid_far_right

    # replacement = average of neighbors (or their median)
    repl = (left + right) // 2

    r[1:-1][spike_mask_center] = repl[spike_mask_center]
    return r


def dem_vector_to_mask(
    elevations: np.ndarray,
    output_width: int,
    output_height: int,
    v_fov_deg: float = 90.0,
    median_kernel: int = 19,
) -> np.ndarray:
    e = np.asarray(elevations).squeeze()
    if e.ndim != 1:
        raise ValueError(f"DEM vector must be 1D after squeeze, got shape {e.shape}")

    num_bins = len(e)
    angles = np.linspace(0, 360, num_bins, endpoint=False)

    order = np.argsort(angles)
    a_sorted = angles[order]
    e_sorted = e[order]

    a_wrap = np.concatenate(([a_sorted[-1] - 360], a_sorted, [a_sorted[0] + 360]))
    e_wrap = np.concatenate(([e_sorted[-1]], e_sorted, [e_sorted[0]]))

    out_az = np.linspace(0, 360, output_width, endpoint=False)
    e_interp = np.interp(out_az, a_wrap, e_wrap, period=360)

    if median_kernel and median_kernel > 1 and median_kernel % 2 == 1:
        from scipy.signal import medfilt

        e_interp = medfilt(e_interp, kernel_size=median_kernel)

    e_interp = np.roll(e_interp, -output_width // 2)

    top = v_fov_deg / 2.0
    rows = np.clip(
        np.round(((top - e_interp) / v_fov_deg) * output_height), 0, output_height
    ).astype(int)

    # remove one-pixel needles
    rows = despike_columns(rows, spike_abs_px=20, neighbor_agree_tol_px=5)

    mask = np.zeros((output_height, output_width), dtype=np.uint8)
    for x, y in enumerate(rows):
        mask[0:y, x] = 255
    return mask


def haversine_matrix(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    B = lat.shape[0]
    lat1 = lat.view(B, 1)
    lat2 = lat.view(1, B)
    lon1 = lon.view(B, 1)
    lon2 = lon.view(1, B)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1 - a, min=1e-12)))
    R = 6371000.0
    return R * c


def smoothap_loss(emb, labels, tau=0.01):
    """
    SmoothAP surrogate (mini-batch version).
    emb: (B,D) L2-normalized embeddings
    labels: (B,) int labels
    tau: temperature for pairwise sigmoid (smaller = sharper ranking)
    Returns: scalar loss = 1 - mean(AP_i)
    """
    # cosine similarities
    S = emb @ emb.t()  # (B,B)
    B = S.size(0)
    eye = torch.eye(B, device=S.device, dtype=torch.bool)
    S = S.masked_fill(eye, float("-inf"))  # exclude self

    # relevance matrix
    rel = labels.view(-1, 1).eq(labels.view(1, -1)) & ~eye  # (B,B) positives per anchor

    # pairwise "softer than" comparisons for each anchor i:
    # P_ij = P(score_j < score_k) aggregated differentiably
    # For anchor i, compare all j vs all k with a sigmoid on (S[i,k] - S[i,j])
    # We want positives ranked above negatives.
    pos_mask = rel
    neg_mask = ~rel & ~eye

    # Avoid empty-positive anchors
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not valid.any():
        return torch.tensor(0.0, device=emb.device, requires_grad=True)

    APs = []
    for i in torch.where(valid)[0]:
        s_i = S[i]  # (B,)
        pos_scores = s_i[pos_mask[i]]  # (#pos,)
        neg_scores = s_i[neg_mask[i]]  # (#neg,)

        # pairwise probability that a positive outranks a negative
        # shape: (#pos, #neg)
        P = torch.sigmoid((pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)) / tau)

        # expected precision at the ranks of positives
        # approx precision for each positive = mean over negatives it beats
        prec_pos = P.mean(dim=1)  # (#pos,)

        # AP for anchor i = mean precision over its positives
        AP = prec_pos.mean()
        APs.append(AP)

    mAP_batch = torch.stack(APs).mean()
    return 1.0 - mAP_batch


def infoNCE_masked_weighted(
    z,
    labels,
    c1,
    c2,
    coord_space,
    tau=0.07,
    radius_exclude=50.0,
    iou_mat: torch.Tensor | None = None,
    gamma: float = 0.0,
    orig_count: int | None = None,
    force_neg_mask: torch.Tensor | None = None,
):
    """
    z: [N,D] L2-normalized embeddings (could be your z_all with flips included).
    labels: [N] (flips must have different labels -> negatives)
    c1,c2: [N] coords
    """
    z = F.normalize(z, dim=1)
    logits = (z @ z.T) / tau  # [N,N]
    N = logits.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    logits.masked_fill_(eye, float("-inf"))  # remove self

    # positives = same label (K=2 ensures exactly one positive per anchor)
    posmask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~eye

    # If orig_count is given, disable ANY positive pairs that involve the flips
    # (anchors in [orig_count:]) and also positives between flips themselves.
    if orig_count is not None:
        N = posmask.size(0)
        flips = torch.zeros(N, dtype=torch.bool, device=posmask.device)
        flips[orig_count:] = True
        # zero positives where either side is a flip
        posmask[flips, :] = False
        posmask[:, flips] = False

    # exclude near negatives (like your r_exclude_m)
    if coord_space == "geo":
        d = haversine_matrix(c1, c2)
    else:
        coords = torch.stack([c1, c2], dim=1)
        d = torch.cdist(coords, coords, p=2)
    near = d < float(radius_exclude)

    if force_neg_mask is None:
        force_neg_mask = torch.zeros_like(posmask)

    # allow negatives if they are far OR explicitly forced (e.g., flips)
    allowed = posmask | (~posmask & (~near | force_neg_mask))

    logits = logits.masked_fill(~allowed, float("-inf"))

    # optional weighting by skyline IoU (bigger weight for low-IoU negatives)
    if iou_mat is not None and gamma > 0:
        # weight ONLY negatives (allowed & ~posmask)
        w = 1.0 + gamma * (1.0 - iou_mat)  # [N,N], >=1
        neg_allowed = allowed & (~posmask)
        w = torch.where(neg_allowed, w, torch.ones_like(w))
        logits = logits + torch.log(torch.clamp(w, min=1e-6))

    # numerator: log-sum-exp over positives (handles multiple positives if K>2)
    num = torch.logsumexp(
        torch.where(posmask, logits, torch.full_like(logits, float("-inf"))), dim=1
    )
    # denominator: log-sum-exp over all allowed (positives + eligible negatives)
    den = torch.logsumexp(logits, dim=1)

    valid = posmask.any(dim=1)
    loss = -(num[valid] - den[valid]).mean()
    return loss


def triplet_semihard_loss_cosine_coords_forced(
    emb,
    labels,
    c1,
    c2,
    coord_space: str,
    margin=0.2,
    r_exclude_m=50.0,
    force_neg_mask: torch.Tensor | None = None,
    iou_mat: torch.Tensor | None = None,
    gamma: float = 0.0,
):
    """
    Cosine-triplet with geo/utm near-neg exclusion + optional forced negatives.
    If iou_mat is provided (B,B), we use pairwise margin:
        m_eff[i,j] = margin + gamma*(1 - IoU[i,j])   (clamped >= 0)
    and apply it both for semi-hard mining and in the hinge.
    """
    device = emb.device
    B = emb.size(0)
    sim = emb @ emb.t()

    neg_large = sim.new_tensor(torch.finfo(sim.dtype).min / 2)
    sim.fill_diagonal_(neg_large)

    labels = labels.view(-1, 1)
    same = labels == labels.t()
    diff = ~same

    # pairwise distances for near-exclusion
    if coord_space == "geo":
        dists_m = haversine_matrix(c1.to(device), c2.to(device))
    else:
        coords = torch.stack([c1.to(device), c2.to(device)], dim=1)
        dists_m = torch.cdist(coords, coords, p=2)

    near = dists_m < r_exclude_m

    # eligible negatives
    if force_neg_mask is not None:
        diff_eff = (diff & (~near)) | force_neg_mask.to(device)
    else:
        diff_eff = diff & (~near)

    # hardest positive per anchor
    pos_sim = sim.masked_fill(~same, neg_large).max(dim=1).values  # (B,)

    # Pairwise margin matrix for mining and loss
    if iou_mat is not None:
        m_eff = torch.clamp(margin + gamma * (1 - iou_mat), min=margin)  # (B,B)
    else:
        m_eff = sim.new_full((B, B), float(margin))

    # semi-hard mining: for each i, eligible j with sim[i,j] > pos_sim[i] - m_eff[i,j]
    thresh = pos_sim.unsqueeze(1) - m_eff  # (B,B)
    semi_mask = diff_eff & (sim > thresh)

    # best semi-hard per i (largest sim)
    semi_sim = sim.masked_fill(~semi_mask, neg_large)
    semi_val, semi_idx = semi_sim.max(dim=1)  # (B,), (B,)

    # fallback: hardest eligible negative if no semi-hard
    fall_sim = sim.masked_fill(~diff_eff, neg_large)
    fall_val, fall_idx = fall_sim.max(dim=1)

    has_semi = semi_mask.any(dim=1)
    chosen_neg_sim = torch.where(has_semi, semi_val, fall_val)  # (B,)
    chosen_neg_idx = torch.where(has_semi, semi_idx, fall_idx)  # (B,)

    # distances in cosine space
    d_ap = 1 - pos_sim
    d_an = 1 - chosen_neg_sim

    # pick per-anchor margin for the chosen negative
    m_anchor = m_eff[torch.arange(B, device=device), chosen_neg_idx]  # (B,)

    # print(f"Using margin {m_anchor}")

    loss = torch.clamp(d_ap - d_an + m_anchor, min=0).mean()
    return loss


def build_pk_dataloaders(
    dataset_dir: Union[Path, Sequence[Path]],
    P: int,
    K: int,
    val_split: float = 0.1,
    random_state: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    min_iou: float = 0.95,
    max_yaw_dev_deg: float = 15.0,
    train_transform=None,
    val_transform=None,
    *,
    force_val_datasets: "Sequence[Union[str, Path]] | None" = None,
):
    if isinstance(dataset_dir, (list, tuple)):
        dataset_dirs = list(dataset_dir)
    else:
        dataset_dirs = [dataset_dir]

    train_ds = SkylineIdentityMultiDataset(
        dataset_dirs,
        mode="train",
        transform=train_transform or make_train_transform(),
        val_split=val_split,
        random_state=random_state,
        min_iou=min_iou,
        max_yaw_dev_deg=max_yaw_dev_deg,
        force_val_datasets=force_val_datasets,
    )
    val_ds = SkylineIdentityMultiDataset(
        dataset_dirs,
        mode="val",
        transform=val_transform or make_val_transform(),
        val_split=val_split,
        random_state=random_state,
        min_iou=min_iou,
        max_yaw_dev_deg=max_yaw_dev_deg,
        force_val_datasets=force_val_datasets,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=PKSampler(
            train_ds.labels, P=P, K=K, shuffle=True, drop_last=True
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=PKSampler(
            val_ds.labels, P=P, K=K, shuffle=False, drop_last=False
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    return train_ds, val_ds, train_loader, val_loader


def skyline_profile_from_mask(t: torch.Tensor, thresh: float = 0.5):
    """
    t: (1,H,W) or (C,H,W) with sky=1 (white), ground=0 (black).
    Returns h_norm in [0,1], where 1 = top, 0 = bottom.
    """
    if t.dim() == 3:
        t = t[0:1]  # use first channel
    sky = (t >= thresh)[0]  # (H,W) bool
    ground = ~sky  # (H,W) bool
    H = t.shape[1]

    # Where there is any ground in the column, take the FIRST ground row from the top.
    has_ground = ground.any(dim=0)  # (W,)
    # argmax on bool gives the index of the first True
    first_ground = ground.float().argmax(dim=0)  # (W,) but 0 if none True

    # If there is no ground at all in a column, set row index to H (fully sky).
    rows = torch.where(has_ground, first_ground, torch.tensor(H, device=t.device))

    # Normalize: top=1.0, bottom=0.0
    h_norm = 1.0 - rows.float().clamp(max=H) / H
    return h_norm


def vertical_translate_semantic(img: Image.Image, ty: int) -> Image.Image:
    if ty == 0:
        return img
    w, h = img.size
    if ty > 0:
        # shift down -> pad sky (white) on top
        out = Image.new(img.mode, (w, h), 255)
        out.paste(img, (0, ty))
    else:
        # shift up -> pad ground (black) on bottom
        out = Image.new(img.mode, (w, h), 0)
        out.paste(img, (0, ty))  # PIL handles negative offsets
    return out


class TrainBinaryTransform:
    def __init__(
        self,
        w_shrink=2,
        morph_p=0.4,
        max_roll_frac=0.02,
        max_v_translate_frac=0.03,
        scale_range=(0.98, 1.02),
    ):
        self.crop = CenterCropHeightPx(400)
        self.to_tensor = transforms.ToTensor()
        self.downsample = HorizontalDownsample(factor=w_shrink, mode="max")
        self.morph = RandomMorphology(p=morph_p, max_kernel=5)
        self.to3 = ToThreeChannels()
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.max_roll_frac = float(max_roll_frac)
        self.max_v_translate_frac = float(max_v_translate_frac)
        self.scale_range = scale_range

    def _circular_roll_pil(self, img, frac):
        # frac in [-max_roll_frac, +max_roll_frac], shift in pixels
        w, h = img.size
        if w <= 1 or frac == 0.0:
            return img

        # map arbitrary signed shift to [0, w-1]
        shift = int(round(frac * w)) % w
        if shift == 0:
            return img

        # take the rightmost 'shift' columns and move them to the left
        right = img.crop((w - shift, 0, w, h))
        left = img.crop((0, 0, w - shift, h))
        out = Image.new(img.mode, (w, h))
        out.paste(right, (0, 0))
        out.paste(left, (shift, 0))
        return out

    def __call__(self, img):
        # 1) crop
        img = self.crop(img)

        # 2) tiny yaw jitter via circular roll (±2% width)
        roll_f = random.uniform(-self.max_roll_frac, self.max_roll_frac)
        img = self._circular_roll_pil(img, roll_f)

        # 3) pitch jitter with semantic padding
        ty_px = int(
            round(self.max_v_translate_frac * img.size[1] * random.uniform(-1.0, 1.0))
        )
        img = vertical_translate_semantic(img, ty_px)

        # 4) tensor ops (binary-safe)
        t = self.to_tensor(img)  # (1,H,W) in [0,1]
        t = self.downsample(t)  # (1,H,W')

        # >>> compute skyline profile BEFORE morph/normalize <<<
        prof = skyline_profile_from_mask(t.clone(), thresh=0.5)  # (W',) float in [0,1]

        # mild morphology then 3ch + normalize for the CNN
        t = self.morph(t).clamp(0, 1)
        t = self.to3(t)  # (3,H,W')
        t = self.norm(t)

        # Return BOTH
        return t, prof


class ValBinaryTransform:
    def __init__(self, target_h=224, w_shrink=2):
        self.crop = CenterCropHeightPx(400)
        # self.resize = ResizeByHeightKeepAR(target_h)
        self.to_tensor = transforms.ToTensor()
        self.downsample = HorizontalDownsample(factor=w_shrink, mode="max")
        self.to3 = ToThreeChannels()
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, img):
        img = self.crop(img)
        t = self.to_tensor(img)  # (1,H,W)
        t = self.downsample(t)  # (1,H,W')

        # profile from the same (aug-free) view
        prof = skyline_profile_from_mask(t.clone(), thresh=0.5)  # (W',)

        t = self.to3(t)
        t = self.norm(t)
        return t, prof


def make_train_transform(w_shrink: int = 2, morph_p: float = 0.5):
    return TrainBinaryTransform(w_shrink=w_shrink, morph_p=morph_p)


def make_val_transform(target_h: int = 224, w_shrink: int = 2):
    return ValBinaryTransform(target_h=target_h, w_shrink=w_shrink)


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128, freeze_layers=12):
        super().__init__()
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for i, child in enumerate(self.backbone.features.children()):
            if i < freeze_layers:
                for p in child.parameters():
                    p.requires_grad = False
        num_features = self.backbone.last_channel
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, embedding_dim, bias=True),
        )

    def forward(self, x):
        z = self.backbone(x)
        z = nn.functional.normalize(z, p=2, dim=1)
        return z


@torch.no_grad()
def compute_val_loss(model, val_loader, coord_space, device, loss_spec: LossSpec):
    model.eval()
    losses = []
    for imgs, profs, labels, c1, c2 in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        c1 = c1.to(device, non_blocking=True)
        c2 = c2.to(device, non_blocking=True)

        z = model(imgs)

        if loss_spec.needs_pairs:
            zf = model(torch.flip(imgs, dims=[-1]))
            z_all = torch.cat([z, zf], dim=0)
            B = z.shape[0]
            labels_all = torch.cat([labels, labels + (labels.max() + 1)], dim=0)
            c1_all = torch.cat([c1, c1], dim=0)
            c2_all = torch.cat([c2, c2], dim=0)

            force_neg_mask = torch.zeros(
                (2 * B, 2 * B), dtype=torch.bool, device=device
            )
            for y in labels.unique():
                idxs = (labels == y).nonzero(as_tuple=False).squeeze(1)
                idxs_f = idxs + B
                force_neg_mask[idxs[:, None], idxs_f[None, :]] = True
                force_neg_mask[idxs_f[:, None], idxs[None, :]] = True

            profs = profs.to(device, non_blocking=True)
            profs_f = torch.flip(profs, dims=[-1])
            prof_all = torch.cat([profs, profs_f], dim=0)
            P1, P2 = prof_all.unsqueeze(1), prof_all.unsqueeze(0)
            inter = torch.minimum(P1, P2).sum(dim=2)
            union = torch.maximum(P1, P2).sum(dim=2).clamp_min(1e-8)
            iou_mat = inter / union

            loss = loss_spec.fn(
                z_all,
                labels_all,
                c1_all,
                c2_all,
                coord_space,
                iou_mat,
                B,
                force_neg_mask,
            )
        else:
            loss = loss_spec.fn(z, labels, None, None, coord_space, None, None, None)

        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def objective(trial):
    seed = 1234 + trial.number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_epochs = 60

    # search space
    loss_type = trial.suggest_categorical(
        "loss_type", ["infonce", "triplet", "smoothap"]
    )
    use_amp = trial.suggest_categorical("use_amp", [True, False])
    freeze_layers = trial.suggest_categorical("freeze_layers", [12, 17, 19])
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    lr_head = trial.suggest_float("lr_head", 5e-4, 5e-3, log=True)
    lr_bb = trial.suggest_float(
        "lr_backbone", 5e-5, 5e-4, log=True
    )  # TODO: if all layers frozen this can be neglected
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    morph_p = trial.suggest_float("morph_p", 0.0, 0.6)
    w_shrink = trial.suggest_categorical("w_shrink", [2, 4])
    P = trial.suggest_categorical(
        "P", [8, 12, 16]
    )  # TODO: Might need to be decreased for RAM
    K = 2  # keep 2. There is only one positive per batch
    use_flips = trial.suggest_categorical("use_flips_as_negs", [True, False])
    r_exclude_m = trial.suggest_categorical("r_exclude_m", [25.0, 50.0, 100.0])

    # loss-specific params
    tau = 0.07
    margin = 0.1
    gamma = 0.0
    if loss_type == "infonce":
        tau = trial.suggest_float("tau", 0.03, 0.2, log=True)
        gamma = trial.suggest_float("gamma", 0.0, 4.0)
    elif loss_type == "triplet":
        margin = trial.suggest_float("margin", 0.02, 0.3)
        gamma = trial.suggest_float("gamma", 0.0, 4.0)
    else:  # smoothap
        tau = trial.suggest_float("tau", 0.001, 0.1, log=True)

    cfg = dict(
        loss_type=loss_type,
        use_amp=use_amp,
        freeze_layers=freeze_layers,
        embedding_dim=embedding_dim,
        lr_head=lr_head,
        lr_backbone=lr_bb,
        weight_decay=weight_decay,
        morph_p=morph_p,
        w_shrink=w_shrink,
        P=P,
        K=K,
        use_flips_as_negs=use_flips,
        r_exclude_m=r_exclude_m,
        tau=tau,
        margin=margin,
        gamma=gamma,
        # num_epochs can be short here; ASHA promotes good trials
    )

    # paths/datasets
    datasets_root_dir = Path(constants.IMAGES_DIR)
    project_root = Path(__file__).resolve().parent.parent
    model_path = (
        project_root / "AI" / "models" / "optuna" / f"optuna_trial_{trial.number}.pth"
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # run (shorter epochs for HPO; ASHA will promote)
    score = train_placer_model_pk(
        dataset_dir=[
            d
            for d in datasets_root_dir.iterdir()
            if (d / "dem_skylines_paired").is_dir()
        ],
        model_save_path=model_path,
        num_epochs=num_epochs,
        config=cfg,
        optuna_trial=trial,
        force_val_datasets={"dataset1"},
    )
    return score


def run_hpo():
    storage = f"sqlite:///{Path(__file__).resolve().parent}/optuna_skyline.db"
    study = optuna.create_study(
        study_name="placer_hpo",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(multivariate=True, group=True, consider_prior=True),
        pruner=HyperbandPruner(min_resource=10, max_resource=60, reduction_factor=3),
    )
    study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=True)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


def train_placer_model_pk(
    dataset_dir: Path,
    model_save_path: Path,
    num_epochs: int = 100,
    P: int = 16,
    K: int = 2,
    use_amp=True,
    r_exclude_m=50.0,
    loss_type="infonce",
    tau=0.07,
    margin=0.1,
    gamma=0.4,
    *,
    config: dict | None = None,
    optuna_trial: "optuna.trial.Trial | None" = None,
    force_val_datasets: "Sequence[Union[str, Path]] | None" = None,  # <-- add
):
    cfg = config or {}
    loss_type = cfg.get("loss_type", loss_type)
    tau = cfg.get("tau", tau)
    margin = cfg.get("margin", margin)
    gamma = cfg.get("gamma", gamma)
    r_exclude_m = cfg.get("r_exclude_m", r_exclude_m)
    use_amp = cfg.get("use_amp", use_amp)
    P = cfg.get("P", P)
    K = cfg.get("K", K)
    freeze_layers = cfg.get("freeze_layers", 12)
    embedding_dim = cfg.get("embedding_dim", 128)
    lr_head = cfg.get("lr_head", 1e-3)
    lr_backbone = cfg.get("lr_backbone", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)
    morph_p = cfg.get("morph_p", 0.4)
    w_shrink = cfg.get("w_shrink", 2)
    use_flips_as_negs = cfg.get("use_flips_as_negs", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf = TrainBinaryTransform(w_shrink=w_shrink, morph_p=morph_p)
    val_tf = ValBinaryTransform(w_shrink=w_shrink)

    train_ds, val_ds, train_loader, val_loader = build_pk_dataloaders(
        dataset_dir,
        P=P,
        K=K,
        min_iou=0.97,
        max_yaw_dev_deg=15.0,
        train_transform=train_tf,
        val_transform=val_tf,
        force_val_datasets=force_val_datasets,
    )

    print(f"Train: {len(train_ds)} items, {len(train_ds.key2label)} unique IDs")
    print(f"Val:   {len(val_ds)} items, {len(val_ds.key2label)} unique IDs")

    coord_space = getattr(train_ds, "coord_space", "geo")
    print(f"Coordinate space detected: {coord_space}")

    loss_spec = make_loss_spec(
        loss_type, tau=tau, margin=margin, gamma=gamma, r_exclude_m=r_exclude_m
    )
    tqdm.write(f"Starting PxK training | loss={loss_spec.name} | coord={coord_space}")

    #  backbone / head with tuned knobs
    model = EmbeddingNet(embedding_dim=embedding_dim, freeze_layers=freeze_layers).to(
        device
    )

    head_params = list(model.backbone.classifier.parameters())
    backbone_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and ("classifier" not in n)
    ]
    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
        ]
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    autocast_ctx = lambda enabled: torch.amp.autocast("cuda", enabled=enabled)

    best_score = -1.0
    best_path = model_save_path.with_name(f"{model_save_path.stem}_best.pth")
    try:
        for epoch in range(num_epochs):
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            if hasattr(val_loader.dataset, "set_epoch"):
                val_loader.dataset.set_epoch(epoch)

            model.train()
            losses = []

            for imgs, profs, labels, c1, c2 in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                leave=False,
            ):
                imgs = imgs.to(device, non_blocking=True)
                profs = profs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                c1 = c1.to(device, non_blocking=True)
                c2 = c2.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(scaler.is_enabled()):
                    z = model(imgs)

                    if loss_spec.needs_pairs and use_flips_as_negs:
                        with torch.no_grad():
                            zf = model(torch.flip(imgs, dims=[-1]))
                        profs_f = torch.flip(profs, dims=[-1])
                        prof_all = torch.cat([profs, profs_f], dim=0)
                        P1, P2 = prof_all.unsqueeze(1), prof_all.unsqueeze(0)
                        inter = torch.minimum(P1, P2).sum(dim=2)
                        union = torch.maximum(P1, P2).sum(dim=2).clamp_min(1e-8)
                        iou_mat = inter / union

                        z_all = torch.cat([z, zf], dim=0)
                        B = z.shape[0]
                        labels_all = torch.cat(
                            [labels, labels + (labels.max() + 1)], dim=0
                        )
                        c1_all = torch.cat([c1, c1], dim=0)
                        c2_all = torch.cat([c2, c2], dim=0)

                        force_neg_mask = torch.zeros(
                            (2 * B, 2 * B), dtype=torch.bool, device=device
                        )
                        for y in labels.unique():
                            idxs = (labels == y).nonzero(as_tuple=False).squeeze(1)
                            idxs_f = idxs + B
                            force_neg_mask[idxs[:, None], idxs_f[None, :]] = True
                            force_neg_mask[idxs_f[:, None], idxs[None, :]] = True

                        loss = loss_spec.fn(
                            z_all,
                            labels_all,
                            c1_all,
                            c2_all,
                            coord_space,
                            iou_mat,
                            B,
                            force_neg_mask,
                        )
                    else:
                        # no flips path (still works with PK=2 positives in-batch)
                        loss = loss_spec.fn(
                            z, labels, c1, c2, coord_space, None, None, None
                        )

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())

            avg_train = float(np.mean(losses)) if losses else 0.0

            #  Validation (same loss)
            model.eval()
            with torch.no_grad():
                all_z, all_y = [], []
                for imgs, _, labels, _, _ in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{num_epochs} [Val]",
                    leave=False,
                ):
                    imgs = imgs.to(device, non_blocking=True)
                    z = model(imgs)
                    all_z.append(z.cpu())
                    all_y.append(labels.cpu())
                embeds = torch.cat(all_z, dim=0)
                labels_all = torch.cat(all_y, dim=0)
                recalls = compute_recall_at_k(embeds, labels_all, K=(1, 5, 10))

            r1, r5, r10 = recalls[1], recalls[5], recalls[10]
            val_loss = compute_val_loss(
                model, val_loader, coord_space, device, loss_spec
            )

            score = (
                r1 + 0.5 * r5 + 0.25 * r10
            )  # composite metric, maybe change to give less value to r1 since this is very random?
            tqdm.write(
                f"Epoch {epoch + 1} | TrainLoss {avg_train:.4f} | ValLoss {val_loss:.4f} | "
                f"R@1 {r1:.3f} R@5 {r5:.3f} R@10 {r10:.3f} | Score {score:.3f}"
            )

            # Optuna reporting + pruning
            if optuna_trial is not None:
                optuna_trial.set_user_attr("best_ckpt", str(best_path))
                optuna_trial.report(score, step=epoch)
                if optuna_trial.should_prune():
                    print(f"Pruned with params:\n{cfg}")
                    raise optuna.TrialPruned()

            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_path)
                tqdm.write(f"New best saved: {best_path.name} (score={best_score:.3f})")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print(f"[Trial {optuna_trial.number}] CUDA OOM with params:\n{cfg}")
            raise optuna.TrialPruned()
        else:
            raise

    print(f"\nTraining complete. Best score={best_score:.3f} | ckpt={best_path.name}")
    return best_score  # Optuna will maximize this


def compute_recall_at_k(embeds, labels, K=(1, 5, 10)):
    sims = embeds @ embeds.T
    neg_large = sims.new_tensor(torch.finfo(sims.dtype).min / 2)
    sims.fill_diagonal_(neg_large)
    topk = sims.topk(k=max(K), dim=1).indices
    recalls = {}
    for k in K:
        topk_k = topk[:, :k]
        hits = (labels[topk_k] == labels.unsqueeze(1)).any(dim=1).float()
        recalls[k] = hits.mean().item()
    return recalls


def run_finetuning_stage_pk():
    try:
        datasets_root_dir = Path(constants.IMAGES_DIR)
        available_datasets = sorted(
            [
                d
                for d in datasets_root_dir.iterdir()
                if d.is_dir() and (d / "dem_skylines_paired").is_dir()
            ]
        )
        if not available_datasets:
            print("No datasets found with paired DEM skylines needed for training.")
            return False

        selected_dirs = available_datasets
        print("\nTraining on ALL datasets:")
        for d in selected_dirs:
            print(f"  - {d.name}")

        project_root = Path(__file__).resolve().parent.parent
        model_save_path = project_root / "AI" / "models" / "finetuned_placer_ALL.pth"

        P = 12
        K = 2

        train_placer_model_pk(
            dataset_dir=selected_dirs,
            model_save_path=model_save_path,
            num_epochs=400,
            P=P,
            K=K,
            margin=0.05,  # Will be base margin. If r_exclude is small, we will effectively force images that are similar to be separated by some base margin that might be too high
            gamma=3.0,  # Gamma will be multiplied with 1 - iou between two negative points, as an added margin required. This means that images with very different iou naturally require a larger margin in the encoding
            r_exclude_m=50.0,  # The higher r_exclude, the higher we can select the base margin.
            loss_type="infonce",
        )
        return True
    except Exception as e:
        print(f"An error occurred during model fine-tuning (PxK): {e}")
        traceback.print_exc()
        return False


def pick_any_binary_panorama(datasets_root: Path) -> Path:
    for d in sorted(datasets_root.iterdir()):
        cand = d / "binary_panorama_aligned"
        if cand.is_dir():
            # take the first png you find
            hits = sorted(cand.glob("*.png"))
            if hits:
                return hits[0]
    raise FileNotFoundError("No binary panorama found.")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    # datasets_root_dir = Path(constants.IMAGES_DIR)
    # try:
    #    sample_img = pick_any_binary_panorama(datasets_root_dir)
    #    visualize_transform_pipeline(
    #        img_path=str(sample_img),
    #        out_dir="debug_vis_sample",
    #        train=True,
    #    )
    # except Exception as e:
    #    print(f"Visualization skipped: {e}")

    # run_finetuning_stage_pk()
    # run_hpo()

    storage = f"sqlite:///{Path(__file__).resolve().parent}/optuna_skyline.db"
    study = optuna.load_study(study_name="placer_hpo", storage=storage)

    best = study.best_trial
    print("Best trial #:", best.number)
    print("Best score :", best.value)
    print("Best params:", best.params)

    # you set this in train_placer_model_pk via optuna_trial.set_user_attr("best_ckpt", ...)
    best_ckpt = best.user_attrs.get("best_ckpt")
    print("Best ckpt  :", best_ckpt)

    datasets_root_dir = Path(constants.IMAGES_DIR)
    selected_dirs = [
        d for d in datasets_root_dir.iterdir() if (d / "dem_skylines_paired").is_dir()
    ]

    project_root = Path(__file__).resolve().parent.parent
    out_path = project_root / "AI" / "models" / "best_longrun.pth"

    score = train_placer_model_pk(
        dataset_dir=selected_dirs,
        model_save_path=out_path,
        num_epochs=400,
        config=best.params,
        optuna_trial=None,
        force_val_datasets={"dataset1"},
    )
    print("Final long-run score:", score)
