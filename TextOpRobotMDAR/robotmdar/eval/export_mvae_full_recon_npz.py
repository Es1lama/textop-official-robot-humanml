import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from robotmdar.dtype import logger as dtypelogger
from robotmdar.dtype import seed
from robotmdar.dtype.abc import Dataset, VAE
from robotmdar.dtype.motion import (
    AbsolutePose,
    MotionDict,
    motion_dict_to_abs_pose,
    motion_dict_to_feature,
)
from robotmdar.eval.export_dar_npz import _primary_text
from robotmdar.eval.export_mvae_recon_npz import _save_sim2sim_npz
from robotmdar.train.manager import MVAEManager


def _torch_motion(sample_motion: Dict[str, np.ndarray]) -> MotionDict:
    return {
        key: torch.tensor(value, dtype=torch.float32).unsqueeze(0)
        for key, value in sample_motion.items()
        if key in {"root_trans_offset", "dof", "root_rot", "contact_mask"}
    }


def _to_device_abs_pose(abs_pose: AbsolutePose, device: str) -> AbsolutePose:
    return {
        "root_trans_offset": abs_pose["root_trans_offset"].to(device),
        "root_rot": abs_pose["root_rot"].to(device),
    }


def _concat_motion_dict(parts: List[MotionDict]) -> MotionDict:
    return {key: torch.cat([part[key] for part in parts], dim=1) for key in parts[0].keys()}


def _tail_motion_dict(motion: MotionDict, start: int) -> MotionDict:
    return {key: value[:, start:] for key, value in motion.items()}


def _unique_texts(sample: Dict) -> List[str]:
    texts = []
    for ann in sample.get("frame_ann", []):
        text = str(ann[2]).strip()
        if text and text not in texts:
            texts.append(text)
    return texts


def _build_text_to_label_index(dataset: Dataset) -> Dict[str, int]:
    text_to_idx: Dict[str, int] = {}
    for idx, sample in enumerate(dataset.raw_data):
        for text in _unique_texts(sample):
            text_to_idx.setdefault(text, idx)
    return text_to_idx


def _manifest_rows(path: Optional[str]) -> Optional[List[Dict]]:
    if not path:
        return None
    manifest_path = Path(path)
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _select_samples(dataset: Dataset, max_motions: int, source_manifest: Optional[str]) -> List[Tuple[str, int]]:
    rows = _manifest_rows(source_manifest)
    if not rows:
        return [
            (f"mvae_{out_idx:04d}", int(dataset.valid_indices[out_idx]))
            for out_idx in range(min(max_motions, len(dataset.valid_indices)))
        ]

    text_to_idx = _build_text_to_label_index(dataset)
    selected: List[Tuple[str, int]] = []
    for row in rows[:max_motions]:
        label_idx = None
        for text in row.get("texts", []):
            label_idx = text_to_idx.get(str(text).strip())
            if label_idx is not None:
                break
        if label_idx is None:
            primary = str(row.get("primary_text", ""))
            for text in primary.split(" | "):
                label_idx = text_to_idx.get(text.strip())
                if label_idx is not None:
                    break
        if label_idx is None:
            raise ValueError(f"Could not match manifest row to validation label: {row}")
        selected.append((str(row.get("name", f"mvae_{len(selected):04d}")), label_idx))
    return selected


def _reconstruct_full_motion(
    dataset: Dataset,
    vae: VAE,
    sample: Dict,
    history_len: int,
    future_len: int,
    device: str,
) -> Tuple[MotionDict, MotionDict]:
    motion_t = _torch_motion(sample["motion"])
    motion_feature, abs_pose = motion_dict_to_feature(motion_t, dataset.skeleton)
    motion_feature = dataset.normalize(motion_feature).to(device)

    pd_abs_pose = _to_device_abs_pose(abs_pose, device)
    gt_abs_pose = _to_device_abs_pose(abs_pose, device)
    recon_parts: List[MotionDict] = []
    gt_parts: List[MotionDict] = []

    total_frames = int(motion_feature.shape[1])
    start = 0
    while start + history_len < total_frames:
        current_future = min(future_len, total_frames - start - history_len)
        if current_future <= 0:
            break

        history_motion = motion_feature[:, start:start + history_len]
        future_motion_gt = motion_feature[
            :,
            start + history_len:start + history_len + current_future,
        ]

        with torch.no_grad():
            latent, _ = vae.encode(future_motion=future_motion_gt, history_motion=history_motion)
            future_motion_recon = vae.decode(latent, history_motion, nfuture=current_future)

        recon_motion_dict = dataset.reconstruct_motion(
            torch.cat([history_motion, future_motion_recon], dim=1),
            abs_pose=pd_abs_pose,
            ret_fk=False,
        )
        gt_motion_dict = dataset.reconstruct_motion(
            torch.cat([history_motion, future_motion_gt], dim=1),
            abs_pose=gt_abs_pose,
            ret_fk=False,
        )

        pd_abs_pose = motion_dict_to_abs_pose(recon_motion_dict, idx=-history_len)
        gt_abs_pose = motion_dict_to_abs_pose(gt_motion_dict, idx=-history_len)

        recon_parts.append(
            recon_motion_dict if not recon_parts else _tail_motion_dict(recon_motion_dict, history_len)
        )
        gt_parts.append(gt_motion_dict if not gt_parts else _tail_motion_dict(gt_motion_dict, history_len))
        start += current_future

    return _concat_motion_dict(recon_parts), _concat_motion_dict(gt_parts)


def main(cfg: DictConfig):
    dtypelogger.set(cfg)
    seed.set(cfg.seed)

    out_dir = Path(str(cfg.export.out_dir))
    max_motions = int(cfg.export.max_motions)
    source_manifest = (
        str(cfg.export.source_manifest)
        if OmegaConf.select(cfg, "export.source_manifest") not in (None, "null", "")
        else None
    )

    meanstd_name = "weighted_meanstd.pkl" if cfg.data.weighted_sample and cfg.data.use_weighted_meanstd else "meanstd.pkl"
    meanstd_path = Path(str(cfg.data.datadir)) / meanstd_name
    if not meanstd_path.exists():
        dtypelogger.logger.info(
            f"{meanstd_path} not found; building normalization stats from train split before export"
        )
        _ = instantiate(cfg.data.train)

    val_data: Dataset = instantiate(cfg.data.val)
    selected = _select_samples(val_data, max_motions=max_motions, source_manifest=source_manifest)

    vae: VAE = instantiate(cfg.vae)
    vae.eval()
    manager: MVAEManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, None, val_data)

    history_len = int(cfg.data.history_len)
    future_len = int(cfg.data.future_len)
    fps = float(val_data.fps)
    manifest = []

    for name, label_idx in selected:
        sample = val_data.raw_data[label_idx]
        texts = _unique_texts(sample)
        recon_motion, gt_motion = _reconstruct_full_motion(
            dataset=val_data,
            vae=vae,
            sample=sample,
            history_len=history_len,
            future_len=future_len,
            device=str(cfg.device),
        )

        recon_path = out_dir / "recon" / f"{name}_recon.npz"
        gt_path = out_dir / "gt" / f"{name}_gt.npz"
        _save_sim2sim_npz(recon_path, val_data, recon_motion, 0, fps, texts)
        _save_sim2sim_npz(gt_path, val_data, gt_motion, 0, fps, texts)

        frames = int(recon_motion["dof"].shape[1])
        source_npz = sample.get("feat_p", sample.get("motion", {}).get("source", ""))
        manifest.append({
            "name": name,
            "primary_text": _primary_text(texts),
            "texts": texts,
            "source_label_idx": int(label_idx),
            "source_npz": str(source_npz),
            "recon_npz": str(recon_path),
            "gt_npz": str(gt_path),
            "fps": fps,
            "frames": frames,
            "duration_sec": frames / fps,
            "history_len": history_len,
            "future_len": future_len,
        })
        print(f"[export_mvae_full_recon_npz] saved {name}: frames={frames}, text={_primary_text(texts)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[export_mvae_full_recon_npz] complete: saved {len(manifest)} motions under {out_dir}")
