import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from robotmdar.dtype import logger as dtypelogger
from robotmdar.dtype import seed
from robotmdar.dtype.abc import Dataset, VAE
from robotmdar.dtype.motion import MotionDict, get_zero_abs_pose, motion_dict_to_abs_pose
from robotmdar.eval.export_dar_npz import (
    ISAAC_29_JOINT_NAMES,
    _build_text_embedding_table,
    _dof23_to_joint29,
    _match_text_batch,
    _primary_text,
    _quat_rotate_inverse_xyzw,
)
from robotmdar.train.manager import MVAEManager


def _sample_motion_dict(motion: MotionDict, sample_idx: int) -> Dict[str, np.ndarray]:
    return {key: value[sample_idx].detach().cpu().numpy().astype(np.float32) for key, value in motion.items()}


def _save_sim2sim_npz(
    path: Path,
    dataset: Dataset,
    motion_t: MotionDict,
    sample_idx: int,
    fps: float,
    texts: List[str],
) -> None:
    motion_np = _sample_motion_dict(motion_t, sample_idx)
    root_pos = motion_np["root_trans_offset"].astype(np.float32)
    root_rot_xyzw = motion_np["root_rot"].astype(np.float32)
    dof_pos = _dof23_to_joint29(motion_np["dof"])
    fk = dataset.skeleton.forward_kinematics(
        {key: value[sample_idx:sample_idx + 1] for key, value in motion_t.items()},
        return_full=False,
    )
    body_pos = fk["global_translation"][0].detach().cpu().numpy().astype(np.float32)
    body_names = np.asarray(dataset.skeleton.body_names, dtype=str)
    if body_pos.shape[1] != body_names.shape[0]:
        keep = min(body_pos.shape[1], body_names.shape[0])
        body_pos = body_pos[:, :keep]
        body_names = body_names[:keep]
    local_body_pos = _quat_rotate_inverse_xyzw(root_rot_xyzw, body_pos - root_pos[:, None, :])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fps=np.asarray(float(fps), dtype=np.float32),
        root_pos=root_pos,
        root_rot=root_rot_xyzw,
        dof_pos=dof_pos,
        joint_names=ISAAC_29_JOINT_NAMES,
        body_names=body_names,
        local_body_pos=local_body_pos,
        texts=np.asarray(texts, dtype=str),
        primary_text=np.asarray(_primary_text(texts), dtype=str),
    )


def _concat_motion_dict(parts: List[MotionDict]) -> MotionDict:
    keys = parts[0].keys()
    return {key: torch.cat([part[key] for part in parts], dim=1) for key in keys}


def _tail_motion_dict(motion: MotionDict, start: int) -> MotionDict:
    return {key: value[:, start:] for key, value in motion.items()}


def main(cfg: DictConfig):
    dtypelogger.set(cfg)
    seed.set(cfg.seed)

    out_dir = Path(str(cfg.export.out_dir))
    max_motions = int(cfg.export.max_motions)
    prefix = str(cfg.export.prefix)

    meanstd_name = 'weighted_meanstd.pkl' if cfg.data.weighted_sample and cfg.data.use_weighted_meanstd else 'meanstd.pkl'
    meanstd_path = Path(str(cfg.data.datadir)) / meanstd_name
    if not meanstd_path.exists():
        dtypelogger.logger.info(
            f"{meanstd_path} not found; building normalization stats from train split before export")
        _ = instantiate(cfg.data.train)

    val_data: Dataset = instantiate(cfg.data.val)
    val_dataiter = iter(val_data)
    text_vocab, text_embeddings = _build_text_embedding_table(val_data)

    vae: VAE = instantiate(cfg.vae)
    vae.eval()
    manager: MVAEManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, None, val_data)

    future_len = int(cfg.data.future_len)
    history_len = int(cfg.data.history_len)
    num_primitive = int(cfg.data.num_primitive)
    fps = float(val_data.fps)

    saved = 0
    manifest = []

    while saved < max_motions:
        batch = next(val_dataiter)
        batch_size_cfg = int(cfg.data.batch_size)
        pd_abs_pose = get_zero_abs_pose((batch_size_cfg,), device=cfg.device)
        gt_abs_pose = get_zero_abs_pose((batch_size_cfg,), device=cfg.device)
        recon_parts: List[MotionDict] = []
        gt_parts: List[MotionDict] = []
        batch_texts: List[List[str]] = [[] for _ in range(batch_size_cfg)]

        for pidx in range(num_primitive):
            motion, cond = batch[pidx]
            matched_texts = _match_text_batch(cond, text_vocab, text_embeddings)
            for sample_idx, text in enumerate(matched_texts):
                if sample_idx < len(batch_texts):
                    batch_texts[sample_idx].append(text)

            motion = motion.to(cfg.device)
            future_motion_gt = motion[:, -future_len:, :]
            history_motion = motion[:, :history_len, :]

            with torch.no_grad():
                latent, _ = vae.encode(future_motion=future_motion_gt, history_motion=history_motion)
                future_motion_recon = vae.decode(latent, history_motion, nfuture=future_len)

            recon_motion_dict = val_data.reconstruct_motion(
                torch.cat([history_motion, future_motion_recon], dim=1),
                abs_pose=pd_abs_pose,
                ret_fk=False,
            )
            gt_motion_dict = val_data.reconstruct_motion(
                torch.cat([history_motion, future_motion_gt], dim=1),
                abs_pose=gt_abs_pose,
                ret_fk=False,
            )

            pd_abs_pose = motion_dict_to_abs_pose(recon_motion_dict, idx=-2)
            gt_abs_pose = motion_dict_to_abs_pose(gt_motion_dict, idx=-2)

            recon_parts.append(recon_motion_dict if pidx == 0 else _tail_motion_dict(recon_motion_dict, history_len))
            gt_parts.append(gt_motion_dict if pidx == 0 else _tail_motion_dict(gt_motion_dict, history_len))

        recon_motion = _concat_motion_dict(recon_parts)
        gt_motion = _concat_motion_dict(gt_parts)
        batch_size = recon_motion["dof"].shape[0]

        for sample_idx in range(batch_size):
            if saved >= max_motions:
                break
            name = f"{prefix}_{saved:04d}"
            texts = batch_texts[sample_idx]
            recon_path = out_dir / "recon" / f"{name}_recon.npz"
            gt_path = out_dir / "gt" / f"{name}_gt.npz"
            _save_sim2sim_npz(recon_path, val_data, recon_motion, sample_idx, fps, texts)
            _save_sim2sim_npz(gt_path, val_data, gt_motion, sample_idx, fps, texts)
            manifest.append({
                "name": name,
                "primary_text": _primary_text(texts),
                "texts": texts,
                "recon_npz": str(recon_path),
                "gt_npz": str(gt_path),
                "fps": fps,
                "frames": int(recon_motion["dof"].shape[1]),
            })
            print(f"[export_mvae_recon_npz] saved {name}: {_primary_text(texts)}")
            saved += 1

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[export_mvae_recon_npz] complete: saved {saved} motions under {out_dir}")

