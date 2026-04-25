from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from robotmdar.dtype import logger as dtypelogger
from robotmdar.dtype import seed
from robotmdar.dtype.abc import Dataset, Denoiser, Diffusion, SSampler, VAE
from robotmdar.dtype.motion import MotionDict, get_zero_abs_pose, motion_dict_to_abs_pose
from robotmdar.eval.generate_dar import ClassifierFreeWrapper, generate_next_motion
from robotmdar.train.manager import DARManager


ISAAC_29_JOINT_NAMES = np.asarray([
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
], dtype=str)


def _dof23_to_joint29(dof: np.ndarray) -> np.ndarray:
    """Invert the dataset adapter that removes the six wrist DoFs for TextOp."""
    if dof.ndim != 2 or dof.shape[1] != 23:
        raise ValueError(f"Expected dof shape [T, 23], got {dof.shape}")
    joint = np.zeros((dof.shape[0], 29), dtype=np.float32)
    joint[:, :19] = dof[:, :19]
    joint[:, 22:26] = dof[:, 19:23]
    return joint


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def _finite_diff(values: np.ndarray, fps: float) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    vel = np.gradient(values.astype(np.float32), axis=0) * float(fps)
    return vel.astype(np.float32)


def _concat_motion_dict(parts: List[MotionDict]) -> MotionDict:
    keys = parts[0].keys()
    return {key: torch.cat([part[key] for part in parts], dim=1) for key in keys}


def _tail_motion_dict(motion: MotionDict, start: int) -> MotionDict:
    return {key: value[:, start:] for key, value in motion.items()}


def _sample_motion_dict(motion: MotionDict, sample_idx: int) -> Dict[str, np.ndarray]:
    return {key: value[sample_idx].detach().cpu().numpy().astype(np.float32) for key, value in motion.items()}


def _quat_rotate_inverse_xyzw(quat: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    qvec = -quat[:, None, :3]
    qw = quat[:, None, 3:4]
    t = 2.0 * np.cross(qvec, vectors)
    return (vectors + qw * t + np.cross(qvec, t)).astype(np.float32)


def _save_sim2sim_npz(path: Path, dataset: Dataset, motion_t: MotionDict, sample_idx: int, fps: float) -> None:
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
    )


def _save_tracker_npz(path: Path, dataset: Dataset, motion_t: MotionDict, sample_idx: int, fps: float) -> None:
    """Save a best-effort TextOpTracker-style file derived from FK."""
    fk = dataset.skeleton.forward_kinematics(
        {key: value[sample_idx:sample_idx + 1] for key, value in motion_t.items()},
        return_full=True,
    )
    motion_np = _sample_motion_dict(motion_t, sample_idx)
    joint_pos = _dof23_to_joint29(motion_np["dof"])
    joint_vel = _finite_diff(joint_pos, fps)

    body_pos = fk["global_translation"][0].detach().cpu().numpy().astype(np.float32)
    body_quat_xyzw = fk["global_rotation"][0].detach().cpu().numpy().astype(np.float32)
    body_quat_wxyz = _xyzw_to_wxyz(body_quat_xyzw).astype(np.float32)
    body_lin_vel = _finite_diff(body_pos, fps)
    body_ang_vel = np.zeros_like(body_pos, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fps=np.asarray([int(round(fps))], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat_wxyz,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
    )


def main(cfg: DictConfig):
    dtypelogger.set(cfg)
    seed.set(cfg.seed)

    out_dir = Path(str(cfg.export.out_dir))
    max_motions = int(cfg.export.max_motions)
    prefix = str(cfg.export.prefix)
    save_gt = bool(cfg.export.save_gt)

    meanstd_name = 'weighted_meanstd.pkl' if cfg.data.weighted_sample and cfg.data.use_weighted_meanstd else 'meanstd.pkl'
    meanstd_path = Path(str(cfg.data.datadir)) / meanstd_name
    if not meanstd_path.exists():
        dtypelogger.logger.info(
            f"{meanstd_path} not found; building normalization stats from train split before export")
        _ = instantiate(cfg.data.train)

    val_data: Dataset = instantiate(cfg.data.val)
    val_dataiter = iter(val_data)

    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)
    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    vae.eval()
    denoiser.eval()

    manager: DARManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, None, val_data)
    cfg_denoiser = ClassifierFreeWrapper(denoiser)

    num_primitive = int(cfg.data.num_primitive)
    future_len = int(cfg.data.future_len)
    history_len = int(cfg.data.history_len)
    fps = float(val_data.fps)

    saved = 0
    batch_id = 0
    while saved < max_motions:
        batch = next(val_dataiter)
        pd_abs_pose = get_zero_abs_pose((cfg.data.batch_size,), device=cfg.device)
        gt_abs_pose = get_zero_abs_pose((cfg.data.batch_size,), device=cfg.device)
        pred_parts: List[MotionDict] = []
        gt_parts: List[MotionDict] = []
        prev_predicted_motion = None

        for pidx in range(num_primitive):
            motion, cond = batch[pidx]
            motion, cond = motion.to(cfg.device), cond.to(cfg.device)
            future_motion_gt = motion[:, -future_len:, :]
            history_motion_gt = motion[:, :history_len, :]

            if prev_predicted_motion is None:
                history_motion = history_motion_gt
            else:
                history_motion = prev_predicted_motion[:, -history_len:, :]

            future_motion_pred, future_motion_pred_dict, pd_abs_pose = generate_next_motion(
                vae=vae,
                denoiser=cfg_denoiser,
                diffusion=diffusion,
                val_data=val_data,
                text_embedding=cond,
                history_motion=history_motion,
                abs_pose=pd_abs_pose,
                future_len=future_len,
                use_full_sample=True,
                guidance_scale=float(cfg.guidance_scale),
            )
            prev_predicted_motion = future_motion_pred

            future_motion_gt_dict = val_data.reconstruct_motion(
                torch.cat([history_motion_gt, future_motion_gt], dim=1),
                abs_pose=gt_abs_pose,
                ret_fk=False,
            )
            gt_abs_pose = motion_dict_to_abs_pose(future_motion_gt_dict, idx=-2)

            pred_parts.append(future_motion_pred_dict if pidx == 0 else _tail_motion_dict(future_motion_pred_dict, history_len))
            gt_parts.append(future_motion_gt_dict if pidx == 0 else _tail_motion_dict(future_motion_gt_dict, history_len))

        pred_motion = _concat_motion_dict(pred_parts)
        gt_motion = _concat_motion_dict(gt_parts)
        batch_size = pred_motion["dof"].shape[0]

        for sample_idx in range(batch_size):
            if saved >= max_motions:
                break
            name = f"{prefix}_{saved:04d}"
            _save_sim2sim_npz(out_dir / "sim2sim" / f"{name}.npz", val_data, pred_motion, sample_idx, fps)
            _save_tracker_npz(out_dir / "tracker" / name / "motion.npz", val_data, pred_motion, sample_idx, fps)

            if save_gt:
                gt_name = f"{name}_gt"
                _save_sim2sim_npz(out_dir / "sim2sim_gt" / f"{gt_name}.npz", val_data, gt_motion, sample_idx, fps)
                _save_tracker_npz(out_dir / "tracker_gt" / gt_name / "motion.npz", val_data, gt_motion, sample_idx, fps)

            print(f"[export_dar_npz] saved {name}")
            saved += 1
        batch_id += 1

    print(f"[export_dar_npz] complete: saved {saved} motions under {out_dir}")
