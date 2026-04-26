#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path
from typing import Iterable, List

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_GMR_XML = (
    "/data/haozhe/zzn/VAR_FM/ws/project/_reference/"
    "GMR_view/assets/robots/g1/g1.xml"
)

ISAAC_TO_MUJOCO23 = np.asarray([
    0, 3, 6, 9, 13, 17,
    1, 4, 7, 10, 14, 18,
    2, 5, 8,
    11, 15, 19, 21,
    12, 16, 20, 22,
], dtype=np.int64)


def _as_text(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return " | ".join(str(x) for x in arr.tolist() if str(x).strip())


def _load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _overlay_text(frame: np.ndarray, title: str, font) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    lines = []
    for line in title.splitlines():
        lines.extend(textwrap.wrap(line, width=82) or [""])
    lines = lines[:4]
    line_height = font.size + 6 if hasattr(font, "size") else 18
    box_height = 18 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 150))
    y = 8
    for line in lines:
        draw.text((12, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height
    return np.asarray(image)


def _iter_npz(paths: List[str]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.glob("*.npz"))
        else:
            yield path


def _fps(value) -> int:
    arr = np.asarray(value)
    return int(round(float(arr.reshape(-1)[0] if arr.shape else arr.item())))


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def _motion_arrays(data_npz: np.lib.npyio.NpzFile):
    files = set(data_npz.files)
    is_gmr_mt = "dof_pos" in files and ("joint_names" in files or "body_names" in files)

    if is_gmr_mt:
        root_pos = data_npz["root_pos"].astype(np.float32)
        root_rot_mujoco = _xyzw_to_wxyz(data_npz["root_rot"].astype(np.float32))
        dof = data_npz["dof_pos"].astype(np.float32)
        if dof.shape[1] == 29:
            # GMR/MT order matches the MuJoCo 23DoF order except for wrist joints.
            dof_mujoco = np.concatenate([dof[:, :19], dof[:, 22:26]], axis=1)
        elif dof.shape[1] == 23:
            dof_mujoco = dof
        else:
            raise ValueError(f"GMR/MT dof_pos must be 23 or 29 DoF, got {dof.shape}")
        return root_pos, root_rot_mujoco, dof_mujoco, "gmr_mt"

    if "joint_pos" in files:
        if "body_pos_w" in files:
            root_pos = data_npz["body_pos_w"][:, 0, :].astype(np.float32)
        else:
            root_pos = data_npz["root_pos"].astype(np.float32)

        if "body_quat_w" in files:
            root_rot_mujoco = data_npz["body_quat_w"][:, 0, :].astype(np.float32)
        else:
            # This follows GMR_view's README for legacy Isaac-style NPZ files.
            root_rot_mujoco = data_npz["root_rot"].astype(np.float32)

        joint_pos = data_npz["joint_pos"].astype(np.float32)
        if joint_pos.shape[1] == 29:
            dof_mujoco = joint_pos[:, ISAAC_TO_MUJOCO23]
        elif joint_pos.shape[1] == 23:
            dof_mujoco = joint_pos
        else:
            raise ValueError(f"Isaac joint_pos must be 23 or 29 DoF, got {joint_pos.shape}")
        return root_pos, root_rot_mujoco, dof_mujoco, "isaac"

    if {"root_pos", "root_rot", "dof_pos"}.issubset(files):
        root_pos = data_npz["root_pos"].astype(np.float32)
        root_rot_mujoco = _xyzw_to_wxyz(data_npz["root_rot"].astype(np.float32))
        dof = data_npz["dof_pos"].astype(np.float32)
        dof_mujoco = np.concatenate([dof[:, :19], dof[:, 22:26]], axis=1) if dof.shape[1] == 29 else dof
        return root_pos, root_rot_mujoco, dof_mujoco, "plain_xyzw"

    raise KeyError("Unsupported NPZ: expected GMR/MT dof_pos or Isaac joint_pos fields")


def render_one(
    npz_path: Path,
    out_path: Path,
    xml_path: Path,
    width: int,
    height: int,
    fps: int | None,
    loops: int,
    title_prefix: str,
) -> None:
    data_npz = np.load(npz_path, allow_pickle=True)
    root_pos, root_rot_wxyz, dof_pos, detected_format = _motion_arrays(data_npz)
    video_fps = fps if fps is not None else _fps(data_npz["fps"])

    primary_text = _as_text(data_npz["primary_text"]) if "primary_text" in data_npz.files else ""
    texts = _as_text(data_npz["texts"]) if "texts" in data_npz.files else ""
    title = f"{title_prefix}: {npz_path.stem} [{detected_format}]"
    if primary_text:
        title = f"{title}\ntext: {primary_text}"
    elif texts:
        title = f"{title}\ntext: {texts}"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.nq - 7 != dof_pos.shape[1]:
        raise ValueError(
            f"XML expects {model.nq - 7} DoF after root, but motion has {dof_pos.shape[1]} DoF"
        )

    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -18.0
    font = _load_font(18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=video_fps, macro_block_size=16)
    try:
        for _ in range(max(1, loops)):
            for frame_idx in range(root_pos.shape[0]):
                qpos = np.zeros(model.nq, dtype=np.float64)
                qpos[:3] = root_pos[frame_idx]
                qpos[3:7] = root_rot_wxyz[frame_idx]
                qpos[7:] = dof_pos[frame_idx]
                mj_data.qpos[:] = qpos
                mujoco.mj_forward(model, mj_data)
                camera.lookat[:] = [float(qpos[0]), float(qpos[1]), 0.8]
                renderer.update_scene(mj_data, camera=camera)
                writer.append_data(_overlay_text(renderer.render(), title, font))
    finally:
        writer.close()
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Render NPZ motions using GMR_view-compatible semantics.")
    parser.add_argument("inputs", nargs="+", help="NPZ files or directories containing NPZ files.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--xml", default=DEFAULT_GMR_XML)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--title-prefix", default="GMR view")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for npz_path in _iter_npz(args.inputs):
        out_path = out_dir / f"{npz_path.stem}_gmr.mp4"
        print(f"[render_gmr_view_mp4] {npz_path} -> {out_path}")
        render_one(
            npz_path=npz_path,
            out_path=out_path,
            xml_path=Path(args.xml),
            width=args.width,
            height=args.height,
            fps=args.fps,
            loops=args.loops,
            title_prefix=args.title_prefix,
        )


if __name__ == "__main__":
    main()
