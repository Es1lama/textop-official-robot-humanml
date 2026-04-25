#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path
from typing import Iterable, List

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_XML = (
    "/data/haozhe/zzn/VAR_FM/ws/project/P_1_Embodied-AI/"
    "sim2sim/assets/g1/g1.xml"
)


def _as_text(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return " | ".join(str(x) for x in arr.tolist() if str(x).strip())


def _load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
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


def _dof_to_29(dof_pos: np.ndarray) -> np.ndarray:
    if dof_pos.shape[1] == 29:
        return dof_pos.astype(np.float32)
    if dof_pos.shape[1] != 23:
        raise ValueError(f"Expected 23 or 29 DoF, got {dof_pos.shape}")
    joint = np.zeros((dof_pos.shape[0], 29), dtype=np.float32)
    joint[:, :19] = dof_pos[:, :19]
    joint[:, 22:26] = dof_pos[:, 19:23]
    return joint


def render_one(
    npz_path: Path,
    out_path: Path,
    xml_path: Path,
    width: int,
    height: int,
    fps: int,
    loops: int,
    title_prefix: str,
) -> None:
    data_npz = np.load(npz_path, allow_pickle=True)
    root_pos = data_npz["root_pos"].astype(np.float32)
    root_rot_xyzw = data_npz["root_rot"].astype(np.float32)
    dof_pos = _dof_to_29(data_npz["dof_pos"].astype(np.float32))
    primary_text = _as_text(data_npz["primary_text"]) if "primary_text" in data_npz.files else ""
    texts = _as_text(data_npz["texts"]) if "texts" in data_npz.files else ""
    title = f"{title_prefix}: {npz_path.stem}"
    if primary_text:
        title = f"{title}\ntext: {primary_text}"
    elif texts:
        title = f"{title}\ntext: {texts}"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.azimuth = 135.0
    camera.elevation = -18.0
    font = _load_font(18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, macro_block_size=16)
    try:
        for _ in range(max(1, loops)):
            for frame_idx in range(root_pos.shape[0]):
                qpos = np.zeros(model.nq, dtype=np.float64)
                qpos[:3] = root_pos[frame_idx]
                qpos[3:7] = root_rot_xyzw[frame_idx, [3, 0, 1, 2]]
                qpos[7:7 + min(model.nq - 7, dof_pos.shape[1])] = dof_pos[
                    frame_idx, :min(model.nq - 7, dof_pos.shape[1])
                ]
                mj_data.qpos[:] = qpos
                mujoco.mj_forward(model, mj_data)
                camera.lookat[:] = [float(qpos[0]), float(qpos[1]), 0.8]
                renderer.update_scene(mj_data, camera=camera)
                frame = renderer.render()
                writer.append_data(_overlay_text(frame, title, font))
    finally:
        writer.close()
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Render exported TextOp/DAR npz motions to MP4.")
    parser.add_argument("inputs", nargs="+", help="NPZ files or directories containing NPZ files.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--title-prefix", default="official DAR reference")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    xml_path = Path(args.xml)
    for npz_path in _iter_npz(args.inputs):
        out_path = out_dir / f"{npz_path.stem}_official.mp4"
        print(f"[render_npz_reference_mp4] {npz_path} -> {out_path}")
        render_one(
            npz_path=npz_path,
            out_path=out_path,
            xml_path=xml_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            loops=args.loops,
            title_prefix=args.title_prefix,
        )


if __name__ == "__main__":
    main()
