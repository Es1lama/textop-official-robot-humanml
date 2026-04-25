#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _npz_text(path: str) -> str:
    if not path:
        return ""
    data = np.load(path, allow_pickle=True)
    if "primary_text" in data.files:
        text = str(np.asarray(data["primary_text"]).item())
        if text.strip():
            return text
    if "texts" in data.files:
        return " | ".join(str(x) for x in data["texts"].tolist() if str(x).strip())
    return ""


def _overlay(frame, title, font):
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


def main():
    parser = argparse.ArgumentParser(description="Overlay motion text metadata on an MP4.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--motion-npz", default="")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    text = _npz_text(args.motion_npz)
    title = args.title or Path(args.input).stem
    if text:
        title = f"{title}\ntext: {text}"

    reader = imageio.get_reader(args.input)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=fps, macro_block_size=16)
    font = _load_font(18)
    try:
        for frame in reader:
            writer.append_data(_overlay(frame, title, font))
    finally:
        reader.close()
        writer.close()


if __name__ == "__main__":
    main()
