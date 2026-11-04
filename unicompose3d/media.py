from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def normalize_image(source: Path, destination: Path):
    with Image.open(source) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError("Image exceeds 40 megapixels.")
        image = ImageOps.exif_transpose(image).convert("RGBA")
        # Preserve supplied cutout transparency; flattening would restore a
        # background and defeat object masks before reconstruction.
        image.thumbnail((1536, 1536))
        image.save(destination, "PNG")
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def video_frames(path: Path, directory: Path, count=12):
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or frames < 4 or fps <= 0:
            raise ValueError("Unable to read the video. Use H.264 MP4 and move around a stationary object.")
        if frames / fps > 180:
            raise ValueError("Video exceeds three minutes. Trim the object scan.")
        selected = []
        # Small sharpest-frame search around evenly-spaced samples; angles must
        # be confirmed by the user rather than assumed from video timestamps.
        for index, center in enumerate(np.linspace(0, frames - 1, min(count, frames), dtype=int)):
            best = None
            for frame_index in sorted(set(int(np.clip(center + offset, 0, frames - 1)) for offset in (-2, 0, 2))):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                if max(frame.shape[:2]) > 1536:
                    ratio = 1536 / max(frame.shape[:2])
                    frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
                sharpness = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if best is None or sharpness > best[0]:
                    best = sharpness, frame, frame_index
            if best:
                name = f"frame_{index:02d}.png"
                cv2.imwrite(str(directory / name), best[1])
                selected.append({"file": name, "time": round(best[2] / fps, 2), "sharpness": round(best[0], 1)})
        if len(selected) < 2:
            raise ValueError("The video has too few clear frames.")
        return selected
    finally:
        capture.release()
