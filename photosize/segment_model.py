"""An open-source segmentation model as an alternative to step 4 of measure().

U^2-Netp (Qin et al., 2020; Apache-2.0), the small variant of U^2-Net, predicts a salient-object
probability for every pixel of a 320 x 320 input. The ONNX file comes from the rembg project
(4.6 MB). It runs on OpenCV's own DNN module, so it adds no dependency.

The model file is downloaded once into a cache directory and checked against its SHA-256 before
every load; it is not part of the repository. Set PHOTOSIZE_MODEL to use a local copy.

Concurrency: one cv2.dnn.Net must not run two forward passes at once, so calls on one instance
are serialised with a lock. The download writes to a unique temporary file and renames it into
place, so two processes fetching at the same time cannot leave a half-written model behind.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
INPUT_PX = 320
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet statistics, as in U^2-Net and rembg
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def model_path() -> Path:
    env = os.environ.get("PHOTOSIZE_MODEL")
    if env:
        return Path(env)
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache / "photosize" / "u2netp.onnx"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_model(path: Path | None = None) -> Path:
    """The model file, downloaded first if missing. Raises if its checksum does not match."""
    path = path or model_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
        os.close(fd)
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp)
            if _sha256(Path(tmp)) != MODEL_SHA256:
                raise RuntimeError(f"downloaded model does not match its SHA-256: {MODEL_URL}")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    if _sha256(path) != MODEL_SHA256:
        raise RuntimeError(f"{path} does not match the expected SHA-256; delete it to download again")
    return path


class U2NetP:
    """Salient-object probability maps, and an object mask in the form measure() accepts."""

    def __init__(self, path: Path | None = None, threshold: float = 0.5, passes: int = 2) -> None:
        self._net = cv2.dnn.readNetFromONNX(str(ensure_model(path)))
        self._lock = threading.Lock()
        self.threshold = threshold
        self.passes = passes

    def probability(self, gray: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Per-pixel probability for a grey crop, at the crop's own size.

        Preprocessing follows rembg: scale by the crop's maximum, ImageNet normalisation, a
        320 x 320 input; the fused output is min-max normalised.
        """
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        small = cv2.resize(rgb, (INPUT_PX, INPUT_PX), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (small / max(1.0, float(small.max())) - _MEAN) / _STD
        blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
        with self._lock:
            self._net.setInput(blob)
            out = np.asarray(self._net.forward(), np.float32)[0, 0]  # the fused map (first output)
        lo, hi = float(out.min()), float(out.max())
        p = (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)
        h, w = gray.shape
        return np.asarray(cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)

    def _mask_in(self, gray: NDArray[np.uint8], box: tuple[int, int, int, int]) -> NDArray[np.uint8]:
        x, y, w, h = box
        mask = np.zeros(gray.shape, np.uint8)
        p = self.probability(gray[y:y + h, x:x + w])
        mask[y:y + h, x:x + w] = np.where(p > self.threshold, 255, 0).astype(np.uint8)
        return mask

    def object_mask(self, gray: NDArray[np.uint8], search: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Pass 1 runs on the whole search area. With passes=2, the model runs again on a crop
        around the largest region found, so the object fills more of the 320 x 320 input and the
        outline is finer.

        Everything outside the search area (the markers, the table) is painted with the sheet's
        median grey first: left in, the markers are the most salient thing in the photo and the
        model segments them instead of the object (LOG.md).
        """
        if cv2.countNonZero(search) == 0:
            return np.zeros_like(search)
        gray = gray.copy()
        gray[search == 0] = np.uint8(np.median(gray[search > 0]))
        bx, by, bw, bh = cv2.boundingRect(search)
        mask = np.asarray(cv2.bitwise_and(self._mask_in(gray, (bx, by, bw, bh)), search), np.uint8)
        if self.passes < 2:
            return mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n < 2:
            return mask
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = (int(v) for v in stats[i, :4])
        pad = max(8, int(0.25 * max(w, h)))
        H, W = gray.shape
        x0, y0 = max(0, x - pad), max(0, y - pad)
        box = (x0, y0, min(W, x + w + pad) - x0, min(H, y + h + pad) - y0)
        fine = cv2.bitwise_and(self._mask_in(gray, box), search)
        rest = mask.copy()
        rest[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = 0  # other regions keep pass 1
        return np.asarray(cv2.bitwise_or(rest, fine), np.uint8)
