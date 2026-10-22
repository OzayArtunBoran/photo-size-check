"""HTTP API: post a photo, get a size with its uncertainty, or a machine-readable refusal.

    uvicorn photosize.api:app --workers 2

The response is built for a frontend that must not show a wrong size. `ok` decides: when it is
false, every size is null and `problems` lists codes from `Code` with a readable detail. Bad
requests (not an image, too large, no reference) get HTTP errors with their own code; a photo
that cannot be measured is not a bad request and gets 200 with ok=false.

Cost and concurrency:
- The photo's pixel count is read from its JPEG or PNG header and checked before decoding, so a
  small file that would decode to a huge image is refused without allocating it.
- measure() keeps no shared state. At most one measurement per CPU core runs at a time (more
  only make each slower), each with one OpenCV thread; further requests wait their turn.
- Endpoints are plain functions, so FastAPI runs them in its thread pool and the event loop stays
  free while a photo is measured.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from .marker import Board, single
from .measure import Code, Measurement, measure

VERSION = "0.3.0"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 64_000_000  # a 50 MP phone photo fits

_slots = threading.BoundedSemaphore(max(1, os.cpu_count() or 1))


class ProblemOut(BaseModel):
    code: Code
    detail: str


class SizeOut(BaseModel):
    ok: bool = Field(description="False: the photo cannot be trusted; every size is null and must not be shown")
    length_mm: float | None
    width_mm: float | None
    length_sd_mm: float | None = Field(description="1 standard deviation; report ±2 sd")
    width_sd_mm: float | None
    problems: list[ProblemOut]
    markers_found: int
    misfit_mm: float | None = Field(description="rms distance of the markers' corners from one plane (several markers)")
    version: str


class ErrorOut(BaseModel):
    error: Literal["bad_image", "image_too_large", "bad_reference"]
    detail: str


class BoardIn(BaseModel):
    side_mm: float = Field(gt=0, description="printed side of every marker, measured on the print")
    markers: dict[int, tuple[float, float]] = Field(min_length=1, description="marker id -> top-left corner in mm")


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    before = cv2.getNumThreads()
    cv2.setNumThreads(1)  # concurrency comes from parallel requests, not from threads inside one
    yield
    cv2.setNumThreads(before)


app = FastAPI(title="photo-size-check", version=VERSION, lifespan=_lifespan,
              description="Size of a flat object from one photo with printed ArUco markers, or a reason to refuse.")


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG or JPEG header, without decoding; None for anything else."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # a start-of-frame segment
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return None


def to_output(m: Measurement) -> SizeOut:
    def size(v: float) -> float | None:
        return round(v, 3) if m.ok else None

    return SizeOut(
        ok=m.ok, length_mm=size(m.length_mm), width_mm=size(m.width_mm),
        length_sd_mm=size(m.length_sd_mm), width_sd_mm=size(m.width_sd_mm),
        problems=[ProblemOut(code=p.code, detail=p.detail) for p in m.problems],
        markers_found=len(m.markers_px),
        misfit_mm=round(m.misfit_mm, 3) if np.isfinite(m.misfit_mm) and len(m.markers_px) > 1 else None,
        version=VERSION,
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": VERSION}


@app.post("/v1/measure", response_model=SizeOut,
          responses={400: {"model": ErrorOut}, 413: {"model": ErrorOut}})
def measure_photo(
    image: Annotated[UploadFile, File(description="JPEG or PNG photo")],
    marker_mm: Annotated[float | None, Form(gt=0, description="one marker (id 0): its printed side in mm")] = None,
    board: Annotated[str | None, Form(description='several markers, JSON: {"side_mm": 30, "markers": {"0": [10, 10], ...}}')] = None,
) -> SizeOut | JSONResponse:
    if (marker_mm is None) == (board is None):
        return _error(400, "bad_reference", "give exactly one of marker_mm or board")
    reference: Board
    if board is not None:
        try:
            parsed = BoardIn.model_validate(json.loads(board))
        except (json.JSONDecodeError, ValidationError) as e:
            return _error(400, "bad_reference", f"board is not valid: {e}")
        reference = Board(parsed.side_mm, dict(parsed.markers))
    else:
        assert marker_mm is not None
        reference = single(marker_mm)

    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return _error(413, "image_too_large", f"more than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    dims = image_dimensions(data)
    if dims is None:
        return _error(400, "bad_image", "not a JPEG or PNG file")
    if dims[0] * dims[1] > MAX_PIXELS:
        return _error(413, "image_too_large", f"{dims[0]} x {dims[1]} pixels; at most {MAX_PIXELS // 1_000_000} MP")
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        return _error(400, "bad_image", "the file could not be decoded")

    with _slots:
        m = measure(np.asarray(decoded, np.uint8), reference)
    return to_output(m)
