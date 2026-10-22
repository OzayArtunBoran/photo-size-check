"""The HTTP API: structured output, refusals without sizes, and request errors with codes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import cv2
import httpx2 as httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from photosize import ObjectSpec, api, make_scene, render_scene
from photosize.synth import CORNER_BOARD


def png(image: NDArray[np.uint8]) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return bytes(buf.tobytes())


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(api.app) as c:
        yield c


def post(client: TestClient, data: bytes, **form: object) -> httpx.Response:
    r: httpx.Response = client.post("/v1/measure", files={"image": ("photo.png", data, "image/png")},
                                    data={k: str(v) for k, v in form.items()})
    return r


def one_object(seed: int = 3) -> bytes:
    scene = render_scene([ObjectSpec("rect", 130.0, 80.0, 60.0, 30.0, 20.0, 40)], rng=np.random.default_rng(seed))
    return png(scene.image)


def test_measures_with_one_marker(client: TestClient) -> None:
    r = post(client, one_object(), marker_mm=40.0)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["problems"] == []
    assert abs(body["length_mm"] - 60.0) < 0.5 and abs(body["width_mm"] - 30.0) < 0.5
    assert body["length_sd_mm"] > 0 and body["width_sd_mm"] > 0
    assert body["markers_found"] == 1 and body["misfit_mm"] is None


def test_measures_with_a_board(client: TestClient) -> None:
    scene = make_scene(np.random.default_rng(4), board=CORNER_BOARD)
    board = {"side_mm": CORNER_BOARD.side_mm, "markers": {str(k): list(v) for k, v in CORNER_BOARD.markers.items()}}
    body = post(client, png(scene.image), board=json.dumps(board)).json()
    assert body["ok"] is True, body["problems"]
    assert abs(body["length_mm"] - scene.length_mm) < 0.5 and abs(body["width_mm"] - scene.width_mm) < 0.5
    assert body["markers_found"] == 4 and body["misfit_mm"] is not None


def test_a_refusal_has_codes_and_no_sizes(client: TestClient) -> None:
    scene = render_scene([], rng=np.random.default_rng(1))  # an empty sheet
    r = post(client, png(scene.image), marker_mm=40.0)
    assert r.status_code == 200  # the request was fine; the photo cannot be measured
    body = r.json()
    assert body["ok"] is False
    assert [p["code"] for p in body["problems"]] == ["no_object"]
    assert all(body[k] is None for k in ("length_mm", "width_mm", "length_sd_mm", "width_sd_mm"))


def test_not_an_image(client: TestClient) -> None:
    r = post(client, b"not a photo", marker_mm=40.0)
    assert r.status_code == 400 and r.json()["error"] == "bad_image"


@pytest.mark.parametrize("form", [{}, {"marker_mm": 40.0, "board": '{"side_mm": 30, "markers": {"0": [0, 0]}}'},
                                  {"board": "{not json"}, {"board": '{"side_mm": -1, "markers": {"0": [0, 0]}}'}])
def test_reference_must_be_exactly_one_valid(client: TestClient, form: dict[str, object]) -> None:
    r = post(client, one_object(), **form)
    assert r.status_code == 400 and r.json()["error"] == "bad_reference"


def test_huge_image_is_refused_before_decoding(client: TestClient) -> None:
    # A PNG header claiming 20000 x 20000 pixels: refused from the header, nothing is decoded.
    header = (b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR"
              + (20000).to_bytes(4, "big") + (20000).to_bytes(4, "big") + b"\x08\x02\x00\x00\x00" + bytes(4))
    r = post(client, header, marker_mm=40.0)
    assert r.status_code == 413 and r.json()["error"] == "image_too_large"


def test_upload_size_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 1000)
    r = post(client, one_object(), marker_mm=40.0)
    assert r.status_code == 413 and r.json()["error"] == "image_too_large"


@pytest.mark.parametrize("progressive", [0, 1])
def test_jpeg_dimensions_come_from_the_header(progressive: int) -> None:
    image = np.zeros((480, 640, 3), np.uint8)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_PROGRESSIVE, progressive])
    assert ok
    assert api.image_dimensions(bytes(buf.tobytes())) == (640, 480)


def test_parallel_requests_give_the_same_answer(client: TestClient) -> None:
    data = one_object(seed=5)

    def call(_: int) -> dict[str, object]:
        result: dict[str, object] = post(client, data, marker_mm=40.0).json()
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(call, range(8)))
    assert all(a == answers[0] for a in answers) and answers[0]["ok"] is True


def test_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "version": api.VERSION}
