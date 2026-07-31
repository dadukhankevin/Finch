"""Media telemetry: the stdlib PNG encoder, the /media route round-trip
(page.json + the run_dir/media mirror), the bounds, automatic image
reporting from live_progress, the explicit producers, and hub
thumbnails. Media is display-side telemetry only — nothing here touches
engines, scores, or the audit path."""
import base64
import json
import struct
import threading
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

from finch4.serve import (MEDIA_MAX_BYTES, MEDIA_MAX_NAMES, live_progress,
                          looks_like_image, media_client, png_data_uri,
                          serve)


def decode_png(data_uri):
    """Independent decoder for the encoder's own format (8-bit, filter 0,
    single-pass): parse chunks, inflate, strip filter bytes."""
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(raw):
        (length,) = struct.unpack(">I", raw[pos:pos + 4])
        tag = raw[pos + 4:pos + 8]
        chunks[tag] = chunks.get(tag, b"") + raw[pos + 8:pos + 8 + length]
        pos += 12 + length
    w, h, depth, color = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert depth == 8
    channels = {0: 1, 2: 3, 6: 4}[color]
    flat = zlib.decompress(chunks[b"IDAT"])
    stride = w * channels + 1
    assert all(flat[y * stride] == 0 for y in range(h))
    rows = b"".join(flat[y * stride + 1:(y + 1) * stride] for y in range(h))
    return np.frombuffer(rows, np.uint8).reshape(h, w, channels)


def test_png_encoder_round_trips_every_layout():
    rng = np.random.default_rng(0)
    hwc = rng.random((5, 7, 3)).astype(np.float32)
    out = decode_png(png_data_uri(hwc))
    assert out.shape == (5, 7, 3)
    assert np.abs(out.astype(np.float64) / 255 - hwc).max() <= 1 / 255
    # channels-first encodes to the same pixels
    assert np.array_equal(decode_png(png_data_uri(hwc.transpose(2, 0, 1))),
                          out)
    # grayscale and uint8 pass-through
    gray = decode_png(png_data_uri(rng.random((6, 4))))
    assert gray.shape == (6, 4, 1)
    u8 = (hwc * 255).astype(np.uint8)
    assert np.array_equal(decode_png(png_data_uri(u8)), u8)
    with pytest.raises(ValueError):
        png_data_uri(np.zeros((10, 10, 7)))     # 7 channels is not an image


def test_looks_like_image():
    assert looks_like_image((96, 96))
    assert looks_like_image((3, 96, 96))
    assert looks_like_image((96, 96, 3))
    assert not looks_like_image((27648,))
    assert not looks_like_image((10, 10, 10))


@pytest.fixture
def tserver(tmp_path):
    srv = serve(str(tmp_path / "run"), port=0, telemetry_only=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def call(name, body=None):
        url = f"http://127.0.0.1:{port}/{name}"
        req = (urllib.request.Request(url) if body is None else
               urllib.request.Request(url, data=json.dumps(body).encode(),
                                      method="POST"))
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    call.port = port
    call.run_dir = str(tmp_path / "run")
    yield call
    srv.shutdown()


def test_media_route_page_and_disk_mirror(tserver):
    call = tserver
    apple = np.zeros((4, 4, 3), dtype=np.uint8)
    apple[1:3, 1:3] = (200, 40, 40)
    call("media", {"name": "best fn0", "kind": "image",
                   "data": png_data_uri(apple), "epoch": 7})
    call("media", {"name": "tour", "kind": "svg",
                   "data": "<svg><circle r='3'/></svg>"})
    call("media", {"name": "worklog", "kind": "text", "data": "hi\nthere"})

    page = call("page.json")
    assert set(page["media"]) == {"best fn0", "tour", "worklog"}
    assert page["media"]["best fn0"]["epoch"] == 7
    assert np.array_equal(decode_png(page["media"]["best fn0"]["data"]),
                          apple)
    # filmstrips are images-only
    assert set(page["filmstrips"]) == {"best fn0"}

    # latest per name is mirrored to disk (spaces sanitized)
    media = f"{call.run_dir}/media"
    disk = open(f"{media}/best_fn0.png", "rb").read()
    assert disk == base64.b64decode(
        page["media"]["best fn0"]["data"].split(",", 1)[1])
    assert open(f"{media}/tour.svg").read().startswith("<svg>")
    assert open(f"{media}/worklog.txt").read() == "hi\nthere"

    # latest wins: repost under the same name replaces, never accumulates
    call("media", {"name": "worklog", "kind": "text", "data": "newer"})
    assert call("page.json")["media"]["worklog"]["data"] == "newer"
    assert open(f"{media}/worklog.txt").read() == "newer"


def test_media_bounds_are_enforced(tserver):
    call = tserver
    with pytest.raises(urllib.error.HTTPError) as e:
        call("media", {"name": "x", "kind": "hologram", "data": "?"})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        call("media", {"name": "x", "kind": "text",
                       "data": "y" * (MEDIA_MAX_BYTES + 1)})
    assert e.value.code == 400
    for i in range(MEDIA_MAX_NAMES):
        call("media", {"name": f"n{i}", "kind": "text", "data": "ok"})
    with pytest.raises(urllib.error.HTTPError) as e:
        call("media", {"name": "one-too-many", "kind": "text", "data": "!"})
    assert e.value.code == 400
    # existing names still update fine at the cap
    call("media", {"name": "n0", "kind": "text", "data": "updated"})


def _page(cb):
    with urllib.request.urlopen(cb.url.replace("/progress",
                                               "/page.json")) as r:
        return json.loads(r.read())


def test_live_progress_auto_images(tmp_path):
    import torch
    from finch4 import solve

    def fitness(phenotypes):
        return -(phenotypes.flatten(1) ** 2).mean(dim=1)

    cb = live_progress(run_dir=str(tmp_path / "img"))
    solve(fitness, output_shape=(8, 8), epochs=3, genes=4, latents=8,
          children=4, founders=2, device="cpu", seed=0,
          progress=cb, progress_every=1)
    page = _page(cb)
    assert set(page["media"]) == {"best fn0"}
    assert decode_png(page["media"]["best fn0"]["data"]).shape == (8, 8, 1)
    assert 1 <= len(page["filmstrips"]["best fn0"]) <= 10
    assert (tmp_path / "img" / "media" / "best_fn0.png").exists()
    cb.server.shutdown()

    # non-image phenotypes post nothing, and images="off" opts out
    for run, kwargs in (("flat", {}), ("off", {"images": "off"})):
        cb = live_progress(run_dir=str(tmp_path / run), **kwargs)
        solve(fitness, output_shape=(16,), epochs=2, genes=4, latents=8,
              children=4, founders=2, device="cpu", seed=0,
              progress=cb, progress_every=1)
        assert _page(cb)["media"] == {}
        cb.server.shutdown()


def test_report_media_and_media_client(tmp_path):
    cb = live_progress(run_dir=str(tmp_path / "r"))
    cb.report_media("tour", svg="<svg width='9' height='9'/>")
    send = media_client(cb.run_dir)          # discovers via server.json
    send("note", text="from another process", epoch=3)
    page = _page(cb)
    assert set(page["media"]) == {"tour", "note"}
    assert page["media"]["note"]["epoch"] == 3
    with pytest.raises(ValueError):
        cb.report_media("empty")             # one of image/svg/text required
    cb.server.shutdown()


def test_environment_report_media(tmp_path):
    from finch4 import Environment

    env = Environment([], name="clsc", live=True,
                      run_dir=str(tmp_path / "r")).compile()
    env.report_media("banner", text="hello from a classic run")
    with urllib.request.urlopen(env.url.replace("/progress",
                                                "/page.json")) as r:
        page = json.loads(r.read())
    assert page["media"]["banner"]["data"] == "hello from a classic run"
    env._server.shutdown()

    Environment([], name="quiet").report_media("x", text="y")   # no-op


def test_hub_cards_show_media_thumbnails(tmp_path, monkeypatch):
    from finch4 import hub
    from finch4.serve import register_run

    monkeypatch.setenv("FINCH4_REGISTRY", str(tmp_path / "reg.jsonl"))
    run = tmp_path / "applerun"
    (run / "media").mkdir(parents=True)
    apple = np.full((4, 4, 3), 128, dtype=np.uint8)
    png = base64.b64decode(png_data_uri(apple).split(",", 1)[1])
    (run / "media" / "best_fn0.png").write_bytes(png)
    (run / "media" / "notes.txt").write_text("not a thumbnail")
    with open(run / "telemetry.jsonl", "w") as f:
        f.write(json.dumps({"epoch": 0, "evaluations": 4,
                            "best": {"fn0": -1.0}}) + "\n")
    register_run(str(run), port=1)

    (card,) = hub.hub_data()["cards"]
    assert [m["name"] for m in card["media"]] == ["best_fn0"]
    assert np.array_equal(decode_png(card["media"][0]["src"]), apple)
