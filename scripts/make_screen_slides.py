"""Slides that draw the marker and test shapes at exact physical size on a MacBook screen.

Without a printer, paper or ruler, the laptop's own screen is the reference: the 14-inch
MacBook Pro panel is 3024 px wide at 254 ppi, exactly 10 px per mm, so 302.4 mm wide. In full
screen at 100% zoom the page spans the whole panel, and the SVG below uses millimetres as its
units. Photographing the slides with a phone tests the real camera (lens, focus, processing,
noise, perspective) against exact ground truth. It does not test real materials, shadows or
thick objects.

Usage: python scripts/make_screen_slides.py  (writes real/screen_slides.html, real/slides.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photosize.marker import DICTIONARY, MARKER_ID  # noqa: E402

PANEL_WIDTH_MM = 302.4  # 3024 px at 10 px/mm (system_profiler: 3024 x 1964 Retina, Mac16,8)
MARKER_MM = 60.0  # 6 cells of 10 mm: every cell edge falls on a whole device pixel
MARKER_AT = (15.0, 15.0)

# (title, expected, shapes); shape = (kind, cx, cy, length, width, angle, grey)
SLIDES = [
    ("rectangle 70 x 35", "measure", [("rect", 190, 100, 70.0, 35.0, 20, 0)]),
    ("ellipse 60 x 40", "measure", [("ellipse", 190, 100, 60.0, 40.0, -30, 0)]),
    ("long rectangle 100 x 25", "measure", [("rect", 195, 110, 100.0, 25.0, 10, 0)]),
    ("card size 85.60 x 53.98", "measure", [("rect", 190, 105, 85.60, 53.98, 0, 0)]),
    ("coin 26.15", "measure", [("ellipse", 190, 100, 26.15, 26.15, 0, 0)]),
    ("grey rectangle 60 x 30", "measure", [("rect", 190, 100, 60.0, 30.0, 35, 150)]),
    ("two objects", "refuse: several_objects", [("rect", 160, 60, 50.0, 25.0, 0, 0), ("ellipse", 235, 140, 40.0, 30.0, 15, 0)]),
    ("touching the marker", "refuse (any reason)", [("rect", 105, 45, 60.0, 30.0, 0, 0)]),
]


def main() -> None:
    cells = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICTIONARY), MARKER_ID, 6)
    cell = MARKER_MM / cells.shape[0]
    mx, my = MARKER_AT
    marker = "".join(
        f'<rect x="{mx + x * cell:g}" y="{my + y * cell:g}" width="{cell:g}" height="{cell:g}"/>'
        for y in range(6) for x in range(6) if cells[y, x] < 128
    )
    slides_svg = []
    for i, (title, _, shapes) in enumerate(SLIDES, start=1):
        parts = []
        for kind, cx, cy, length, width, angle, grey in shapes:
            fill = f"rgb({grey},{grey},{grey})"
            if kind == "rect":
                parts.append(f'<rect x="{-length / 2:g}" y="{-width / 2:g}" width="{length:g}" height="{width:g}" '
                             f'fill="{fill}" transform="translate({cx} {cy}) rotate({angle})"/>')
            else:
                parts.append(f'<ellipse rx="{length / 2:g}" ry="{width / 2:g}" fill="{fill}" '
                             f'transform="translate({cx} {cy}) rotate({angle})"/>')
        slides_svg.append(f'<g class="slide" id="s{i}" style="display:none">{"".join(parts)}'
                          f'<text x="15" y="185" font-size="3" fill="rgb(232,232,232)">slide {i}: {title}</text></g>')

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>photo-size-check slides</title><style>
html, body {{ margin: 0; height: 100%; background: #fff; overflow: hidden; cursor: none; font-family: -apple-system, Helvetica, sans-serif; }}
#start {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; cursor: default; font-size: 20px; color: #222; text-align: center; }}
#start button {{ font-size: 22px; padding: 12px 28px; }}
#check {{ font-size: 16px; }}
svg {{ position: absolute; left: 0; top: 0; width: 100vw; height: 100vh; display: none; }}
</style></head><body>
<div id="start">
  <div>photo-size-check: slides at exact size on this MacBook's screen</div>
  <div id="check">checking…</div>
  <button id="go">Start full screen</button>
  <div style="font-size:15px;color:#666">→ / space: next slide · ←: previous · Esc: exit</div>
</div>
<svg id="stage" xmlns="http://www.w3.org/2000/svg">
  <g shape-rendering="crispEdges">{marker}</g>
  {"".join(slides_svg)}
</svg>
<script>
const PANEL_MM = {PANEL_WIDTH_MM};
const stage = document.getElementById('stage');
const check = document.getElementById('check');
let current = 1;
const total = {len(SLIDES)};
function status() {{
  // The page spans the whole built-in panel (302.4 mm) in full screen at 100% zoom, whatever
  // macOS scaling is chosen, so 1 mm = screen.width / 302.4 css px.
  const full = !!document.fullscreenElement && Math.abs(innerWidth - screen.width) <= 1;
  const ok = full;
  check.innerHTML = `screen ${{screen.width}} css px wide (x${{devicePixelRatio}}) · page ${{innerWidth}} css px · full screen: ${{full ? 'yes' : 'no'}}` +
    `<br><b style="color:${{ok ? '#080' : '#c00'}}">${{ok ? 'OK: 1 mm = ' + (screen.width / PANEL_MM).toFixed(3) + ' css px' : 'Use the built-in MacBook screen, press Cmd+0, then Start full screen'}}</b>`;
  return ok;
}}
function layout() {{
  const hmm = PANEL_MM * innerHeight / innerWidth;
  stage.setAttribute('viewBox', `0 0 ${{PANEL_MM}} ${{hmm.toFixed(3)}}`);
}}
function show(i) {{
  current = Math.min(Math.max(i, 1), total);
  document.querySelectorAll('.slide').forEach(s => s.style.display = 'none');
  document.getElementById('s' + current).style.display = '';
}}
document.getElementById('go').onclick = async () => {{
  await document.documentElement.requestFullscreen();
}};
document.addEventListener('fullscreenchange', () => {{
  const on = !!document.fullscreenElement;
  setTimeout(() => {{
    const ok = status();
    document.getElementById('start').style.display = on && ok ? 'none' : 'flex';
    stage.style.display = on && ok ? 'block' : 'none';
    layout(); show(current);
  }}, 300);
}});
addEventListener('resize', () => {{ status(); layout(); }});
addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight' || e.key === ' ') show(current + 1);
  if (e.key === 'ArrowLeft') show(current - 1);
}});
const q = new URLSearchParams(location.search);
if (q.has('slide')) {{  // render check only: show one slide without the full-screen gate
  document.getElementById('start').style.display = 'none';
  stage.style.display = 'block';
  layout(); show(+q.get('slide'));
}} else {{
  status();
}}
</script></body></html>"""
    out = Path(__file__).resolve().parents[1] / "real"
    out.mkdir(exist_ok=True)
    (out / "screen_slides.html").write_text(html)
    truth = [{"slide": i, "title": t, "expected": e, "marker_mm": MARKER_MM,
              "objects": [dict(zip(("shape", "cx_mm", "cy_mm", "length_mm", "width_mm", "angle_deg", "grey"), s)) for s in shapes]}
             for i, (t, e, shapes) in enumerate(SLIDES, start=1)]
    (out / "slides.json").write_text(json.dumps(truth, indent=1))
    print("wrote", out / "screen_slides.html", "and", out / "slides.json")


if __name__ == "__main__":
    main()
