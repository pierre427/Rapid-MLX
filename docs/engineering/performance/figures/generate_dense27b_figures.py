"""Generate paper figures for continuous self-MTP batching.

The script uses only the Python standard library plus a local Chrome executable
for CPU-rendered PNG review copies. It reads the adjacent CSV files and writes
deterministic SVG masters and 1800-pixel-wide PNGs.
"""

from __future__ import annotations

import csv
import html
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTH = 1800
HEIGHT = 1000

INK = "#16202a"
MUTED = "#5f6b76"
GRID = "#d9dee3"
FRAME = "#9aa6b2"
BLUE = "#2563a6"
ORANGE = "#d97706"
RED = "#b7352d"
GREEN = "#25835b"
PALE_BLUE = "#eaf2fb"
PALE_ORANGE = "#fff3df"
PALE_RED = "#fbe9e7"
PALE_GREEN = "#e9f6ef"
WHITE = "#ffffff"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 26,
    weight: int = 400,
    anchor: str = "start",
    fill: str = INK,
    italic: bool = False,
) -> str:
    style = "font-style:italic;" if italic else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" style="{style}">{esc(value)}</text>'
    )


def multiline(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 24,
    weight: int = 400,
    fill: str = INK,
    gap: float = 1.25,
    anchor: str = "start",
) -> str:
    spans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else size * gap
        spans.append(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{esc(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">' + "".join(spans) + "</text>"
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = INK,
    width: float = 2,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dashed}/>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = WHITE,
    stroke: str = "none",
    radius: float = 0,
    stroke_width: float = 2,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius:.1f}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"/>'
    )


def circle(
    x: float,
    y: float,
    radius: float,
    *,
    fill: str,
    stroke: str = WHITE,
    stroke_width: float = 3,
) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def polygon(
    points: list[tuple[float, float]],
    *,
    fill: str,
    stroke: str = WHITE,
    stroke_width: float = 3,
) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def svg_document(
    body: list[str], *, title: str, width: int = WIDTH, height: int = HEIGHT
) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        f'<title id="title">{esc(title)}</title>\n'
        '<desc id="desc">Paper figure generated from the adjacent machine-readable CSV files.</desc>\n'
        f'<rect width="100%" height="100%" fill="{WHITE}"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def read_csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def marker(x: float, y: float, kind: str, color: str) -> str:
    if kind == "circle":
        return circle(x, y, 10, fill=color)
    if kind == "square":
        return rect(x - 10, y - 10, 20, 20, fill=color, stroke=WHITE, stroke_width=3)
    return polygon([(x, y - 13), (x + 13, y), (x, y + 13), (x - 13, y)], fill=color)


def plot_panel(
    body: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    x_max: float,
    y_max: float,
    y_ticks: list[float],
    title_value: str,
    y_label: str,
    series: list[tuple[list[dict[str, str]], str, str, str, str]],
) -> None:
    left, right, top, bottom = 95, 28, 55, 85
    px, py = x + left, y + top
    pw, ph = width - left - right, height - top - bottom

    body.append(text(x + 8, y + 30, title_value, size=28, weight=600))
    for tick in y_ticks:
        ty = py + ph - (tick / y_max) * ph
        body.append(line(px, ty, px + pw, ty, stroke=GRID, width=1.5))
        body.append(
            text(px - 14, ty + 8, f"{tick:g}", size=21, anchor="end", fill=MUTED)
        )
    for tick in [0, 8, 16, 24, 32, 40]:
        tx = px + (tick / x_max) * pw
        body.append(line(tx, py, tx, py + ph, stroke=GRID, width=1.2))
        body.append(text(tx, py + ph + 34, tick, size=21, anchor="middle", fill=MUTED))
    body.append(rect(px, py, pw, ph, fill="none", stroke=FRAME, stroke_width=1.5))
    body.append(
        text(px + pw / 2, py + ph + 70, "Active lanes N", size=23, anchor="middle")
    )
    body.append(
        f'<text x="{x + 28:.1f}" y="{py + ph / 2:.1f}" transform="rotate(-90 {x + 28:.1f} {py + ph / 2:.1f})" '
        f'text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="23" fill="{INK}">{esc(y_label)}</text>'
    )

    cap_x = px + (16 / x_max) * pw
    body.append(line(cap_x, py, cap_x, py + ph, stroke=GREEN, width=3, dash="10 8"))
    body.append(text(cap_x + 10, py + 27, "measured knee / cap", size=19, fill=GREEN))

    for rows, value_field, color, kind, _label in series:
        points = []
        for row in rows:
            vx = px + (float(row["n"]) / x_max) * pw
            vy = py + ph - (float(row[value_field]) / y_max) * ph
            points.append((vx, vy))
        if len(points) > 1:
            coords = " ".join(f"{vx:.1f},{vy:.1f}" for vx, vy in points)
            body.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for vx, vy in points:
            body.append(marker(vx, vy, kind, color))


def generate_throughput() -> None:
    rows = read_csv("dense27b-throughput-scaling.csv")
    groups = {
        key: [row for row in rows if row["protocol"] == key]
        for key in {row["protocol"] for row in rows}
    }
    series = [
        (
            groups["single_process_sweep"],
            "aggregate_tps",
            BLUE,
            "circle",
            "single-process sweep",
        ),
        (
            groups["fresh_process_single_n"],
            "aggregate_tps",
            ORANGE,
            "square",
            "fresh-process single-N",
        ),
        (
            groups["controller_selected_run"],
            "aggregate_tps",
            RED,
            "diamond",
            "controller-selected run",
        ),
    ]
    body = [
        text(
            80,
            68,
            "Dense 27B self-MTP scaling: throughput rises, then saturates",
            size=40,
            weight=600,
        ),
        text(
            80,
            108,
            "Qwen3.8-27B-oQ4e-mtp, k=2, M=3N, context about 256; protocols shown separately",
            size=24,
            fill=MUTED,
        ),
    ]
    plot_panel(
        body,
        x=55,
        y=145,
        width=835,
        height=690,
        x_max=42,
        y_max=300,
        y_ticks=[0, 50, 100, 150, 200, 250, 300],
        title_value="A. Aggregate service throughput",
        y_label="Aggregate throughput (token/s)",
        series=series,
    )
    per_lane_series = [
        (
            groups["single_process_sweep"],
            "per_lane_tps",
            BLUE,
            "circle",
            "single-process sweep",
        ),
        (
            groups["fresh_process_single_n"],
            "per_lane_tps",
            ORANGE,
            "square",
            "fresh-process single-N",
        ),
        (
            groups["controller_selected_run"],
            "per_lane_tps",
            RED,
            "diamond",
            "controller-selected run",
        ),
    ]
    plot_panel(
        body,
        x=910,
        y=145,
        width=835,
        height=690,
        x_max=42,
        y_max=55,
        y_ticks=[0, 10, 20, 30, 40, 50],
        title_value="B. Per-lane throughput tradeoff",
        y_label="Per-lane throughput (token/s)",
        series=per_lane_series,
    )

    legends = [
        (BLUE, "circle", "single-process sweep (raw-log-attested)"),
        (ORANGE, "square", "fresh-process single-N (transcript/screenshot-attested)"),
        (RED, "diamond", "controller-selected N=40 run (raw-log-attested)"),
    ]
    lx = 105
    for color, kind, label in legends:
        body.append(marker(lx, 882, kind, color))
        body.append(text(lx + 24, 890, label, size=21))
        lx += 520 if kind != "square" else 650
    body.append(
        text(
            80,
            950,
            "The N=16 fresh-process point reaches 270.0 token/s (5.4x the N=1 baseline); the real N=40 controller run falls to 51.7 token/s.",
            size=22,
            fill=MUTED,
        )
    )
    (HERE / "dense27b-throughput-scaling.svg").write_text(
        svg_document(body, title="Dense 27B self-MTP throughput scaling"),
        encoding="utf-8",
    )


def generate_admission() -> None:
    rows = read_csv("dense27b-admission-curve.csv")
    body = [
        text(
            80,
            68,
            "Memory-only admission over-shoots the dense-model compute knee",
            size=40,
            weight=600,
        ),
        text(
            80,
            108,
            "Real SelfMTPLaneAdmissionController decisions at 95.1 GiB free and a 20 GiB hard reserve",
            size=24,
            fill=MUTED,
        ),
    ]

    x, y, width, height = 80, 175, 1050, 650
    left, right, top, bottom = 110, 45, 50, 95
    px, py = x + left, y + top
    pw, ph = width - left - right, height - top - bottom
    contexts = [float(row["context_tokens"]) for row in rows]
    import math

    lo, hi = math.log2(min(contexts)), math.log2(max(contexts))

    def sx(value: float) -> float:
        return px + (math.log2(value) - lo) / (hi - lo) * pw

    def sy(value: float) -> float:
        return py + ph - value / 44 * ph

    body.append(rect(px, sy(44), pw, sy(16) - sy(44), fill=PALE_RED, opacity=0.85))
    body.append(
        text(
            px + pw - 10,
            sy(39),
            "above measured compute cap",
            size=20,
            anchor="end",
            fill=RED,
        )
    )
    for tick in [0, 8, 16, 24, 32, 40]:
        ty = sy(tick)
        body.append(line(px, ty, px + pw, ty, stroke=GRID, width=1.5))
        body.append(text(px - 15, ty + 7, tick, size=21, anchor="end", fill=MUTED))
    for context in contexts:
        tx = sx(context)
        body.append(line(tx, py, tx, py + ph, stroke=GRID, width=1.2))
        body.append(
            text(
                tx,
                py + ph + 35,
                f"{int(context):,}",
                size=20,
                anchor="middle",
                fill=MUTED,
            )
        )
    body.append(rect(px, py, pw, ph, fill="none", stroke=FRAME, stroke_width=1.5))
    body.append(line(px, sy(16), px + pw, sy(16), stroke=GREEN, width=4, dash="12 8"))
    body.append(
        text(
            px + pw - 8,
            sy(16) - 12,
            "new saturation cap N=16",
            size=21,
            anchor="end",
            fill=GREEN,
        )
    )
    pre_points = [
        (sx(float(row["context_tokens"])), sy(float(row["pre_fix_admitted_n"])))
        for row in rows
    ]
    post_points = [
        (sx(float(row["context_tokens"])), sy(float(row["post_fix_admitted_n"])))
        for row in rows
    ]
    pre_coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in pre_points)
    post_coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in post_points)
    body.append(
        f'<polyline points="{pre_coords}" fill="none" stroke="{BLUE}" '
        'stroke-width="5" stroke-linejoin="round"/>'
    )
    body.append(
        f'<polyline points="{post_coords}" fill="none" stroke="{GREEN}" '
        'stroke-width="5" stroke-linejoin="round" stroke-dasharray="13 8"/>'
    )
    for (vx, vy), row in zip(pre_points, rows):
        body.append(circle(vx, vy, 11, fill=BLUE))
        body.append(
            text(
                vx,
                vy - 18,
                f"N={row['pre_fix_admitted_n']}",
                size=20,
                anchor="middle",
                weight=600,
                fill=BLUE,
            )
        )
        body.append(
            text(
                vx,
                vy + 31,
                f"{row['modeled_lane_gib_k2']} GiB/lane",
                size=18,
                anchor="middle",
                fill=MUTED,
            )
        )
    for vx, vy in post_points:
        body.append(marker(vx, vy, "square", GREEN))
    body.append(circle(px + 30, 175, 9, fill=BLUE))
    body.append(text(px + 50, 182, "pre-fix: memory-only", size=19, fill=BLUE))
    body.append(marker(px + 290, 175, "square", GREEN))
    body.append(
        text(px + 312, 182, "post-fix: memory + N=16 compute cap", size=19, fill=GREEN)
    )
    body.append(
        text(
            px + pw / 2,
            py + ph + 80,
            "Per-lane context tokens (log scale)",
            size=23,
            anchor="middle",
        )
    )
    body.append(
        f'<text x="{x + 30}" y="{py + ph / 2}" transform="rotate(-90 {x + 30} {py + ph / 2})" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="23" fill="{INK}">Admitted lanes N</text>'
    )

    bx, by, bw, bh = 1220, 235, 480, 500
    body.append(
        text(bx, 200, "Measured consequence at short context", size=28, weight=600)
    )
    bar_values = [("N=16\ncompute cap", 270.0, GREEN), ("N=40\ncontroller", 51.7, RED)]
    max_value = 300
    baseline = by + bh
    for index, (label, value, color) in enumerate(bar_values):
        bar_x = bx + 45 + index * 235
        bar_h = value / max_value * bh
        body.append(rect(bar_x, baseline - bar_h, 145, bar_h, fill=color, radius=8))
        body.append(
            text(
                bar_x + 72.5,
                baseline - bar_h - 16,
                f"{value:.1f}",
                size=28,
                anchor="middle",
                weight=600,
                fill=color,
            )
        )
        lines = label.split("\n")
        body.append(
            multiline(
                bar_x + 72.5, baseline + 35, lines, size=21, anchor="middle", gap=1.2
            )
        )
    body.append(line(bx + 10, baseline, bx + bw, baseline, stroke=FRAME, width=2))
    body.append(
        text(
            bx + bw / 2,
            baseline + 105,
            "Aggregate throughput (token/s)",
            size=22,
            anchor="middle",
        )
    )
    body.append(rect(bx + 20, 820, 440, 100, fill=PALE_RED, radius=8))
    body.append(
        multiline(
            bx + 40,
            856,
            [
                "N=40 is 2.5x beyond the measured knee",
                "and delivers 80.9% less aggregate throughput.",
            ],
            size=21,
            weight=600,
            fill=RED,
        )
    )
    body.append(
        text(
            80,
            952,
            "Pre-fix admission and N=40 execution are raw-log-attested; post-fix cap is source/test-attested at 1a0a2474. N=16 throughput is transcript/screenshot-attested.",
            size=21,
            fill=MUTED,
        )
    )
    (HERE / "dense27b-admission-calibration.svg").write_text(
        svg_document(body, title="Dense 27B admission calibration"), encoding="utf-8"
    )


def generate_flashnext() -> None:
    throughput = read_csv("flashnext-throughput-scaling.csv")
    admission = read_csv("flashnext-admission-curve.csv")
    static = [row for row in throughput if row["protocol"] == "static_sweep"]
    controller = next(
        row for row in throughput if row["protocol"] == "controller_selected_run"
    )
    body = [
        text(
            80,
            68,
            "Flash-Next: the memory ceiling binds before the compute cap",
            size=40,
            weight=600,
        ),
        text(
            80,
            108,
            "Qwen3.8-Flash-Next, k=2, context about 256, PLE-NVMe sidecar active; all values raw-log-attested",
            size=24,
            fill=MUTED,
        ),
    ]

    # Panel A: throughput ladder.
    x, y, width, height = 65, 175, 830, 670
    left, right, top, bottom = 95, 30, 50, 85
    px, py = x + left, y + top
    pw, ph = width - left - right, height - top - bottom

    def sx_n(value: float) -> float:
        return px + value / 13 * pw

    def sy_tps(value: float) -> float:
        return py + ph - value / 250 * ph

    body.append(
        text(x + 5, y + 28, "A. Static ladder and controller run", size=28, weight=600)
    )
    for tick in [0, 50, 100, 150, 200, 250]:
        ty = sy_tps(tick)
        body.append(line(px, ty, px + pw, ty, stroke=GRID, width=1.4))
        body.append(text(px - 14, ty + 7, tick, size=21, anchor="end", fill=MUTED))
    for tick in [0, 2, 4, 6, 8, 10, 12]:
        tx = sx_n(tick)
        body.append(line(tx, py, tx, py + ph, stroke=GRID, width=1.2))
        body.append(text(tx, py + ph + 34, tick, size=21, anchor="middle", fill=MUTED))
    body.append(rect(px, py, pw, ph, fill="none", stroke=FRAME, stroke_width=1.5))
    points = [
        (sx_n(float(row["n"])), sy_tps(float(row["aggregate_tps"]))) for row in static
    ]
    coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in points)
    body.append(
        f'<polyline points="{coords}" fill="none" stroke="{BLUE}" '
        'stroke-width="5" stroke-linejoin="round"/>'
    )
    for (vx, vy), row in zip(points, static):
        color = RED if int(row["n"]) == 12 else BLUE
        body.append(circle(vx, vy, 11, fill=color))
        body.append(
            text(
                vx,
                vy - 18,
                f"{row['aggregate_tps']} t/s",
                size=19,
                anchor="middle",
                weight=600,
                fill=color,
            )
        )
        body.append(
            text(
                vx,
                vy + 31,
                f"{row['free_after_gib']} GiB free",
                size=17,
                anchor="middle",
                fill=MUTED,
            )
        )
    cx = sx_n(float(controller["n"]))
    cy = sy_tps(float(controller["aggregate_tps"]))
    body.append(marker(cx, cy, "diamond", GREEN))
    body.append(
        text(
            cx + 22,
            cy + 28,
            "controller N=10: 226.0 t/s",
            size=18,
            weight=600,
            fill=GREEN,
        )
    )
    body.append(
        text(px + pw / 2, py + ph + 75, "Active lanes N", size=23, anchor="middle")
    )
    body.append(
        f'<text x="{x + 25}" y="{py + ph / 2}" transform="rotate(-90 {x + 25} {py + ph / 2})" '
        f'text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="23" fill="{INK}">Aggregate throughput (token/s)</text>'
    )

    # Panel B: controller decisions by context.
    x, y, width, height = 925, 175, 810, 670
    left, right, top, bottom = 95, 30, 50, 85
    px, py = x + left, y + top
    pw, ph = width - left - right, height - top - bottom
    import math

    contexts = [float(row["context_tokens"]) for row in admission]
    lo, hi = math.log2(min(contexts)), math.log2(max(contexts))

    def sx_context(value: float) -> float:
        return px + (math.log2(value) - lo) / (hi - lo) * pw

    def sy_lanes(value: float) -> float:
        return py + ph - value / 18 * ph

    body.append(
        text(x + 5, y + 28, "B. Real memory-aware admission", size=28, weight=600)
    )
    for tick in [0, 4, 8, 12, 16]:
        ty = sy_lanes(tick)
        body.append(line(px, ty, px + pw, ty, stroke=GRID, width=1.4))
        body.append(text(px - 14, ty + 7, tick, size=21, anchor="end", fill=MUTED))
    for context in contexts:
        tx = sx_context(context)
        body.append(line(tx, py, tx, py + ph, stroke=GRID, width=1.2))
        body.append(
            text(
                tx,
                py + ph + 34,
                f"{int(context):,}",
                size=19,
                anchor="middle",
                fill=MUTED,
            )
        )
    body.append(rect(px, py, pw, ph, fill="none", stroke=FRAME, stroke_width=1.5))
    body.append(
        line(
            px, sy_lanes(16), px + pw, sy_lanes(16), stroke=ORANGE, width=4, dash="12 8"
        )
    )
    body.append(
        text(
            px + pw - 8,
            sy_lanes(16) + 27,
            "compute cap N=16 (non-binding)",
            size=20,
            anchor="end",
            fill=ORANGE,
        )
    )
    decision_points = [
        (sx_context(float(row["context_tokens"])), sy_lanes(float(row["admitted_n"])))
        for row in admission
    ]
    decision_coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in decision_points)
    body.append(
        f'<polyline points="{decision_coords}" fill="none" stroke="{GREEN}" '
        'stroke-width="5" stroke-linejoin="round"/>'
    )
    for (vx, vy), row in zip(decision_points, admission):
        body.append(marker(vx, vy, "square", GREEN))
        admitted_n = int(row["admitted_n"])
        label_y = vy - 18 if admitted_n > 2 else vy - 42
        memory_y = vy + 31 if admitted_n > 2 else vy - 17
        body.append(
            text(
                vx,
                label_y,
                f"N={row['admitted_n']}",
                size=19,
                anchor="middle",
                weight=600,
                fill=GREEN,
            )
        )
        body.append(
            text(
                vx,
                memory_y,
                f"{row['modeled_lane_gib_k2']} GiB/lane",
                size=17,
                anchor="middle",
                fill=MUTED,
            )
        )
    body.append(
        text(
            px + pw / 2,
            py + ph + 75,
            "Per-lane context tokens (log scale)",
            size=23,
            anchor="middle",
        )
    )
    body.append(
        f'<text x="{x + 25}" y="{py + ph / 2}" transform="rotate(-90 {x + 25} {py + ph / 2})" '
        f'text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="23" fill="{INK}">Admitted lanes N</text>'
    )
    body.append(
        rect(
            980, 875, 700, 68, fill=PALE_GREEN, stroke=GREEN, radius=8, stroke_width=1.5
        )
    )
    body.append(
        multiline(
            1330,
            902,
            [
                "Controller N=10 delivers 226.0 t/s,",
                "within 2.3% of the N=8 peak (231.3 t/s).",
            ],
            size=19,
            anchor="middle",
            weight=600,
            fill=GREEN,
            gap=1.2,
        )
    )
    body.append(
        text(
            80,
            970,
            "The blind N=12 sweep falls to 33.5 t/s at 9.6 GiB free; memory-aware admission stops at N=10 before that collapse.",
            size=21,
            fill=MUTED,
        )
    )
    (HERE / "flashnext-throughput-admission.svg").write_text(
        svg_document(body, title="Flash-Next throughput and admission"),
        encoding="utf-8",
    )


def generate_two_ceiling_comparison() -> None:
    body = [
        text(
            80,
            68,
            "Two models validate the two-ceiling admission rule",
            size=40,
            weight=600,
        ),
        text(
            80,
            108,
            "Admitted lanes = min(memory-safe lanes, compute-saturation cap); whichever ceiling binds first wins",
            size=24,
            fill=MUTED,
        ),
    ]

    # Panel A: ceiling comparison.
    x, y, width, height = 70, 180, 850, 680
    left, right, top, bottom = 210, 35, 70, 95
    px, py = x + left, y + top
    pw, ph = width - left - right, height - top - bottom

    def sx(value: float) -> float:
        return px + value / 44 * pw

    body.append(text(x + 5, y + 28, "A. Which ceiling binds?", size=28, weight=600))
    for tick in [0, 8, 16, 24, 32, 40]:
        tx = sx(tick)
        body.append(line(tx, py, tx, py + ph, stroke=GRID, width=1.4))
        body.append(text(tx, py + ph + 35, tick, size=21, anchor="middle", fill=MUTED))
    body.append(rect(px, py, pw, ph, fill="none", stroke=FRAME, stroke_width=1.5))
    models = [
        ("Dense 27B", 40, 16, 16, "compute binds", py + 145),
        ("Flash-Next", 10, 16, 10, "memory binds", py + 365),
    ]
    for label, memory_n, compute_n, final_n, verdict, yy in models:
        body.append(text(px - 25, yy + 7, label, size=24, anchor="end", weight=600))
        body.append(line(sx(0), yy, sx(memory_n), yy, stroke=GRID, width=8))
        body.append(circle(sx(memory_n), yy, 13, fill=BLUE))
        body.append(
            text(
                sx(memory_n),
                yy - 24,
                f"memory N={memory_n}",
                size=19,
                anchor="middle",
                fill=BLUE,
            )
        )
        body.append(marker(sx(compute_n), yy, "square", ORANGE))
        body.append(
            text(
                sx(compute_n),
                yy + 42,
                f"compute N={compute_n}",
                size=19,
                anchor="middle",
                fill=ORANGE,
            )
        )
        body.append(marker(sx(final_n), yy, "diamond", GREEN))
        body.append(
            text(
                sx(final_n),
                yy - 55,
                f"admit N={final_n}",
                size=20,
                anchor="middle",
                weight=600,
                fill=GREEN,
            )
        )
        body.append(
            text(
                px + pw / 2,
                yy + 82,
                verdict,
                size=21,
                anchor="middle",
                weight=600,
                fill=GREEN,
            )
        )
    body.append(
        text(px + pw / 2, py + ph + 78, "Concurrent lanes N", size=23, anchor="middle")
    )

    # Panel B: measured consequence.
    x, y, width, height = 950, 180, 780, 680
    body.append(
        text(x + 5, y + 28, "B. Measured controller alignment", size=28, weight=600)
    )
    chart_x, chart_y, chart_w, chart_h = x + 85, y + 90, width - 120, height - 190
    baseline = chart_y + chart_h
    body.append(
        line(chart_x, baseline, chart_x + chart_w, baseline, stroke=FRAME, width=2)
    )
    for tick in [0, 50, 100, 150, 200, 250, 300]:
        ty = baseline - tick / 300 * chart_h
        body.append(line(chart_x, ty, chart_x + chart_w, ty, stroke=GRID, width=1.3))
        body.append(text(chart_x - 14, ty + 7, tick, size=20, anchor="end", fill=MUTED))
    groups = [
        ("Dense 27B", [("peak N=16", 270.0, GREEN), ("pre-fix N=40", 51.7, RED)]),
        ("Flash-Next", [("peak N=8", 231.3, GREEN), ("controller N=10", 226.0, BLUE)]),
    ]
    group_centers = [chart_x + 170, chart_x + 495]
    for (group_label, bars), center in zip(groups, group_centers):
        for idx, (bar_label, value, color) in enumerate(bars):
            bar_x = center - 105 + idx * 125
            bar_h = value / 300 * chart_h
            body.append(rect(bar_x, baseline - bar_h, 95, bar_h, fill=color, radius=6))
            body.append(
                text(
                    bar_x + 47.5,
                    baseline - bar_h - 14,
                    f"{value:.1f}",
                    size=22,
                    anchor="middle",
                    weight=600,
                    fill=color,
                )
            )
            body.append(
                text(
                    bar_x + 47.5,
                    baseline + 31,
                    bar_label,
                    size=17,
                    anchor="middle",
                    fill=MUTED,
                )
            )
        body.append(
            text(
                center, baseline + 78, group_label, size=23, anchor="middle", weight=600
            )
        )
    body.append(
        f'<text x="{x + 25}" y="{chart_y + chart_h / 2}" transform="rotate(-90 {x + 25} {chart_y + chart_h / 2})" '
        f'text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="23" fill="{INK}">Aggregate throughput (token/s)</text>'
    )
    body.append(
        rect(
            1010,
            880,
            650,
            72,
            fill=PALE_GREEN,
            stroke=GREEN,
            radius=8,
            stroke_width=1.5,
        )
    )
    body.append(
        multiline(
            1335,
            910,
            [
                "Dense needs the compute cap; Flash is already memory-limited.",
                "The same min-rule selects the appropriate boundary.",
            ],
            size=20,
            weight=600,
            fill=GREEN,
            anchor="middle",
            gap=1.25,
        )
    )
    body.append(
        text(
            80,
            975,
            "Dense fresh-process peak is transcript/screenshot-attested; all Flash values and the dense N=40 run are raw-log-attested.",
            size=21,
            fill=MUTED,
        )
    )
    (HERE / "two-ceiling-model-comparison.svg").write_text(
        svg_document(body, title="Two-ceiling model comparison"), encoding="utf-8"
    )


def arrow(
    body: list[str],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = INK,
    width: float = 3,
) -> None:
    body.append(line(x1, y1, x2, y2, stroke=color, width=width))
    body.append(
        polygon(
            [(x2, y2), (x2 - 14, y2 - 8), (x2 - 14, y2 + 8)],
            fill=color,
            stroke=color,
            stroke_width=1,
        )
    )


def workflow_box(
    body: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    fill: str,
    stroke: str,
    sublabel: str | None = None,
) -> None:
    body.append(
        rect(x, y, width, height, fill=fill, stroke=stroke, radius=10, stroke_width=2.5)
    )
    body.append(
        text(
            x + width / 2,
            y + 39,
            label,
            size=23,
            anchor="middle",
            weight=600,
            fill=stroke,
        )
    )
    if sublabel:
        body.append(
            text(x + width / 2, y + 70, sublabel, size=18, anchor="middle", fill=MUTED)
        )


def generate_workflow() -> None:
    height = 1120
    body = [
        text(
            80,
            68,
            "Continuous self-MTP composes two independent batching axes",
            size=40,
            weight=600,
        ),
        text(
            80,
            108,
            "Each panel is one target-model weight stream; the lower-right cell is the new execution shape",
            size=24,
            fill=MUTED,
        ),
        text(
            1050,
            160,
            "more requests per target pass",
            size=23,
            weight=600,
            anchor="middle",
            fill=GREEN,
        ),
        line(560, 170, 1540, 170, stroke=GREEN, width=3),
        polygon(
            [(1540, 170), (1524, 161), (1524, 179)],
            fill=GREEN,
            stroke=GREEN,
            stroke_width=1,
        ),
    ]
    body.append(
        f'<text x="48" y="610" transform="rotate(-90 48 610)" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="23" font-weight="600" fill="{ORANGE}">more proposed positions per request</text>'
    )
    body.append(line(64, 290, 64, 920, stroke=ORANGE, width=3))
    body.append(
        polygon(
            [(64, 920), (55, 904), (73, 904)],
            fill=ORANGE,
            stroke=ORANGE,
            stroke_width=1,
        )
    )

    panels = [
        (
            110,
            210,
            760,
            350,
            "A",
            "Ordinary autoregressive decode",
            "1 request x 1 position",
            1,
            1,
            False,
        ),
        (
            920,
            210,
            760,
            350,
            "B",
            "Ordinary continuous batching",
            "N requests x 1 position",
            3,
            1,
            False,
        ),
        (
            110,
            610,
            760,
            350,
            "C",
            "Single-request self-MTP",
            "1 request x (k+1) positions",
            1,
            3,
            True,
        ),
        (
            920,
            610,
            760,
            350,
            "D",
            "Continuous self-MTP batching",
            "N requests x (k+1) positions",
            3,
            3,
            True,
        ),
    ]

    for (
        x,
        y,
        width,
        panel_h,
        letter,
        title_value,
        shape,
        n_rows,
        n_cols,
        speculative,
    ) in panels:
        is_new = letter == "D"
        body.append(
            rect(
                x,
                y,
                width,
                panel_h,
                fill=PALE_GREEN if is_new else WHITE,
                stroke=GREEN if is_new else FRAME,
                radius=12,
                stroke_width=4 if is_new else 2,
            )
        )
        body.append(
            text(
                x + 28,
                y + 42,
                letter,
                size=28,
                weight=600,
                fill=GREEN if is_new else BLUE,
            )
        )
        body.append(text(x + 72, y + 42, title_value, size=27, weight=600))
        body.append(text(x + 72, y + 75, shape, size=20, fill=MUTED))

        matrix_x = x + 75
        matrix_y = y + 125
        if speculative:
            body.append(
                rect(
                    matrix_x - 8,
                    matrix_y - 37,
                    280,
                    30,
                    fill=PALE_ORANGE,
                    stroke=ORANGE,
                    radius=5,
                    stroke_width=1.5,
                )
            )
            body.append(
                text(
                    matrix_x + 132,
                    matrix_y - 15,
                    "MTP head proposes d1, d2",
                    size=17,
                    anchor="middle",
                    fill=ORANGE,
                    weight=600,
                )
            )
        for row_index in range(n_rows):
            row_label = chr(ord("A") + row_index)
            cy = matrix_y + row_index * 47
            body.append(
                text(
                    matrix_x - 22,
                    cy + 25,
                    row_label,
                    size=18,
                    anchor="end",
                    weight=600,
                    fill=BLUE,
                )
            )
            for col_index in range(n_cols):
                cx = matrix_x + col_index * 92
                label = "t" if col_index == 0 else f"d{col_index}"
                cell_fill = PALE_BLUE if col_index == 0 else PALE_ORANGE
                cell_stroke = BLUE if col_index == 0 else ORANGE
                body.append(
                    rect(
                        cx,
                        cy,
                        76,
                        38,
                        fill=cell_fill,
                        stroke=cell_stroke,
                        radius=5,
                        stroke_width=2,
                    )
                )
                body.append(
                    text(
                        cx + 38,
                        cy + 26,
                        label,
                        size=17,
                        anchor="middle",
                        fill=cell_stroke,
                        weight=600,
                    )
                )

        target_x = x + 390
        target_label = "target verify" if speculative else "target forward"
        workflow_box(
            body,
            target_x,
            y + 120,
            295,
            105,
            target_label,
            fill=PALE_BLUE,
            stroke=BLUE,
            sublabel="stream weights once",
        )
        arrow(body, x + 340, y + 173, target_x, y + 173, color=BLUE)
        output_y = y + 295 if is_new else y + 275
        if is_new:
            body.append(
                text(
                    x + width / 2,
                    output_y,
                    "per-lane accept / rollback / EOS / commit",
                    size=21,
                    anchor="middle",
                    weight=600,
                    fill=GREEN,
                )
            )
            body.append(
                text(
                    x + width / 2,
                    output_y + 32,
                    "up to (k+1)N verified positions per weight stream",
                    size=19,
                    anchor="middle",
                    fill=MUTED,
                )
            )
        elif letter == "C":
            body.append(
                text(
                    x + width / 2,
                    output_y,
                    "accept longest verified prefix; rollback suffix",
                    size=20,
                    anchor="middle",
                    weight=600,
                    fill=GREEN,
                )
            )
            body.append(
                text(
                    x + width / 2,
                    output_y + 32,
                    "up to k+1 positions per weight stream",
                    size=19,
                    anchor="middle",
                    fill=MUTED,
                )
            )
        elif letter == "B":
            body.append(
                text(
                    x + width / 2,
                    output_y,
                    "one next-token position per lane",
                    size=20,
                    anchor="middle",
                    weight=600,
                    fill=GREEN,
                )
            )
            body.append(
                text(
                    x + width / 2,
                    output_y + 32,
                    "N positions per weight stream",
                    size=19,
                    anchor="middle",
                    fill=MUTED,
                )
            )
        else:
            body.append(
                text(
                    x + width / 2,
                    output_y,
                    "one next-token position",
                    size=20,
                    anchor="middle",
                    weight=600,
                    fill=GREEN,
                )
            )
            body.append(
                text(
                    x + width / 2,
                    output_y + 32,
                    "1 position per weight stream",
                    size=19,
                    anchor="middle",
                    fill=MUTED,
                )
            )

    legend_y = 1045
    for x, color, label in [
        (145, BLUE, "current/target position"),
        (520, ORANGE, "MTP-proposed position"),
        (940, GREEN, "verified per-lane commit"),
    ]:
        body.append(rect(x, legend_y - 19, 28, 20, fill=color, radius=3))
        body.append(text(x + 42, legend_y, label, size=20))
    body.append(
        text(
            1480,
            legend_y,
            "Scheduling changes; target distribution does not.",
            size=19,
            anchor="middle",
            fill=MUTED,
            italic=True,
        )
    )
    (HERE / "continuous-self-mtp-workflow.svg").write_text(
        svg_document(
            body, title="Continuous self-MTP execution workflow", height=height
        ),
        encoding="utf-8",
    )


def render_pngs() -> None:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome.exists():
        raise RuntimeError(f"Chrome is required for PNG rendering: {chrome}")
    sizes = {
        "dense27b-throughput-scaling": (1800, 1000),
        "dense27b-admission-calibration": (1800, 1000),
        "continuous-self-mtp-workflow": (1800, 1120),
        "flashnext-throughput-admission": (1800, 1000),
        "two-ceiling-model-comparison": (1800, 1000),
    }
    for stem, (width, height) in sizes.items():
        source = (HERE / f"{stem}.svg").resolve()
        destination = (HERE / f"{stem}.png").resolve()
        subprocess.run(
            [
                str(chrome),
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--screenshot={destination}",
                f"--window-size={width},{height}",
                source.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    generate_throughput()
    generate_admission()
    generate_flashnext()
    generate_two_ceiling_comparison()
    generate_workflow()
    render_pngs()


if __name__ == "__main__":
    main()
