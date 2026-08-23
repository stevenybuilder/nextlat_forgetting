"""Digitize bar heights from the vector (SVG) figures of arXiv:2511.05963v4.

Figures 5 (Countdown equation validity) and 6 (Path-Star accuracy) are matplotlib
SVGs, so exact bar tops can be recovered from path geometry. All other figures in
the paper are raster PNGs and CANNOT be digitized this way.

Calibration: bars are drawn under transform matrix(1,0,0,-1,0,H); the axes
rectangle spans exactly the y-range [0, 100] of the plotted percentage axis
(verified against the y-tick glyph baselines: 6 ticks 0,20,...,100 evenly spaced
over exactly the axes height).

Usage: python3 digitize_figs.py star.svg | cd.svg
"""
import re
import sys


def parse_path(d):
    pts, x, y = [], 0.0, 0.0
    for cmd, arg in re.findall(r"([MLHVCZ])([^MLHVCZ]*)", d):
        n = [float(v) for v in re.findall(r"-?\d*\.?\d+(?:e-?\d+)?", arg)]
        if cmd in "ML":
            for i in range(0, len(n), 2):
                x, y = n[i], n[i + 1]
                pts.append((x, y))
        elif cmd == "H":
            for v in n:
                x = v
                pts.append((x, y))
        elif cmd == "V":
            for v in n:
                y = v
                pts.append((x, y))
        elif cmd == "C":
            for i in range(0, len(n), 6):
                x, y = n[i + 4], n[i + 5]
                pts.append((x, y))
    return pts


def main(path):
    s = open(path, encoding="utf-8").read()
    body = s[s.find("</defs>"):]
    fills = re.findall(
        r'<path transform="matrix\(1,0,0,-1,0,[\d.]+\)" d="([^"]+)" fill="(#[0-9a-fA-F]{6})"',
        body,
    )
    # axes rectangle = the white patch with the largest area that is not the figure bg
    boxes = [(parse_path(d), c) for d, c in fills]
    axes = [(min(y for _, y in p), max(y for _, y in p))
            for p, c in boxes if c == "#ffffff"]
    y0, y1 = sorted(axes, key=lambda t: t[1] - t[0])[-2]  # second largest = axes area
    for p, c in boxes:
        ys = [y for _, y in p]
        xs = [x for x, _ in p]
        if c == "#ffffff" or max(ys) <= y0 + 0.01:
            continue
        print("%s xc=%7.2f top=%9.4f value=%6.2f" %
              (c, (min(xs) + max(xs)) / 2, max(ys), (max(ys) - y0) / (y1 - y0) * 100))


if __name__ == "__main__":
    main(sys.argv[1])
