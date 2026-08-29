from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


LAYOUT_NAME = re.compile(
    r"^(?P<family>.+)_level(?P<level>[1-5])_sample(?P<sample>[1-4])$"
)


def parse_layout_name(name: str) -> tuple[str, int, int]:
    match = LAYOUT_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"not a LIBERO-Plus target-displacement task: {name}")
    return (
        match.group("family"),
        int(match.group("level")),
        int(match.group("sample")),
    )


def parse_language(text: str) -> str:
    match = re.search(r"\(:language\s+(.+?)\)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("BDDL has no language field")
    return match.group(1).strip()


def parse_region_centers(text: str) -> dict[str, np.ndarray]:
    centers: dict[str, np.ndarray] = {}
    current: str | None = None
    looking_for_range = False
    in_regions = False
    for line in text.splitlines():
        if line.strip() == "(:regions":
            in_regions = True
            continue
        if in_regions and line.strip() == "(:fixtures":
            break
        region_match = re.match(r"^\s{6}\(([A-Za-z0-9_]+)\s*$", line)
        if in_regions and region_match is not None:
            current = region_match.group(1)
            looking_for_range = False
            continue
        if current is not None and line.strip() == "(:ranges (":
            looking_for_range = True
            continue
        if looking_for_range:
            numbers = re.findall(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", line
            )
            if len(numbers) == 4:
                x0, y0, x1, y1 = (float(value) for value in numbers)
                centers[current] = np.asarray(
                    [(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float64
                )
                looking_for_range = False
    return centers


def parse_initial_supports(text: str) -> dict[str, str]:
    block_match = re.search(r"\(:init(?P<body>.*?)\)\s*\n\s*\(:goal", text, re.DOTALL)
    if block_match is None:
        raise ValueError("BDDL has no init block")
    supports: dict[str, str] = {}
    for _, subject, support in re.findall(
        r"\((On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)",
        block_match.group("body"),
    ):
        supports[subject] = support
    return supports


def resolve_subject_xy(
    subject: str,
    supports: dict[str, str],
    centers: dict[str, np.ndarray],
    *,
    _seen: frozenset[str] = frozenset(),
) -> np.ndarray:
    if subject in _seen:
        raise ValueError(f"cyclic BDDL support relation involving {subject}")
    if subject not in supports:
        raise KeyError(f"no initial support for {subject}")
    support = supports[subject]
    if support.startswith("main_table_"):
        region = support.removeprefix("main_table_")
        if region not in centers:
            raise KeyError(f"no coordinate range for region {region}")
        return centers[region].copy()
    if support in supports:
        return resolve_subject_xy(
            support, supports, centers, _seen=_seen.union({subject})
        )
    candidates = [name for name in supports if support.startswith(f"{name}_")]
    if not candidates:
        raise KeyError(f"cannot resolve support {support} for {subject}")
    parent = max(candidates, key=len)
    return resolve_subject_xy(
        parent, supports, centers, _seen=_seen.union({subject})
    )


def task_geometry(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    centers = parse_region_centers(text)
    supports = parse_initial_supports(text)
    subjects = ("akita_black_bowl_1", "akita_black_bowl_2")
    positions = {
        subject: resolve_subject_xy(subject, supports, centers) for subject in subjects
    }
    return {
        "language": parse_language(text),
        "supports": supports,
        "region_centers": centers,
        "subject_xy": positions,
    }


def prompt_source_for_support(support: str) -> str:
    mapping = {
        "main_table_between_plate_ramekin_region": (
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
        ),
        "main_table_table_center": (
            "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
        ),
        "main_table_next_to_box_region": (
            "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
        ),
        "main_table_next_to_plate_region": (
            "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
        ),
        "main_table_next_to_ramekin_region": (
            "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
        ),
        "cookies_1": (
            "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
        ),
        "glazed_rim_porcelain_ramekin_1": (
            "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
        ),
        "flat_stove_1_cook_region": (
            "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
        ),
        "wooden_cabinet_1_top_side": (
            "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
        ),
    }
    if support not in mapping:
        raise KeyError(f"support has no verbatim official prompt source: {support}")
    return mapping[support]
