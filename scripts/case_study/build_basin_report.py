#!/usr/bin/env python3
"""Build the tracked summary tables and SVG figures for the basin case study.

The builder deliberately depends only on the Python standard library.  Its inputs are the
hash-bound checkpoint evaluations and the two archived Lightning metrics files.  It does not load
model weights or perform any new inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
from collections.abc import Iterable, Sequence
from typing import Any


SCHEMA = "nextlat_forgetting/basin_case_study_summary/1"
RUNS = {
    "nextlat-s1234-base": {"seed": 1234, "role": "generalizing trajectory"},
    "nextlat-s1235-base": {"seed": 1235, "role": "shortcut trajectory"},
}
COLORS = {1234: "#0072B2", 1235: "#D55E00"}
HISTORICAL_FINAL = {
    1234: {"correct": 19_991, "total": 20_000},
    1235: {"correct": 3_663, "total": 20_000},
}
HISTORICAL_SOURCE = (
    "git:5c71e5f:docs/DECISION_D47_NEXTLAT_SEED1235_REPRODUCIBILITY_DIAGNOSTIC.md"
)


class ReportBuildError(RuntimeError):
    """Raised when a report input violates the frozen scientific contract."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def write_json(path: pathlib.Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= correct <= total or total <= 0:
        raise ReportBuildError("invalid binomial count")
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def load_checkpoint_evaluations(input_root: pathlib.Path) -> dict[int, list[dict[str, Any]]]:
    trajectory_root = input_root / "results" / "trajectory"
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for job_id, run in RUNS.items():
        records: list[dict[str, Any]] = []
        for path in sorted((trajectory_root / job_id).glob("step_*.json")):
            if path.name.endswith("_repeat.json") or path.name.endswith(".progress.json"):
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema") != "nextlat_forgetting/basin_checkpoint_evaluation/1":
                raise ReportBuildError(f"unexpected checkpoint-evaluation schema: {path}")
            if value.get("job_id") != job_id or value.get("seed") != run["seed"]:
                raise ReportBuildError(f"checkpoint-evaluation identity mismatch: {path}")
            exact = value["exact_path_accuracy"]
            branch = value["teacher_forced_first_decision"]["accuracy"]
            if exact["total"] != 20_000 or branch["total"] != 20_000:
                raise ReportBuildError(f"unexpected evaluation corpus size: {path}")
            records.append(value)
        records.sort(key=lambda item: int(item["step"]))
        if len(records) != 10 or len({item["step"] for item in records}) != 10:
            raise ReportBuildError(f"expected ten unique periodic checkpoints for {job_id}")
        by_seed[int(run["seed"])] = records

    all_records = [item for records in by_seed.values() for item in records]
    invariant_fields = [
        "base_evaluator_sha256",
        "config_sha256",
        "dataset_sha256",
        "evaluator_sha256",
        "upstream_commit",
    ]
    for field in invariant_fields:
        if len({item[field] for item in all_records}) != 1:
            raise ReportBuildError(f"checkpoint evaluations disagree on {field}")
    if len({json.dumps(item["runtime"], sort_keys=True) for item in all_records}) != 1:
        raise ReportBuildError("checkpoint evaluations disagree on runtime controls")
    return by_seed


def load_loss_trajectory(
    input_root: pathlib.Path,
    seed: int,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    job_id = next(job for job, run in RUNS.items() if run["seed"] == seed)
    path = input_root / "artifacts" / "runs" / job_id / "metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("step") != "0" for row in rows):
        raise ReportBuildError(f"expected the archived all-zero metrics step column in {path}")

    train_values = [float(row["loss"]) for row in rows if row.get("loss")]
    if len(train_values) != 19_999:
        raise ReportBuildError(f"expected 19,999 ordered training rows in {path}")
    train_windows = []
    for start in range(0, len(train_values), 1_000):
        values = train_values[start : start + 1_000]
        train_windows.append(
            {
                "window_end_update": min(start + 1_000, 20_000),
                "count": len(values),
                "median_total_loss": statistics.median(values),
            }
        )

    validation_rows = [row for row in rows if row.get("val/loss")]
    if len(validation_rows) != 19:
        raise ReportBuildError(f"expected 19 ordered validation rows in {path}")
    validation = [
        {
            "update": index * 1_000,
            "total_loss": float(row["val/loss"]),
            "next_token_loss": float(row["val/next_token_loss"]),
            "logged_exact_path_accuracy": float(row["val_(5, 5)/test_accuracy"]),
        }
        for index, row in enumerate(validation_rows, start=1)
    ]

    # The logger's step column is unusable, so validate the order-based reconstruction against
    # every retained checkpoint filename, which embeds the corresponding validation loss.
    run_spec = next(item for item in freeze["runs"] if int(item["seed"]) == seed)
    anchors = 0
    for checkpoint in run_spec["checkpoints"]:
        step = int(checkpoint["step"])
        if step > 19_000:
            continue
        encoded_loss = float(pathlib.Path(checkpoint["filename"]).stem.rsplit("_", 1)[1])
        reconstructed = validation[step // 1_000 - 1]["total_loss"]
        if round(reconstructed, 4) != encoded_loss:
            raise ReportBuildError(
                f"validation-order reconstruction does not match checkpoint name at step {step}"
            )
        anchors += 1
    if anchors != 9:
        raise ReportBuildError(f"expected nine filename anchors for seed {seed}, found {anchors}")

    return {
        "metrics_csv": {
            "path": f"artifacts/runs/{job_id}/metrics.csv",
            "sha256": sha256_file(path),
        },
        "step_column_status": "all_zero_logger_artifact",
        "validation_update_reconstruction": (
            "ordered 1,000-update cadence; independently checked against nine checkpoint "
            "filename loss anchors"
        ),
        "train_1000_update_windows": train_windows,
        "validation": validation,
    }


def checkpoint_rows(by_seed: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        for value in by_seed[seed]:
            exact = value["exact_path_accuracy"]
            branch_block = value["teacher_forced_first_decision"]
            branch = branch_block["accuracy"]
            rows.append(
                {
                    "seed": seed,
                    "role": RUNS[value["job_id"]]["role"],
                    "step": int(value["step"]),
                    "exact_correct": int(exact["correct"]),
                    "exact_total": int(exact["total"]),
                    "exact_path_accuracy": float(exact["value"]),
                    "exact_path_wilson_95_low": wilson_interval(
                        int(exact["correct"]), int(exact["total"])
                    )[0],
                    "exact_path_wilson_95_high": wilson_interval(
                        int(exact["correct"]), int(exact["total"])
                    )[1],
                    "first_decision_accuracy": float(branch["value"]),
                    "first_decision_gold_margin_mean": float(
                        branch_block["gold_logit_margin"]["mean"]
                    ),
                    "token_1_accuracy": float(value["per_token_accuracy"][0]["value"]),
                    "token_2_accuracy": float(value["per_token_accuracy"][1]["value"]),
                    "token_3_accuracy": float(value["per_token_accuracy"][2]["value"]),
                    "token_4_accuracy": float(value["per_token_accuracy"][3]["value"]),
                    "token_5_accuracy": float(value["per_token_accuracy"][4]["value"]),
                    "checkpoint_sha256": value["checkpoint_sha256"],
                }
            )
    return rows


def write_csv(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ReportBuildError("refusing to write empty CSV")
    fieldnames = list(rows[0])
    fieldnames.extend(key for row in rows for key in row if key not in fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def _svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{title}</title>",
        f"<desc>{description}</desc>",
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#222}",
        ".title{font-size:22px;font-weight:650}.axis{font-size:13px}.label{font-size:15px}",
        ".note{font-size:12px;fill:#555}.grid{stroke:#ddd;stroke-width:1}",
        ".frame{fill:#fff;stroke:#888;stroke-width:1}.line{fill:none;stroke-width:3}",
        ".point{stroke:#fff;stroke-width:1.5}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]


def _line_path(points: Iterable[tuple[float, float]]) -> str:
    values = list(points)
    return " ".join(("M" if index == 0 else "L") + fmt(x) + "," + fmt(y)
                    for index, (x, y) in enumerate(values))


def _legend(lines: list[str], *, y: float, include_styles: bool = False) -> None:
    for index, seed in enumerate(sorted(COLORS)):
        x = 128 + index * 250
        lines.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 34}" y2="{y}" '
            f'stroke="{COLORS[seed]}" stroke-width="4"/>'
        )
        lines.append(
            f'<text class="label" x="{x + 44}" y="{y + 5}">seed {seed}: '
            f'{"solver" if seed == 1234 else "shortcut"}</text>'
        )
    if include_styles:
        lines.append(
            f'<line x1="670" y1="{y}" x2="704" y2="{y}" stroke="#333" stroke-width="3"/>'
        )
        lines.append(f'<text class="label" x="714" y="{y + 5}">training median</text>')
        lines.append(
            f'<line x1="890" y1="{y}" x2="924" y2="{y}" stroke="#333" '
            f'stroke-width="3" stroke-dasharray="8 6"/>'
        )
        lines.append(f'<text class="label" x="934" y="{y + 5}">validation</text>')


def accuracy_svg(rows: Sequence[dict[str, Any]]) -> str:
    width, height = 1120, 650
    left, right, top, bottom = 100, 50, 95, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    x = lambda step: left + step / 20_000 * plot_w
    y = lambda value: top + (1 - value) * plot_h
    lines = _svg_start(
        width,
        height,
        "Held-out Path-Star behavior across retained checkpoints",
        "Seed 1234 transitions from zero to near-perfect exact-path accuracy between retained "
        "steps 1000 and 3000; seed 1235 plateaus near 18 percent.",
    )
    lines.append('<text class="title" x="100" y="42">Two selected runs enter different behavioral basins</text>')
    lines.append('<text class="note" x="100" y="66">Frozen 20,000-example G(5,5) corpus; lines connect retained checkpoints only</text>')
    lines.append(f'<rect class="frame" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>')
    for value in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        yy = y(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{fmt(yy)}" x2="{left + plot_w}" y2="{fmt(yy)}"/>')
        lines.append(f'<text class="axis" text-anchor="end" x="{left - 12}" y="{fmt(yy + 5)}">{value:.1f}</text>')
    for step in [0, 5_000, 10_000, 15_000, 20_000]:
        xx = x(step)
        lines.append(f'<line class="grid" x1="{fmt(xx)}" y1="{top}" x2="{fmt(xx)}" y2="{top + plot_h}"/>')
        lines.append(f'<text class="axis" text-anchor="middle" x="{fmt(xx)}" y="{top + plot_h + 28}">{step // 1000}k</text>')
    lines.append(f'<text class="label" text-anchor="middle" x="{left + plot_w / 2}" y="{height - 22}">optimizer updates</text>')
    lines.append(f'<text class="label" text-anchor="middle" transform="translate(25 {top + plot_h / 2}) rotate(-90)">accuracy</text>')
    for seed in sorted(COLORS):
        seed_rows = [row for row in rows if row["seed"] == seed]
        points = [(x(row["step"]), y(row["exact_path_accuracy"])) for row in seed_rows]
        lines.append(f'<path class="line" stroke="{COLORS[seed]}" d="{_line_path(points)}"/>')
        for xx, yy in points:
            lines.append(f'<circle class="point" cx="{fmt(xx)}" cy="{fmt(yy)}" r="5" fill="{COLORS[seed]}"/>')
    _legend(lines, y=height - 63)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def margin_svg(rows: Sequence[dict[str, Any]]) -> str:
    width, height = 1120, 650
    left, right, top, bottom = 100, 50, 95, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = -6.0, 13.0
    x = lambda step: left + step / 20_000 * plot_w
    y = lambda value: top + (y_max - value) / (y_max - y_min) * plot_h
    lines = _svg_start(
        width,
        height,
        "Gold-token margin at the first nontrivial path decision",
        "The solver develops a large positive margin while the shortcut becomes increasingly "
        "confident in a wrong next node on average.",
    )
    lines.append('<text class="title" x="100" y="42">The shortcut becomes confidently wrong</text>')
    lines.append('<text class="note" x="100" y="66">Mean gold-next-node logit minus the largest non-gold logit, conditioned on the gold source</text>')
    lines.append(f'<rect class="frame" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>')
    for value in [-5, 0, 5, 10]:
        yy = y(value)
        stroke = "#777" if value == 0 else "#ddd"
        stroke_width = 1.5 if value == 0 else 1
        lines.append(f'<line x1="{left}" y1="{fmt(yy)}" x2="{left + plot_w}" y2="{fmt(yy)}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
        lines.append(f'<text class="axis" text-anchor="end" x="{left - 12}" y="{fmt(yy + 5)}">{value:+d}</text>')
    for step in [0, 5_000, 10_000, 15_000, 20_000]:
        xx = x(step)
        lines.append(f'<line class="grid" x1="{fmt(xx)}" y1="{top}" x2="{fmt(xx)}" y2="{top + plot_h}"/>')
        lines.append(f'<text class="axis" text-anchor="middle" x="{fmt(xx)}" y="{top + plot_h + 28}">{step // 1000}k</text>')
    lines.append(f'<text class="label" text-anchor="middle" x="{left + plot_w / 2}" y="{height - 22}">optimizer updates</text>')
    lines.append(f'<text class="label" text-anchor="middle" transform="translate(25 {top + plot_h / 2}) rotate(-90)">mean gold margin (logits)</text>')
    for seed in sorted(COLORS):
        seed_rows = [row for row in rows if row["seed"] == seed]
        points = [(x(row["step"]), y(row["first_decision_gold_margin_mean"])) for row in seed_rows]
        lines.append(f'<path class="line" stroke="{COLORS[seed]}" d="{_line_path(points)}"/>')
        for xx, yy in points:
            lines.append(f'<circle class="point" cx="{fmt(xx)}" cy="{fmt(yy)}" r="5" fill="{COLORS[seed]}"/>')
    _legend(lines, y=height - 63)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def loss_svg(loss_by_seed: dict[int, dict[str, Any]]) -> str:
    width, height = 1120, 650
    left, right, top, bottom = 100, 50, 95, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    log_min, log_max = math.log10(0.01), math.log10(4.0)
    x = lambda step: left + step / 20_000 * plot_w
    y = lambda value: top + (log_max - math.log10(value)) / (log_max - log_min) * plot_h
    lines = _svg_start(
        width,
        height,
        "Training and validation loss trajectories",
        "Training loss falls for both selected runs, but validation loss separates after the "
        "generalizing transition.",
    )
    lines.append('<text class="title" x="100" y="42">Training loss alone does not identify the solution</text>')
    lines.append('<text class="note" x="100" y="66">Training: median in ordered 1,000-row windows; validation: reconstructed 1,000-update cadence; log scale</text>')
    lines.append(f'<rect class="frame" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>')
    for value, label in [(0.01, "0.01"), (0.03, "0.03"), (0.1, "0.1"), (0.3, "0.3"), (1.0, "1"), (3.0, "3")]:
        yy = y(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{fmt(yy)}" x2="{left + plot_w}" y2="{fmt(yy)}"/>')
        lines.append(f'<text class="axis" text-anchor="end" x="{left - 12}" y="{fmt(yy + 5)}">{label}</text>')
    for step in [0, 5_000, 10_000, 15_000, 20_000]:
        xx = x(step)
        lines.append(f'<line class="grid" x1="{fmt(xx)}" y1="{top}" x2="{fmt(xx)}" y2="{top + plot_h}"/>')
        lines.append(f'<text class="axis" text-anchor="middle" x="{fmt(xx)}" y="{top + plot_h + 28}">{step // 1000}k</text>')
    lines.append(f'<text class="label" text-anchor="middle" x="{left + plot_w / 2}" y="{height - 22}">optimizer updates</text>')
    lines.append(f'<text class="label" text-anchor="middle" transform="translate(25 {top + plot_h / 2}) rotate(-90)">total objective loss</text>')
    for seed in sorted(COLORS):
        loss = loss_by_seed[seed]
        train_points = [
            (x(item["window_end_update"]), y(item["median_total_loss"]))
            for item in loss["train_1000_update_windows"]
        ]
        val_points = [(x(item["update"]), y(item["total_loss"])) for item in loss["validation"]]
        lines.append(f'<path class="line" stroke="{COLORS[seed]}" d="{_line_path(train_points)}"/>')
        lines.append(
            f'<path class="line" stroke="{COLORS[seed]}" stroke-dasharray="8 6" '
            f'd="{_line_path(val_points)}"/>'
        )
    _legend(lines, y=height - 63, include_styles=True)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def build(input_root: pathlib.Path, project_root: pathlib.Path, output_root: pathlib.Path) -> dict[str, Any]:
    freeze_path = project_root / "manifests" / "case_study" / "basin" / "artifacts.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    by_seed = load_checkpoint_evaluations(input_root)
    rows = checkpoint_rows(by_seed)
    loss_by_seed = {seed: load_loss_trajectory(input_root, seed, freeze) for seed in sorted(by_seed)}

    index_path = input_root / "results" / "trajectory" / "evaluation_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index["result_count"] != 20 or index["deterministic_repeat"]["status"] != "PASS":
        raise ReportBuildError("trajectory index is incomplete or deterministic repeat did not pass")
    reference = input_root / "results" / "trajectory" / "nextlat-s1234-base" / "step_20000.json"
    repeat = input_root / "results" / "trajectory" / "nextlat-s1234-base" / "step_20000_repeat.json"
    if reference.read_bytes() != repeat.read_bytes():
        raise ReportBuildError("local deterministic-repeat files are not byte-identical")

    first_1234, last_1234 = by_seed[1234][0], by_seed[1234][-1]
    early_1235, last_1235 = by_seed[1235][1], by_seed[1235][-1]
    new_1235_correct = int(last_1235["exact_path_accuracy"]["correct"])
    historical_1235 = HISTORICAL_FINAL[1235]
    discrepancy = historical_1235["correct"] - new_1235_correct

    summary = {
        "schema": SCHEMA,
        "study_date": "2026-08-27",
        "design": {
            "retrospective_outcome_selected": True,
            "inferential_unit": "training run",
            "run_count": 2,
            "population_inference_authorized": False,
            "formal_hypothesis_test": None,
            "heldout_examples_per_checkpoint": 20_000,
            "heldout_examples_are_not_independent_training_runs": True,
        },
        "provenance": {
            "freeze": {
                "path": "manifests/case_study/basin/artifacts.json",
                "sha256": sha256_file(freeze_path),
            },
            "evaluation_index": {
                "path": "results/trajectory/evaluation_index.json",
                "sha256": sha256_file(index_path),
            },
            "upstream_commit": first_1234["upstream_commit"],
            "dataset_sha256": first_1234["dataset_sha256"],
            "runtime": first_1234["runtime"],
            "gpu": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "loss_metrics": {str(seed): loss_by_seed[seed]["metrics_csv"] for seed in sorted(loss_by_seed)},
        },
        "deterministic_repeat": {
            "status": "PASS",
            "seed": 1234,
            "step": 20_000,
            "byte_identical": True,
            "sha256": sha256_file(reference),
        },
        "results": {
            "seed_1234": {
                "role": "selected generalizing trajectory",
                "retained_checkpoint_transition_interval": [1_000, 3_000],
                "interval_notation": "(1000, 3000]",
                "step_1000_exact_path_accuracy": first_1234["exact_path_accuracy"]["value"],
                "step_3000_exact_path_accuracy": by_seed[1234][1]["exact_path_accuracy"]["value"],
                "step_20000_exact_path_accuracy": last_1234["exact_path_accuracy"]["value"],
                "step_20000_first_decision_gold_margin_mean": last_1234[
                    "teacher_forced_first_decision"
                ]["gold_logit_margin"]["mean"],
                "archived_in_training_validation": {
                    "step_2000_logged_exact_path_accuracy": loss_by_seed[1234]["validation"][1][
                        "logged_exact_path_accuracy"
                    ],
                    "status": "supporting_nonfrozen_evaluator_trace",
                },
            },
            "seed_1235": {
                "role": "selected shortcut trajectory",
                "step_4000_exact_path_accuracy": early_1235["exact_path_accuracy"]["value"],
                "steps_4000_to_20000_exact_path_accuracy_range": [
                    min(item["exact_path_accuracy"]["value"] for item in by_seed[1235][1:]),
                    max(item["exact_path_accuracy"]["value"] for item in by_seed[1235][1:]),
                ],
                "step_20000_exact_path_accuracy": last_1235["exact_path_accuracy"]["value"],
                "step_20000_first_decision_gold_margin_mean": last_1235[
                    "teacher_forced_first_decision"
                ]["gold_logit_margin"]["mean"],
                "step_20000_per_token_accuracy": [
                    item["value"] for item in last_1235["per_token_accuracy"]
                ],
            },
        },
        "historical_evaluator_comparison": {
            "source": HISTORICAL_SOURCE,
            "same_checkpoint": True,
            "historical_correct": historical_1235["correct"],
            "historical_total": historical_1235["total"],
            "historical_accuracy": historical_1235["correct"] / historical_1235["total"],
            "new_correct": new_1235_correct,
            "new_total": int(last_1235["exact_path_accuracy"]["total"]),
            "new_accuracy": last_1235["exact_path_accuracy"]["value"],
            "correct_count_difference_historical_minus_new": discrepancy,
            "absolute_accuracy_difference": discrepancy / historical_1235["total"],
            "interpretation": (
                "The 12-example (0.06 percentage-point) difference does not alter basin "
                "classification. Its exact cause was not isolated; the two evaluator/runtime "
                "paths must not be silently merged."
            ),
        },
        "loss_log_caveat": {
            "step_column_status": "all_zero_logger_artifact",
            "validation_cadence_reconstructed": True,
            "training_summary": "median of ordered rows in 1,000-row windows",
        },
        "cost": {
            "new_training_usd": 0.0,
            "checkpoint_evaluation_session_usd_approx": 0.382,
            "basis": "Vast provider balance decrement; not an itemized per-checkpoint bill",
            "frozen_spend_stop_usd": 5.0,
        },
        "claim_boundary": {
            "supported": (
                "For these two outcome-selected artifacts, the retained trajectories separate "
                "by step 3000 and remain in distinct behavioral regimes through step 20000."
            ),
            "not_supported": [
                "NextLat population success probability",
                "a method comparison against GPT, BST, MTP, JTP, or HLP",
                "a unique causal explanation for basin selection",
                "a claim that seed 1235 is intrinsically defective",
            ],
        },
    }

    write_csv(output_root / "checkpoint_summary.csv", rows)
    loss_rows: list[dict[str, Any]] = []
    for seed in sorted(loss_by_seed):
        for item in loss_by_seed[seed]["train_1000_update_windows"]:
            loss_rows.append({"seed": seed, "series": "training_window_median", **item})
        for item in loss_by_seed[seed]["validation"]:
            loss_rows.append(
                {
                    "seed": seed,
                    "series": "validation",
                    "window_end_update": item["update"],
                    "count": "",
                    "median_total_loss": item["total_loss"],
                    "next_token_loss": item["next_token_loss"],
                    "logged_exact_path_accuracy": item["logged_exact_path_accuracy"],
                }
            )
    write_csv(output_root / "loss_summary.csv", loss_rows)
    write_text(output_root / "figures" / "exact_path_accuracy.svg", accuracy_svg(rows))
    write_text(output_root / "figures" / "first_decision_margin.svg", margin_svg(rows))
    write_text(output_root / "figures" / "loss_trajectory.svg", loss_svg(loss_by_seed))
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    summary = build(
        pathlib.Path(args.input_root).resolve(),
        pathlib.Path(args.project_root).resolve(),
        pathlib.Path(args.output_root).resolve(),
    )
    print(json.dumps({"schema": summary["schema"], "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
