from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FAMILIES = ("jpeg", "blur", "scale", "noise", "color", "crop")
SOURCES = ("CommunityForensics-Small", "GenImage", "SID_Set")
EXPECTED_CONDITIONS = {
    "clean",
    "jpeg_90",
    "jpeg_70",
    "jpeg_50",
    "jpeg_30",
    "blur_0.5",
    "blur_1.0",
    "blur_2.0",
    "scale_0.5",
    "scale_0.25",
    "noise_0.02",
    "noise_0.05",
    "noise_0.10",
    "color_-0.20",
    "color_0.20",
    "crop_0.80",
}
DEGRADED_CONDITIONS = EXPECTED_CONDITIONS - {"clean"}
# Mirrors src/metrics.py: robust_score = 0.8 * mean + 0.2 * worst over degraded conditions.
ROBUST_MEAN_WEIGHT = 0.8
ROBUST_WORST_WEIGHT = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply pre-declared robustness gates to one or more evaluation JSON files."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    # Current-round bound gates (pre-registered defaults; see docs/MODERN_ADAPTATION_PLAN.md).
    parser.add_argument("--min-robust-score", type=float, default=0.9325)
    parser.add_argument("--min-cf-robust-score", type=float, default=0.9239)
    parser.add_argument("--max-source-robust-drop", type=float, default=0.003)
    parser.add_argument("--max-clean-drop", type=float, default=0.002)
    parser.add_argument("--max-global-family-drop", type=float, default=0.003)
    parser.add_argument("--max-source-family-drop", type=float, default=0.005)
    parser.add_argument("--max-noise-family-drop", type=float, default=0.002)
    parser.add_argument("--max-blur2-drop", type=float, default=0.002)
    # Historical five-gate thresholds (report-only; never gates acceptance).
    parser.add_argument("--min-robust-gain", type=float, default=0.001)
    parser.add_argument("--min-noise-gain", type=float, default=0.001)
    parser.add_argument("--legacy-max-family-drop", type=float, default=0.005)
    return parser.parse_args()


def load_metrics(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "conditions" not in payload or "robustness" not in payload:
        raise ValueError(f"Missing conditions/robustness in {source}")
    payload["_path"] = str(source)
    return payload


def aucs(payload: dict[str, Any]) -> dict[str, float]:
    values = {
        name: float(metrics["overall"]["roc_auc"])
        for name, metrics in payload["conditions"].items()
    }
    if set(values) != EXPECTED_CONDITIONS:
        missing = sorted(EXPECTED_CONDITIONS - set(values))
        extra = sorted(set(values) - EXPECTED_CONDITIONS)
        raise ValueError(f"Expected the full 16-condition suite; missing={missing}, extra={extra}")
    non_finite = sorted(name for name, value in values.items() if not math.isfinite(value))
    if non_finite:
        raise ValueError(f"Non-finite AUC values: {non_finite}")
    return values


def per_source_aucs(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {source: {} for source in SOURCES}
    for name, metrics in payload["conditions"].items():
        by_dataset = metrics.get("by_dataset")
        if not isinstance(by_dataset, dict):
            raise ValueError(f"Condition {name!r} is missing by_dataset metrics")
        for source in SOURCES:
            entry = by_dataset.get(source)
            if entry is None or entry.get("roc_auc") is None:
                raise ValueError(f"Condition {name!r} is missing by_dataset AUC for {source!r}")
            auc = float(entry["roc_auc"])
            if not math.isfinite(auc):
                raise ValueError(f"Non-finite by_dataset AUC: {name!r} / {source!r}")
            values[source][name] = auc
    return values


def family_means(values: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for family in FAMILIES:
        family_values = [
            value for condition, value in values.items() if condition.startswith(family + "_")
        ]
        if not family_values:
            raise ValueError(f"Evaluation is missing the {family!r} family")
        result[family] = sum(family_values) / len(family_values)
    return result


def robust_score_from_condition_aucs(values: dict[str, float]) -> float:
    degraded = [values[name] for name in DEGRADED_CONDITIONS]
    mean_auc = sum(degraded) / len(degraded)
    worst_auc = min(degraded)
    return ROBUST_MEAN_WEIGHT * mean_auc + ROBUST_WORST_WEIGHT * worst_auc


def source_summaries(
    per_source: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for source, values in per_source.items():
        summaries[source] = {
            "robust_score": robust_score_from_condition_aucs(values),
            "family_means": family_means(values),
        }
    return summaries


def deltas(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: candidate[key] - baseline[key] for key in candidate}


def current_gates(
    candidate_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    def record(name: str, value: float, limit: float, kind: str) -> None:
        if kind == "absolute":
            passed = value >= limit
            margin = value - limit
        else:
            passed = value <= limit
            margin = limit - value
        checks[name] = {
            "kind": kind,
            "value": value,
            "limit": limit,
            "margin": margin,
            "pass": passed,
        }

    record(
        "overall_robust_score",
        candidate_summary["robust_score"],
        args.min_robust_score,
        "absolute",
    )
    record(
        "cf_robust_score",
        candidate_summary["sources"]["CommunityForensics-Small"]["robust_score"],
        args.min_cf_robust_score,
        "absolute",
    )
    source_robust_deltas = {
        source: candidate_summary["sources"][source]["robust_score"]
        - baseline_summary["sources"][source]["robust_score"]
        for source in SOURCES
    }
    for source in ("GenImage", "SID_Set"):
        record(
            f"{source}_robust_drop",
            -source_robust_deltas[source],
            args.max_source_robust_drop,
            "drop",
        )
    clean_delta = candidate_summary["condition_aucs"]["clean"] - baseline_summary["condition_aucs"]["clean"]
    record("clean_drop", -clean_delta, args.max_clean_drop, "drop")
    for family in FAMILIES:
        record(
            f"global_{family}_family_drop",
            -(candidate_summary["family_means"][family] - baseline_summary["family_means"][family]),
            args.max_global_family_drop,
            "drop",
        )
    for source in SOURCES:
        for family in FAMILIES:
            delta = (
                candidate_summary["sources"][source]["family_means"][family]
                - baseline_summary["sources"][source]["family_means"][family]
            )
            record(f"{source}_{family}_family_drop", -delta, args.max_source_family_drop, "drop")
    record(
        "noise_family_drop",
        -(candidate_summary["family_means"]["noise"] - baseline_summary["family_means"]["noise"]),
        args.max_noise_family_drop,
        "drop",
    )
    blur2_delta = (
        candidate_summary["condition_aucs"]["blur_2.0"] - baseline_summary["condition_aucs"]["blur_2.0"]
    )
    record("blur_2.0_drop", -blur2_delta, args.max_blur2_drop, "drop")

    failed = sorted(name for name, check in checks.items() if not check["pass"])
    return {
        "checks": checks,
        "failed": failed,
        "pass": not failed,
    }


def historical_gates(
    candidate_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """The original five-gate logic, kept verbatim as a historical reference.

    These values are reported only; they never gate acceptance in the current round.
    """
    robust_delta = candidate_summary["robust_score"] - baseline_summary["robust_score"]
    clean_delta = candidate_summary["condition_aucs"]["clean"] - baseline_summary["condition_aucs"]["clean"]
    blur2_delta = (
        candidate_summary["condition_aucs"]["blur_2.0"] - baseline_summary["condition_aucs"]["blur_2.0"]
    )
    family_deltas = deltas(candidate_summary["family_means"], baseline_summary["family_means"])
    gates = {
        "robust_gain": robust_delta >= args.min_robust_gain,
        "blur_2.0_guard": blur2_delta >= -args.max_blur2_drop,
        "clean_guard": clean_delta >= -args.max_clean_drop,
        "all_family_guards": all(
            delta >= -args.legacy_max_family_drop for delta in family_deltas.values()
        ),
        "noise_gain": family_deltas["noise"] >= args.min_noise_gain,
    }
    return {
        "note": "historical reference only; does not gate acceptance",
        "robust_delta": robust_delta,
        "clean_delta": clean_delta,
        "blur_2.0_delta": blur2_delta,
        "family_deltas": family_deltas,
        "gates": gates,
        "all_pass": all(gates.values()),
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    condition_aucs = aucs(payload)
    return {
        "epoch": payload.get("epoch"),
        "robust_score": float(payload["robustness"]["robust_score"]),
        "condition_aucs": condition_aucs,
        "family_means": family_means(condition_aucs),
        "sources": source_summaries(per_source_aucs(payload)),
    }


def compare(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    gates = current_gates(candidate_summary, baseline_summary, args)
    score = candidate_summary["robust_score"]
    return {
        "epoch": candidate_summary["epoch"],
        "robust_score": score,
        "robust_delta": score - baseline_summary["robust_score"],
        "source_robust_scores": {
            source: candidate_summary["sources"][source]["robust_score"] for source in SOURCES
        },
        "source_robust_deltas": {
            source: candidate_summary["sources"][source]["robust_score"]
            - baseline_summary["sources"][source]["robust_score"]
            for source in SOURCES
        },
        "clean_delta": candidate_summary["condition_aucs"]["clean"]
        - baseline_summary["condition_aucs"]["clean"],
        "gates": gates,
        "historical_reference": historical_gates(candidate_summary, baseline_summary, args),
        "accepted": gates["pass"] and math.isfinite(score),
    }


def render_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Robustness gate report")
    lines.append(f"baseline: {result['baseline']}")
    lines.append("")
    lines.append("[current-round bound gates] (pre-registered; gating)")
    for key, value in result["current_thresholds"].items():
        lines.append(f"  {key} = {value}")
    lines.append("")
    lines.append("[historical five gates] (reference only; never gates acceptance)")
    for key, value in result["historical_thresholds"].items():
        lines.append(f"  {key} = {value}")
    for item in result["comparisons"]:
        lines.append("")
        lines.append(f"candidate: {item['path']}")
        status = "ACCEPTED" if item["accepted"] else "REJECTED"
        lines.append(
            f"  verdict: {status}  epoch={item['epoch']}  robust={item['robust_score']:.6f} "
            f"(delta={item['robust_delta']:+.6f})"
        )
        for source in SOURCES:
            lines.append(
                f"    {source}: robust={item['source_robust_scores'][source]:.6f} "
                f"(delta={item['source_robust_deltas'][source]:+.6f})"
            )
        lines.append("  bound-gate checks:")
        for name, check in item["gates"]["checks"].items():
            mark = "PASS" if check["pass"] else "FAIL"
            kind = check["kind"]
            if kind == "absolute":
                detail = f"value={check['value']:.6f} >= {check['limit']:.6f}"
            else:
                detail = f"drop={check['value']:.6f} <= {check['limit']:.6f}"
            lines.append(f"    [{mark}] {name}: {detail} (margin={check['margin']:+.6f})")
        lines.append("  historical reference gates (not gating):")
        history = item["historical_reference"]
        for name, passed in history["gates"].items():
            mark = "PASS" if passed else "FAIL"
            lines.append(f"    [{mark}] {name}")
    lines.append("")
    selected = result["selected"]
    lines.append(f"selected={selected['path'] if selected else 'NONE'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baseline = load_metrics(args.baseline)
    baseline_summary = summarize(baseline)
    comparisons = []
    for path in args.candidate:
        candidate = load_metrics(path)
        comparison = compare(baseline_summary, summarize(candidate), args)
        comparison["path"] = candidate["_path"]
        comparisons.append(comparison)
    accepted = [item for item in comparisons if item["accepted"]]
    accepted.sort(key=lambda item: item["robust_score"], reverse=True)
    result = {
        "baseline": baseline["_path"],
        "baseline_summary": {
            "robust_score": baseline_summary["robust_score"],
            "clean_auc": baseline_summary["condition_aucs"]["clean"],
            "source_robust_scores": {
                source: baseline_summary["sources"][source]["robust_score"] for source in SOURCES
            },
        },
        "current_thresholds": {
            "min_robust_score": args.min_robust_score,
            "min_cf_robust_score": args.min_cf_robust_score,
            "max_source_robust_drop": args.max_source_robust_drop,
            "max_clean_drop": args.max_clean_drop,
            "max_global_family_drop": args.max_global_family_drop,
            "max_source_family_drop": args.max_source_family_drop,
            "max_noise_family_drop": args.max_noise_family_drop,
            "max_blur2_drop": args.max_blur2_drop,
        },
        "historical_thresholds": {
            "note": "historical reference only; does not gate acceptance",
            "min_robust_gain": args.min_robust_gain,
            "max_blur2_drop": args.max_blur2_drop,
            "max_clean_drop": args.max_clean_drop,
            "max_family_drop": args.legacy_max_family_drop,
            "min_noise_gain": args.min_noise_gain,
        },
        "comparisons": comparisons,
        "selected": accepted[0] if accepted else None,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_text = render_report(result)
    report_path = output.with_suffix(".txt")
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    print(f"report_json={output}")
    print(f"report_text={report_path}")


if __name__ == "__main__":
    main()
