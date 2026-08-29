from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapter import build_checkpoint_model
from src.data import read_manifest
from src.transforms import IMAGENET_MEAN, IMAGENET_STD


VIEWS = ("decoded", "jpeg_q75", "hash_random_reencode", "neutralized_random_reencode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit label/format association and compression-history sensitivity using internal data only."
    )
    parser.add_argument("--dataset-root", default="Dataset")
    parser.add_argument("--train-manifest", default="Dataset/manifests/training_multisource.csv")
    parser.add_argument("--val-manifest", default="Dataset/manifests/validation_multisource.csv")
    parser.add_argument("--checkpoint", default="outputs/multisource_blur_finetune/best.pt")
    parser.add_argument("--output-dir", default="reports/compression_history_audit")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--format-probe-per-stratum", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _extension(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".jpe"}:
        return "jpeg"
    return suffix.removeprefix(".") or "none"


def _cramers_v(table: pd.DataFrame) -> float:
    observed = table.to_numpy(dtype=np.float64)
    total = observed.sum()
    if total == 0:
        return math.nan
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / total
    mask = expected > 0
    chi2 = float(np.sum(((observed - expected) ** 2)[mask] / expected[mask]))
    rows, columns = observed.shape
    denominator = total * min(rows - 1, columns - 1)
    return math.sqrt(chi2 / denominator) if denominator > 0 else math.nan


def _sample_by_stratum(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    strata = list(frame.groupby(["dataset", "source_class"], sort=True, dropna=False))
    if limit <= 0 or len(frame) <= limit:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    base = limit // len(strata)
    pieces: list[pd.DataFrame] = []
    used: set[int] = set()
    for offset, (_, group) in enumerate(strata):
        take = min(base, len(group))
        part = group.sample(n=take, random_state=seed + offset)
        pieces.append(part)
        used.update(part.index.tolist())
    remainder = limit - sum(len(piece) for piece in pieces)
    if remainder:
        pool = frame.loc[~frame.index.isin(used)]
        pieces.append(pool.sample(n=min(remainder, len(pool)), random_state=seed + 999))
    return pd.concat(pieces).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _roundtrip(image: Image.Image, codec: str, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    options: dict[str, Any] = {"format": codec, "quality": quality}
    if codec == "JPEG":
        options["subsampling"] = 2
    image.save(buffer, **options)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _hash_assignment(relative_path: str, seed: int) -> tuple[str, int, int]:
    digest = hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).digest()
    codec = ("JPEG", "WEBP")[digest[0] % 2]
    quality = (50, 65, 80, 95)[digest[1] % 4]
    noise_seed = int.from_bytes(digest[2:10], "little", signed=False)
    return codec, quality, noise_seed


def _make_view(image: Image.Image, relative_path: str, view: str, seed: int) -> Image.Image:
    image = image.convert("RGB")
    if view == "decoded":
        return image
    if view == "jpeg_q75":
        return _roundtrip(image, "JPEG", 75)
    codec, quality, noise_seed = _hash_assignment(relative_path, seed)
    if view == "hash_random_reencode":
        return _roundtrip(image, codec, quality)
    if view == "neutralized_random_reencode":
        # Sensitivity stress test, not a perfect causal removal of compression history:
        # mild resampling and one-LSB noise reduce residual codec fingerprints before
        # a label-blind, path-hash-assigned re-encoding.
        width, height = image.size
        reduced = image.resize(
            (max(1, round(width * 0.875)), max(1, round(height * 0.875))), Image.Resampling.BICUBIC
        )
        image = reduced.resize((width, height), Image.Resampling.BICUBIC)
        rng = np.random.default_rng(noise_seed)
        pixels = np.asarray(image, dtype=np.int16)
        noise = rng.integers(-1, 2, size=pixels.shape, dtype=np.int16)
        image = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8), mode="RGB")
        return _roundtrip(image, codec, quality)
    raise ValueError(f"Unknown view: {view}")


class AuditDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: Path, frame: pd.DataFrame, image_size: int, seed: int) -> None:
        self.root = root
        self.frame = frame.reset_index(drop=True)
        self.seed = seed
        resize_size = round(image_size / 0.875)
        self.preprocess = T.Compose(
            [
                T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        relative_path = str(row["path"])
        with Image.open(self.root / Path(relative_path)) as handle:
            image = handle.convert("RGB")
            images = {
                view: self.preprocess(_make_view(image.copy(), relative_path, view, self.seed))
                for view in VIEWS
            }
        codec, quality, _ = _hash_assignment(relative_path, self.seed)
        return {
            "images": images,
            "path": relative_path,
            "label": int(row["binary_label"]),
            "dataset": str(row["dataset"]),
            "source_class": str(row["source_class"]),
            "assigned_codec": codec,
            "assigned_quality": quality,
        }


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else math.nan


def _metrics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_specs: list[tuple[str, pd.Series]] = [("overall", pd.Series(True, index=rows.index))]
    for dataset in sorted(rows["dataset"].unique()):
        group_specs.append((f"dataset:{dataset}", rows["dataset"].eq(dataset)))
    for source in sorted(rows.loc[rows["label"].eq(1), "source_class"].unique()):
        group_specs.append(
            (f"real_vs:{source}", rows["label"].eq(0) | rows["source_class"].eq(source))
        )
    for view in VIEWS:
        for group, mask in group_specs:
            part = rows.loc[mask]
            labels = part["label"].to_numpy(dtype=np.int64)
            scores = part[view].to_numpy(dtype=np.float64)
            records.append(
                {
                    "view": view,
                    "group": group,
                    "count": len(part),
                    "roc_auc": _auc(labels, scores),
                    "average_precision": (
                        float(average_precision_score(labels, scores))
                        if len(np.unique(labels)) == 2
                        else math.nan
                    ),
                    "mean_score_real": float(scores[labels == 0].mean()) if np.any(labels == 0) else math.nan,
                    "mean_score_fake": float(scores[labels == 1].mean()) if np.any(labels == 1) else math.nan,
                }
            )
    return pd.DataFrame(records)


def _paired_bootstrap(rows: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    labels = rows["label"].to_numpy(dtype=np.int64)
    clean = rows["decoded"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for view in VIEWS[1:]:
        scores = rows[view].to_numpy(dtype=np.float64)
        deltas: list[float] = []
        for _ in range(repeats):
            indices = rng.integers(0, len(rows), len(rows))
            sampled_labels = labels[indices]
            if len(np.unique(sampled_labels)) != 2:
                continue
            deltas.append(_auc(sampled_labels, scores[indices]) - _auc(sampled_labels, clean[indices]))
        values = np.asarray(deltas)
        output.append(
            {
                "view": view,
                "auc_delta_vs_decoded": _auc(labels, scores) - _auc(labels, clean),
                "bootstrap_delta_ci_low": float(np.quantile(values, 0.025)),
                "bootstrap_delta_ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_repeats": len(values),
            }
        )
    return pd.DataFrame(output)


def _probe_actual_formats(
    frame: pd.DataFrame, root: Path, per_stratum: int, seed: int
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for offset, ((dataset, source), group) in enumerate(
        frame.groupby(["dataset", "source_class"], sort=True)
    ):
        sample = group.sample(n=min(per_stratum, len(group)), random_state=seed + offset)
        for _, row in sample.iterrows():
            path = str(row["path"])
            try:
                with Image.open(root / Path(path)) as image:
                    actual_format = str(image.format or "unknown").lower()
            except Exception:
                actual_format = "read_error"
            records.append(
                {
                    "dataset": dataset,
                    "source_class": source,
                    "binary_label": int(row["binary_label"]),
                    "extension": _extension(path),
                    "pil_format": actual_format,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifests = {
        "train": (Path(args.train_manifest), read_manifest(args.train_manifest)),
        "validation": (Path(args.val_manifest), read_manifest(args.val_manifest)),
    }
    format_rows: list[pd.DataFrame] = []
    associations: list[dict[str, Any]] = []
    probes: list[pd.DataFrame] = []
    for split, (path, frame) in manifests.items():
        audit = frame[["dataset", "source_class", "binary_label", "path"]].copy()
        audit["extension"] = audit["path"].map(_extension)
        grouped = (
            audit.groupby(["dataset", "source_class", "binary_label", "extension"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        grouped.insert(0, "split", split)
        format_rows.append(grouped)
        table = pd.crosstab(audit["extension"], audit["binary_label"])
        associations.append(
            {
                "split": split,
                "manifest": str(path.resolve()),
                "rows": len(audit),
                "cramers_v_extension_vs_label": _cramers_v(table),
                "jpeg_share_real": float(audit.loc[audit.binary_label.eq(0), "extension"].eq("jpeg").mean()),
                "jpeg_share_fake": float(audit.loc[audit.binary_label.eq(1), "extension"].eq("jpeg").mean()),
            }
        )
        probe = _probe_actual_formats(frame, root, args.format_probe_per_stratum, args.seed)
        probe.insert(0, "split", split)
        pil_table = pd.crosstab(probe["pil_format"], probe["binary_label"])
        associations[-1].update(
            {
                "pil_probe_rows": len(probe),
                "sampled_cramers_v_pil_format_vs_label": _cramers_v(pil_table),
                "sampled_pil_jpeg_share_real": float(
                    probe.loc[probe.binary_label.eq(0), "pil_format"].eq("jpeg").mean()
                ),
                "sampled_pil_jpeg_share_fake": float(
                    probe.loc[probe.binary_label.eq(1), "pil_format"].eq("jpeg").mean()
                ),
            }
        )
        probes.append(probe)

    pd.concat(format_rows, ignore_index=True).to_csv(output / "format_label_counts.csv", index=False)
    pd.DataFrame(associations).to_csv(output / "format_label_association.csv", index=False)
    probe_counts = (
        pd.concat(probes, ignore_index=True)
        .groupby(["split", "dataset", "source_class", "binary_label", "extension", "pil_format"])
        .size()
        .reset_index(name="count")
    )
    probe_counts.to_csv(output / "sampled_pil_format_counts.csv", index=False)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    image_size = int(config["data"]["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Legacy checkpoints (no adapter config) load as the bare base exactly as
    # before; adapter-enabled checkpoints build the wrapped model.
    model = build_checkpoint_model(config, checkpoint["model_state"])
    model.eval().to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    sampled = _sample_by_stratum(manifests["validation"][1], args.sample_size, args.seed)
    sampled[["path", "dataset", "source_class", "binary_label"]].to_csv(
        output / "diagnostic_sample_manifest.csv", index=False
    )
    dataset = AuditDataset(root, sampled, image_size, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch_scores: dict[str, np.ndarray] = {}
            for view in VIEWS:
                images = batch["images"][view].to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(images).flatten()
                batch_scores[view] = torch.sigmoid(logits).float().cpu().numpy()
            for index, relative_path in enumerate(batch["path"]):
                record: dict[str, Any] = {
                    "path": relative_path,
                    "dataset": batch["dataset"][index],
                    "source_class": batch["source_class"][index],
                    "label": int(batch["label"][index]),
                    "assigned_codec": batch["assigned_codec"][index],
                    "assigned_quality": int(batch["assigned_quality"][index]),
                }
                record.update({view: float(batch_scores[view][index]) for view in VIEWS})
                predictions.append(record)
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"diagnostic {batch_index}/{len(loader)}")

    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(output / "diagnostic_predictions.csv", index=False)
    metrics = _metrics(prediction_frame)
    metrics.to_csv(output / "diagnostic_metrics.csv", index=False)
    bootstrap = _paired_bootstrap(prediction_frame, args.bootstrap_repeats, args.seed)
    bootstrap.to_csv(output / "paired_auc_deltas.csv", index=False)
    assignment_balance = (
        prediction_frame.groupby(["label", "assigned_codec", "assigned_quality"])
        .size()
        .reset_index(name="count")
    )
    assignment_balance.to_csv(output / "reencode_assignment_balance.csv", index=False)

    overall = metrics.loc[metrics.group.eq("overall")].set_index("view")
    association_frame = pd.DataFrame(associations).set_index("split")
    summary = {
        "checkpoint": str(checkpoint_path),
        "internal_manifest": str(manifests["validation"][0].resolve()),
        "sample_size": len(prediction_frame),
        "seed": args.seed,
        "device": str(device),
        "format_association": associations,
        "overall_auc": {view: float(overall.loc[view, "roc_auc"]) for view in VIEWS},
        "auc_delta_vs_decoded": {
            row["view"]: float(row["auc_delta_vs_decoded"])
            for row in bootstrap.to_dict(orient="records")
        },
    }
    (output / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Compression-history shortcut audit",
        "",
        "## Scope and protocol",
        "",
        f"- Internal data only; no WildFake rows were read. Diagnostic manifest: `{summary['internal_manifest']}`.",
        f"- Frozen checkpoint: `{checkpoint_path}`; paired stratified sample: {len(prediction_frame):,} images; seed: {args.seed}.",
        "- `decoded`: ordinary decoded input.",
        "- `jpeg_q75`: every decoded image is saved once more as JPEG quality 75.",
        "- `hash_random_reencode`: codec (JPEG/WebP) and quality (50/65/80/95) are assigned only by a seeded SHA-256 hash of the relative path, never by label.",
        "- `neutralized_random_reencode`: adds mild resize round-trip and one-LSB deterministic noise before the same label-blind re-encoding. This is a sensitivity stress test, not a pure causal intervention.",
        "",
        "## Static association",
        "",
        "| Split | Rows | JPEG suffix real | JPEG suffix fake | V(suffix,label) | PIL probe rows | PIL JPEG real | PIL JPEG fake | V(PIL format,label) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in association_frame.iterrows():
        lines.append(
            f"| {split} | {int(row['rows']):,} | {row['jpeg_share_real']:.4f} | "
            f"{row['jpeg_share_fake']:.4f} | {row['cramers_v_extension_vs_label']:.4f} | "
            f"{int(row['pil_probe_rows']):,} | {row['sampled_pil_jpeg_share_real']:.4f} | "
            f"{row['sampled_pil_jpeg_share_fake']:.4f} | "
            f"{row['sampled_cramers_v_pil_format_vs_label']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-model paired diagnostic",
            "",
            "| View | ROC AUC | AUC delta vs decoded | 95% paired-bootstrap interval | Mean score real | Mean score fake |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    bootstrap_index = bootstrap.set_index("view")
    for view in VIEWS:
        row = overall.loc[view]
        if view == "decoded":
            delta, interval = 0.0, "reference"
        else:
            delta_row = bootstrap_index.loc[view]
            delta = float(delta_row["auc_delta_vs_decoded"])
            interval = (
                f"[{delta_row['bootstrap_delta_ci_low']:.4f}, "
                f"{delta_row['bootstrap_delta_ci_high']:.4f}]"
            )
        lines.append(
            f"| {view} | {row['roc_auc']:.4f} | {delta:+.4f} | {interval} | "
            f"{row['mean_score_real']:.4f} | {row['mean_score_fake']:.4f} |"
        )
    random_delta = float(bootstrap_index.loc["hash_random_reencode", "auc_delta_vs_decoded"])
    neutral_delta = float(
        bootstrap_index.loc["neutralized_random_reencode", "auc_delta_vs_decoded"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Filename suffix is perfectly associated with label here, but suffix is not the same as the decoded container: most sampled SID tampered files have a `.png` name while PIL identifies JPEG bytes. The sampled PIL-format association is therefore the more relevant static signal and remains strong but not perfect. A model cannot read the suffix directly; it can exploit codec residue correlated with the decoded history.",
            f"- Label-blind random re-encoding changed AUC by {random_delta:+.4f}; the stronger neutralization stress test changed it by {neutral_delta:+.4f}. Large negative shifts support sensitivity to codec/resampling history, but do not prove that all lost signal was a shortcut because valid forensic cues are also perturbed.",
            "- A single static JPEG conversion is not a remedy: original JPEG real images become effectively double-compressed, while original PNG fake images are typically compressed once. The original history can therefore survive uniform JPEG re-encoding.",
            "- Recommended mitigation, if used in training, is label-independent randomized codec/quality assignment with exposure balance verified by label, while retaining a no-extra-reencoding branch. Evaluate it only on internal selection/confirmation splits before freezing the candidate.",
            "",
            "## Reproducibility",
            "",
            "```powershell",
            "conda run -n jam python scripts/audit_compression_history.py",
            "```",
            "",
            "Detailed tables and paired predictions are stored beside this report.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output / 'report.md'}")


if __name__ == "__main__":
    main()
