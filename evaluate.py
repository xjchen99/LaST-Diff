"""Evaluate saved CAMUS sequences against NIfTI or prepared NumPy labels."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from evaluation.metrics import dice_scores, temporal_metrics, validate_labels
from utils.sequences import discover_sequences, load_frame


def canonical_name(name):
    for suffix in ("_half_sequence_pd.nii.gz", "_half_sequence.nii.gz", "_pd.nii.gz", ".nii.gz"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def index_nifti(directory):
    result = {}
    for path in sorted(Path(directory).glob("*.nii.gz")):
        name = canonical_name(path.name)
        if name in result:
            raise ValueError("Duplicate case ID: %s" % name)
        result[name] = path
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gt-dir", type=Path)
    group.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--case-list", type=Path, help="Optional external list of patient/view IDs, e.g. the CAMUS test_poor.txt list.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    predictions = index_nifti(args.pred_dir)
    if args.gt_dir:
        targets = index_nifti(args.gt_dir)
    else:
        targets = {name: labels for name, _, labels in discover_sequences(
            args.data_root, split=args.split, frames=args.frames, require_labels=True)}
    if args.case_list:
        cases = [canonical_name(s.strip()) for s in args.case_list.read_text().splitlines() if s.strip()]
        if len(cases) != len(set(cases)):
            raise ValueError("Duplicate IDs in case list.")
    else:
        if set(predictions) != set(targets):
            raise ValueError("Prediction and GT case sets differ; use --case-list for an explicit subset.")
        cases = sorted(targets)
    if not cases or any(name not in targets or name not in predictions for name in cases):
        raise ValueError("Empty selection or missing prediction/GT cases.")
    files = ["per_case.csv", "prediction_temporal.csv", "gt_temporal.csv", "summary.json"]
    if not args.overwrite and any((args.output_dir / name).exists() for name in files):
        raise FileExistsError("Evaluation output exists; choose another directory or --overwrite.")
    rows, pred_rows, gt_rows = [], [], []
    for name in cases:
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(str(predictions[name])))
        target = (sitk.GetArrayFromImage(sitk.ReadImage(str(targets[name]))) if args.gt_dir else
                  np.stack([load_frame(p, args.size, True) for p in targets[name]]))
        validate_labels(prediction)
        validate_labels(target)
        if prediction.shape != target.shape or prediction.shape[0] != args.frames:
            raise ValueError("Shape/frame mismatch for %s" % name)
        pred_metrics = temporal_metrics(prediction)
        gt_metrics = temporal_metrics(target, ground_truth=True)
        filename = name + ".nii.gz"
        row = {"filename": filename}
        row.update({"dice_L%d" % label: value for label, value in dice_scores(prediction, target).items()})
        for key in pred_metrics:
            row["error_" + key] = abs(pred_metrics[key] - gt_metrics[key])
        rows.append(row)
        pred_rows.append(dict(filename=filename, **pred_metrics))
        converted = {}
        for key, value in gt_metrics.items():
            prefix, label = key.split("_")
            converted[{"tdice": "tdsc", "smooth": "tcs", "jitter": "acd"}[prefix] + "_" + label] = value
        gt_rows.append(dict(filename=filename, **converted))
    summary = {"n_sequences": len(rows), "evaluation_protocol": "historical", "distance_units": "pixels",
               "aggregation": "mean across view sequences; DSC pools all frames within each sequence",
               "metrics": {}}
    for key in rows[0]:
        if key == "filename":
            continue
        values = np.array([row[key] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        summary["metrics"][key] = {"n": len(values), "mean": float(values.mean()) if len(values) else None,
                                   "std": float(values.std()) if len(values) else None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / files[0], rows)
    write_csv(args.output_dir / files[1], pred_rows)
    write_csv(args.output_dir / files[2], gt_rows)
    (args.output_dir / files[3]).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
