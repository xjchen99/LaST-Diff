"""Prepare paired CAMUS half sequences as 16-frame NumPy datasets."""
import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

def read_ids(path):
    identifiers = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate patient IDs in %s" % path)
    if any(not item.startswith("patient") or "/" in item or "\\" in item for item in identifiers):
        raise ValueError("Unexpected patient ID in %s" % path)
    return identifiers


def resample_sequence(volume, frames, is_label=False):
    """Preserve the research pipeline's rounded sampling / hybrid interpolation."""
    count, height, width = volume.shape
    if count == frames:
        return volume
    if count > frames:
        indices = np.round(np.linspace(0, count - 1, frames)).astype(np.int32)
        return volume[np.clip(indices, 0, count - 1)]
    image = sitk.GetImageFromArray(volume)
    spacing = image.GetSpacing()
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    result = sitk.Resample(
        image, [width, height, frames], sitk.Transform(), interpolator,
        image.GetOrigin(), [spacing[0], spacing[1], spacing[2] * count / frames],
        image.GetDirection(), 0.0, image.GetPixelID(),
    )
    return sitk.GetArrayFromImage(result)


def normalize(volume):
    volume = volume.astype(np.float32)
    lo, hi = volume.min(), volume.max()
    return 2.0 * (volume - lo) / (hi - lo) - 1.0 if hi - lo > 1e-8 else np.full_like(volume, -1.0)


def resize(volume, size, is_label=False):
    import cv2
    interpolation = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    return np.stack([cv2.resize(frame, (size, size), interpolation=interpolation)
                     for frame in volume])


def save_frames(volume, directory, is_label=False):
    directory.mkdir(parents=True, exist_ok=False)
    dtype = np.uint8 if is_label else np.float32
    for index, frame in enumerate(volume):
        np.save(directory / ("timeidx=%d.npy" % index), frame[None].astype(dtype))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing patientXXXX folders.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True,
                        help="CAMUS directory containing subgroup_training/validation/testing.txt.")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--unnormalized", action="store_true", help="Keep image intensities for baseline preparation.")
    parser.add_argument("--images-only", action="store_true", help="Prepare images for inference without annotations.")
    args = parser.parse_args()
    if args.frames < 1 or args.size < 1:
        parser.error("Frames and size must be positive.")
    train = read_ids(args.split_dir / "subgroup_training.txt")
    validation = read_ids(args.split_dir / "subgroup_validation.txt")
    test = read_ids(args.split_dir / "subgroup_testing.txt")
    if set(train) & set(validation) or (set(train) | set(validation)) & set(test):
        raise ValueError("Patient split overlap.")
    splits = {"train": train + validation, "test": test}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory; preprocessing does not overwrite data.")
    jobs = []
    for split, patients in splits.items():
        for patient in patients:
            for view in ("2CH", "4CH"):
                stem = "%s_%s_half_sequence" % (patient, view)
                image = args.raw_root / patient / (stem + ".nii.gz")
                label = args.raw_root / patient / (stem + "_gt.nii.gz")
                if not image.is_file() or (not args.images_only and not label.is_file()):
                    raise FileNotFoundError("Missing paired CAMUS half sequence: %s" % stem)
                jobs.append((split, patient, view, image, label))
    for split, patient, view, image_path, label_path in jobs:
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        if image.ndim != 3 or not np.isfinite(image).all():
            raise ValueError("Expected a finite [frames, height, width] image: %s" % image_path)
        label = None
        if not args.images_only:
            label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
            if label.shape != image.shape or not np.isin(label, [0, 1, 2, 3]).all():
                raise ValueError("Invalid CAMUS label sequence: %s" % label_path)
        image = resample_sequence(image, args.frames)
        if not args.unnormalized:
            image = normalize(image)
        folder = Path(split) / patient / ("%s_half_sequence" % view)
        save_frames(resize(image, args.size), args.output_dir / "images" / folder)
        if label is not None:
            label = resample_sequence(label.astype(np.uint8), args.frames, True)
            save_frames(resize(label, args.size, True), args.output_dir / "labels" / folder, True)
    record = {"raw_root": str(args.raw_root.resolve()), "frames": args.frames, "size": args.size,
              "normalized": not args.unnormalized, "images_only": args.images_only,
              "training_includes_validation": True, "splits": splits}
    (args.output_dir / "preprocessing.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Prepared %d view sequences in %s" % (len(jobs), args.output_dir))


if __name__ == "__main__":
    main()
