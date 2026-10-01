"""Historical CAMUS evaluation definitions used by this release."""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def validate_labels(volume):
    if volume.ndim != 3 or volume.shape[0] < 3:
        raise ValueError("Expected a [frames, height, width] label sequence with at least three frames.")
    if not np.isin(volume, [0, 1, 2, 3]).all():
        raise ValueError("Expected CAMUS labels 0, 1, 2, 3.")


def dice_scores(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Prediction/GT shape mismatch.")
    scores = {}
    for label in (1, 2, 3):
        p, g = prediction == label, target == label
        denominator = p.sum() + g.sum()
        scores[label] = float(2.0 * (p & g).sum() / denominator) if denominator else 1.0
    return scores


def _boundary_and_distance(mask, ground_truth):
    if ground_truth:
        padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
        eroded = np.ones_like(mask, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
        boundary = mask.astype(bool) & ~eroded
        image = sitk.GetImageFromArray(boundary.astype(np.uint8))
        distance = sitk.GetArrayFromImage(sitk.DanielssonDistanceMap(
            image, inputIsBinary=True, squaredDistance=False, useImageSpacing=False))
    else:
        boundary = mask.astype(np.uint8) - ndimage.binary_erosion(mask)
        distance = ndimage.distance_transform_edt(1 - boundary)
        boundary = boundary.astype(bool)
    return boundary, distance


def temporal_metrics(volume, ground_truth=False):
    """Compute the fixed historical protocol for predictions or ground truth.

    Predictions use SciPy ACD; GT uses Danielsson ACD. L2 temporal Dice
    and ACD use labels 1+2. Prediction L2 TCS uses label 2; GT uses 1+2.
    """
    validate_labels(volume)
    result = {}
    for label in (1, 2, 3):
        masks = volume == label
        # Saved tables used the combined LV region for L2 tDSC/ACD.
        if label == 2:
            masks = np.isin(volume, [1, 2])
        adjacent = []
        for left, right in zip(masks[:-1], masks[1:]):
            denominator = left.sum() + right.sum()
            adjacent.append(2.0 * (left & right).sum() / denominator if denominator else 1.0)
        # Historical prediction TCS alone used myocardium, while GT used LV union.
        area_masks = (volume == label) if not ground_truth else masks
        areas = np.array([mask.sum() for mask in area_masks], dtype=np.float32)
        smooth = float(np.abs(np.diff(areas / areas.max(), n=2)).mean()) if areas.max() else 0.0
        boundaries = [_boundary_and_distance(mask, ground_truth) if mask.any() else None for mask in masks]
        distances = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if left is not None and right is not None:
                b1, d1 = left
                b2, d2 = right
                distances.append((float(np.mean(d2[b1])) + float(np.mean(d1[b2]))) / 2.0)
        result["tdice_L%d" % label] = float(np.mean(adjacent))
        result["smooth_L%d" % label] = smooth
        result["jitter_L%d" % label] = float(np.mean(distances)) if distances else float("nan")
    return result
