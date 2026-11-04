"""Select only SAM masks that honor explicit positive/negative points."""
import numpy as np

MASK_POLICY = "prompt-consistent-v2"


def select_mask(masks, scores, points=(), labels=()):
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError("SAM returned an invalid mask.")
    height, width = masks.shape[1:]
    valid = np.isfinite(scores)
    for (x, y), label in zip(points, labels):
        px = min(width - 1, max(0, int(x * width)))
        py = min(height - 1, max(0, int(y * height)))
        valid &= masks[:, py, px] == bool(label)
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("SAM did not satisfy the keep/remove points. Add points, redraw the box, or edit the mask.")
    selected = int(indices[np.argmax(scores[indices])])
    return masks[selected].astype("uint8") * 255
