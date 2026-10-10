from __future__ import annotations

import math

import numpy as np


def nice_grid_step(span_m: float, target_divisions: int = 10) -> float:
    """Return a round 1/2/5×10ⁿ step giving roughly ``target_divisions`` cells."""
    target = max(float(span_m) / max(int(target_divisions), 1), 1e-6)
    power = 10.0 ** math.floor(math.log10(target))
    fraction = target / power
    multiplier = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return multiplier * power


def _label(cv2, image: np.ndarray, text: str, origin: tuple[int, int],
           font_scale: float, thickness: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    height, width = image.shape[:2]
    unit = max(min(height, width) / 1000.0, 1.0)
    pad = max(3, int(round(4 * unit)))
    x = min(max(origin[0], pad), max(width - text_w - 2 * pad, pad))
    y = min(max(origin[1], text_h + pad), max(height - baseline - pad, text_h + pad))
    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def overlay_metric_grid(cv2, image: np.ndarray, minimum: np.ndarray, maximum: np.ndarray,
                        resolution_m_per_px: float, step_m: float | None = None) -> tuple[np.ndarray, float]:
    """Overlay a north-up ENU metre grid on a BGR image and return it with the chosen step."""
    result = image.copy()
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    span = maximum - minimum
    height, width = result.shape[:2]
    resolution = float(resolution_m_per_px)
    # About 14 cells on the short side, but never more than 30 on a very long side.
    # Thus density follows the image proportions instead of an absolute metre threshold.
    target_cell_m = max(float(span.min()) / 14.0, float(span.max()) / 30.0)
    step = float(step_m or nice_grid_step(target_cell_m, 1))

    eastings = np.arange(math.ceil(minimum[0] / step) * step, maximum[0] + step * 1e-6, step)
    northings = np.arange(math.ceil(minimum[1] / step) * step, maximum[1] + step * 1e-6, step)
    vertical = [int(round((east - minimum[0]) / resolution)) for east in eastings]
    horizontal = [int(round((maximum[1] - north) / resolution)) for north in northings]

    relative_size = min(height, width)
    # The exported grid is deliberately restrained: white only and half the former
    # relative stroke width. The interactive canvas draws its own dynamic grid.
    line_width = max(1, int(round(relative_size / 1800)))
    for x in vertical:
        if 0 <= x < width:
            cv2.line(result, (x, 0), (x, height - 1), (255, 255, 255), line_width, cv2.LINE_AA)
    for y in horizontal:
        if 0 <= y < height:
            cv2.line(result, (0, y), (width - 1, y), (255, 255, 255), line_width, cv2.LINE_AA)

    font_scale = max(0.55, relative_size / 900.0)
    text_thickness = max(1, int(round(relative_size / 1400.0)))
    label_offset = max(5, int(round(relative_size / 180.0)))
    cell_px = step / resolution
    font = cv2.FONT_HERSHEY_SIMPLEX
    east_label_width = max((cv2.getTextSize(f"E {int(round(value))} m", font, font_scale,
                                           text_thickness)[0][0] for value in eastings), default=0)
    north_label_height = max((cv2.getTextSize(f"N {int(round(value))} m", font, font_scale,
                                              text_thickness)[0][1] for value in northings), default=0)
    east_stride = max(1, int(math.ceil((east_label_width + 2 * label_offset) / max(cell_px, 1))))
    north_stride = max(1, int(math.ceil((north_label_height + 2 * label_offset) / max(cell_px, 1))))
    for index, (east, x) in enumerate(zip(eastings, vertical)):
        if index % east_stride:
            continue
        if 0 <= x < width:
            _label(cv2, result, f"E {int(round(east))} m", (x + label_offset, label_offset),
                   font_scale, text_thickness)
    top_clearance = north_label_height + 3 * label_offset
    for index, (north, y) in enumerate(zip(northings, horizontal)):
        if index % north_stride:
            continue
        if 0 <= y < height:
            if y < top_clearance:
                continue
            _label(cv2, result, f"N {int(round(north))} m", (label_offset, y - label_offset),
                   font_scale, text_thickness)
    return result, step
