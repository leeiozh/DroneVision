"""Basic photo adjustments shared by the browser preview and the projections.

The pipeline is defined on sRGB values in [0, 1] and matches the SVG filter built by
``web_static/tools.js`` (``color-interpolation-filters="sRGB"``) step for step:

1. per-channel linear stage, clamped:  x' = contrast * (x * gain_c * 2**exposure + brightness - 0.5) + 0.5,
   where gain_R = 1 + 0.25 * temperature, gain_B = 1 - 0.25 * temperature, gain_G = 1;
2. gamma, clamped:                      x' = x ** (1 / gamma)   (gamma > 1 brightens mid-tones);
3. saturation (feColorMatrix "saturate" coefficients 0.213/0.715/0.072), clamped.

Every step is per pixel, so applying it after warping is equivalent to applying it to the frame.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {"exposure": 0.0, "brightness": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0, "temperature": 0.0}
LIMITS = {"exposure": (-3.0, 3.0), "brightness": (-0.5, 0.5), "contrast": (0.2, 3.0), "gamma": (0.3, 3.0),
          "saturation": (0.0, 3.0), "temperature": (-1.0, 1.0)}


def normalize(params: dict | None) -> dict:
    """Clip known keys to their limits and fill in the neutral defaults."""
    result = dict(DEFAULTS)
    for key, value in (params or {}).items():
        if key in DEFAULTS and value is not None:
            low, high = LIMITS[key]
            result[key] = float(np.clip(float(value), low, high))
    return result


def is_neutral(params: dict | None) -> bool:
    params = normalize(params)
    return all(abs(params[key] - DEFAULTS[key]) < 1e-6 for key in DEFAULTS)


def describe(params: dict | None) -> str:
    params = normalize(params)
    if is_neutral(params):
        return "без коррекции"
    return ", ".join(f"{key} {params[key]:+.2f}" if DEFAULTS[key] == 0 else f"{key} ×{params[key]:.2f}"
                     for key in DEFAULTS if abs(params[key] - DEFAULTS[key]) >= 1e-6)


def _saturation_matrix_rgb(s: float) -> np.ndarray:
    return np.array([[0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
                     [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
                     [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s]], dtype=np.float64)


def apply_adjustments(image_bgr: np.ndarray, params: dict | None) -> np.ndarray:
    """Return an adjusted copy of a uint8 BGR (or BGRA, alpha untouched) image."""
    params = normalize(params)
    if is_neutral(params):
        return image_bgr.copy()
    import cv2

    x = np.arange(256, dtype=np.float64) / 255.0
    gains = {"b": 1 - 0.25 * params["temperature"], "g": 1.0, "r": 1 + 0.25 * params["temperature"]}
    luts = []
    for channel in "bgr":
        linear = params["contrast"] * (x * gains[channel] * 2 ** params["exposure"] + params["brightness"] - 0.5) + 0.5
        curved = np.clip(linear, 0, 1) ** (1 / params["gamma"])
        luts.append(np.round(np.clip(curved, 0, 1) * 255).astype(np.uint8))
    colour = image_bgr[..., :3]
    result = cv2.merge([cv2.LUT(np.ascontiguousarray(colour[..., i]), luts[i]) for i in range(3)])
    if abs(params["saturation"] - 1) >= 1e-6:
        # cv2.transform keeps uint8 and saturates, i.e. clamps like the SVG filter.
        matrix = _saturation_matrix_rgb(params["saturation"])[::-1, ::-1]  # RGB -> BGR order
        result = cv2.transform(result, matrix)
    if image_bgr.shape[-1] == 4:
        result = np.dstack([result, image_bgr[..., 3]])
    return result
