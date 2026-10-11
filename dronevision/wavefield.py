"""«Течения и спектр волн»: surface current field and wind-wave spectra from the video itself.

Method (Young et al. 1985; Senet et al. 2001; for drones Streßer, Carrasco & Horstmann 2017): short image
sequences of the sea surface are rectified to the ground, their 3-D spectrum P(kx, ky, ω) is computed, and
the wave energy is found on the dispersion shell

    ω = sqrt(g·k) + k·U        (deep water)

A current U tilts the shell; a weighted least-squares fit of the strongest spectral points gives U for
every ground cell.  The waves that carry the signal are 2–20 m long, so U is the current of the upper
~0.5–1.5 m (wavelength / 4π), averaged over the cell and the record.

Everything is automatic:
* cell size and resolution follow the pixel footprint on the water; cells cover the part of the frame
  where the waves are resolved;
* the attitude of every frame is taken from the image when the sea horizon is visible (``horizon.py``):
  pitch and roll from the horizon line, heading drift from land or clouds on the horizon.  A drift of
  0.7° in a minute (DJI_0531) is 6 m at 500 m — a false current of 0.1 m/s if left to the log;
* the ground scale is checked by the same fit (``scale``): a wrong height or pitch stretches all
  wavenumbers and shows up as a scale ≠ 1;
* a camera heading that drifts without being logged turns the whole picture about the drone and looks like
  a current across the line of sight, proportional to the range (DJI_0879, 2025-09-05: 0.16°/s gave 1.2 m/s
  at 450 m).  With three or more cells this rotation is separated from a uniform current by the pattern of
  the cell vectors (``_rigid_field``) and removed when the heading was not tracked along the horizon;
* a moving drone turns every error of the camera heading and of the ground scale into a false current
  (speed × error): the result is then marked unreliable and the sensitivity is spelled out.

The spectra are spectra of image brightness, not of elevation: peak wavelength, period and directions are
reliable, energy levels are relative.  Checked on DJI_0531 against a manual analysis and a synthetic cube.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from . import horizon
from .cvio import imwrite, open_video
from .geometry import camera_basis, ground_corners, ground_points, horizon_crop_row, pixel_rays
from .naming import metadata as naming_metadata, output_stem
from .pose import frame_poses, load_context

G = 9.81
CELL_PX = 256
TARGET_RATE_HZ = 7.5
MAX_SECONDS = 90.0
MAX_CELLS = 12
MIN_SECONDS = 20.0


def _dependencies():
    import cv2
    from pyproj import Geod
    return cv2, Geod(ellps="WGS84")


# --- spectra and the dispersion fit ---------------------------------------------------------------------

def spectrum(cube: np.ndarray, dt: float, dx: float):
    """Power P[ky, kx, ω] of a cube [t, y, x] for ω > 0, with wavenumbers of the *propagation* direction.

    Returns (P, k_east, k_north, omega); image rows run south, so k_north = +2π·f_y of the transform.
    """
    data = cube.astype(np.float32)
    data -= data.mean(axis=0, keepdims=True)
    data /= data.std(axis=0, keepdims=True) + 1e-3
    nt, ny, nx = data.shape
    data *= np.hanning(nt)[:, None, None].astype(np.float32)
    data *= np.hanning(ny)[None, :, None].astype(np.float32)
    data *= np.hanning(nx)[None, None, :].astype(np.float32)
    transform = np.fft.rfftn(np.transpose(data, (1, 2, 0)))          # [y, x, t]; rfft along time
    power = (transform.real ** 2 + transform.imag ** 2).astype(np.float32)
    # A wave cos(k·x − ωt) has its positive-frequency component at −k.
    k_east = -2 * np.pi * np.fft.fftfreq(nx, dx)
    k_north = 2 * np.pi * np.fft.fftfreq(ny, dx)
    omega = 2 * np.pi * np.fft.rfftfreq(nt, dt)
    return power, k_east, k_north, omega


def _shell_points(power, k_east, k_north, omega, k_min: float, k_max: float, top: int = 6000):
    """Strongest spectral points in a wide band around the still-water dispersion shell."""
    kx, ky = np.meshgrid(k_east, k_north)
    k = np.hypot(kx, ky)
    still = np.sqrt(G * k)[..., None]
    w = omega[None, None, :]
    band = (k[..., None] > k_min) & (k[..., None] < k_max) & (w > 0.5 * still) & (w < 1.6 * still) & (w > 0.8)
    values = np.where(band, power, 0)
    count = min(top, int(band.sum()))
    if count < 200:
        return None
    flat = np.argpartition(values.ravel(), -count)[-count:]
    iy, ix, it = np.unravel_index(flat, values.shape)
    return omega[it], kx[iy, ix], ky[iy, ix], values[iy, ix, it].astype(np.float64)


SHELL_WIDTH = 0.15      # rad/s: final width of the shell in the fit, about the frequency resolution of half a record


def fit_current(points, scale: float = 1.0) -> dict | None:
    """Current U that puts most spectral energy on the dispersion shell; ``scale`` multiplies all wavenumbers.

    Weighted least squares with Gaussian weights exp(−r²/2σ²) on the frequency residual r, σ shrinking from
    0.8 to ``SHELL_WIDTH`` rad/s: points off the shell (noise, harmonics, another cell's leakage) lose their
    weight smoothly instead of pulling the plane, so a weak shell among noise is still found.
    """
    if points is None:
        return None
    w, kx, ky, p = points
    kx, ky = kx * scale, ky * scale
    k = np.hypot(kx, ky)
    design, target = np.column_stack([kx, ky]), w - np.sqrt(G * k)
    current = np.zeros(2)
    for sigma in np.geomspace(0.8, SHELL_WIDTH, 12):
        residual = target - design @ current
        root = np.sqrt(p * np.exp(-0.5 * (residual / sigma) ** 2))
        if (root > 0).sum() < 100:
            return None
        current = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
    residual = target - design @ current
    on_shell = p * np.exp(-0.5 * (residual / SHELL_WIDTH) ** 2)
    keep = np.abs(residual) < 2 * SHELL_WIDTH
    if keep.sum() < 100:
        return None
    weights = p[keep]
    # Spread of the wave directions: with one narrow swell only the along-wave component is measured.
    angle = np.arctan2(kx[keep], ky[keep])
    resultant = np.hypot((weights * np.sin(angle)).sum(), (weights * np.cos(angle)).sum()) / weights.sum()
    return {"u": float(current[0]), "v": float(current[1]), "points": int(keep.sum()),
            "rms": float(np.sqrt((weights * residual[keep] ** 2).sum() / weights.sum())),
            # Share of the selected energy lying on the shell: ~0.75 for clean waves, ~0.15 for noise alone.
            "share": float(on_shell.sum() / p.sum()), "direction_resultant": float(resultant), "keep": keep}


def best_scale(points, steps=np.geomspace(0.6, 1.6, 34)) -> tuple[float, float] | None:
    """Wavenumber scale that puts most energy on the shell (1 = the ground scale of the rectification is right)."""
    rows = []
    for s in steps:
        fit = fit_current(points, float(s))
        if fit:
            rows.append((float(s), fit["share"]))
    if len(rows) < 5:
        return None
    rows = np.array(rows)
    j = int(rows[:, 1].argmax())
    if 0 < j < len(rows) - 1:
        a, b, c = rows[j - 1, 1], rows[j, 1], rows[j + 1, 1]
        offset = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0.0
        return float(rows[j, 0] * (rows[j + 1, 0] / rows[j, 0]) ** offset), float(b)
    return float(rows[j, 0]), float(rows[j, 1])


def _rigid_field(cells: list[dict], camera: np.ndarray) -> dict | None:
    """Uniform current plus a rotation about the camera, fitted to the cell vectors: v = U + Ω ẑ × (r − camera).

    A heading drift of the camera moves every ground point across the line of sight by Ω·range, so cells to
    the left and right of the axis get opposite along-sight components — a pattern a uniform current cannot
    make.  Needs three or more cells off one line; ``sigma_m_s`` says how well U is separated from Ω.
    """
    rows = [c for c in cells if c.get("u_east_m_s") is not None and c["quality"] != "failed"]
    solution = None
    while len(rows) >= 3:
        design, target = [], []
        for c in rows:
            x, y = c["east_m"] - camera[0], c["north_m"] - camera[1]
            design += [[1, 0, -y], [0, 1, x]]
            target += [c["u_east_m_s"], c["v_north_m_s"]]
        design, target = np.array(design, float), np.array(target, float)
        normal = design.T @ design
        if np.linalg.cond(normal) > 1e12:
            return None
        solution = np.linalg.solve(normal, design.T @ target)
        misfit = np.hypot(*(design @ solution - target).reshape(-1, 2).T)
        worst = int(misfit.argmax())
        # One cell far off the common pattern (a failed fit that looked plausible) is dropped once.
        if len(rows) >= 5 and misfit[worst] > max(0.2, 3 * float(np.median(misfit))):
            rows = rows[:worst] + rows[worst + 1:]
            continue
        break
    if solution is None:
        return None
    degrees = max(2 * len(rows) - 3, 1)
    variance = max(float((misfit ** 2).sum()) / degrees, 0.03 ** 2)   # a cell vector is never better than 3 cm/s
    covariance = np.linalg.inv(normal) * variance
    return {"u": float(solution[0]), "v": float(solution[1]), "omega_deg_s": float(np.rad2deg(solution[2])),
            "omega_rad_s": float(solution[2]), "omega_sigma_deg_s": float(np.rad2deg(np.sqrt(covariance[2, 2]))),
            "sigma_m_s": float(np.sqrt(covariance[0, 0] + covariance[1, 1])),
            "residual_m_s": float(np.sqrt((misfit ** 2).mean())), "cells": len(rows)}


# --- geometry of the cells ------------------------------------------------------------------------------

def _pixel_footprint(pose: dict, i: int, east_north: np.ndarray, width: int, height: int):
    """(range-direction, cross-direction) ground size of one image pixel at a sea point; None outside."""
    basis = camera_basis(pose["yaw"][i], pose["pitch"][i], pose["roll"][i])
    relative = np.asarray(east_north, float) - pose["offset"][i]
    vector = np.array([relative[0], relative[1], -pose["altitude"][i]])
    depth = vector @ basis[0]
    if depth <= 0:
        return None
    focal = pose["focal_px"][i]
    u = width / 2 + focal * (vector @ basis[1]) / depth
    v = height / 2 + focal * (vector @ basis[2]) / depth
    margin = 0.02
    if not (margin * width <= u <= (1 - margin) * width and margin * height <= v <= (1 - margin) * height):
        return None
    slant = float(np.linalg.norm(vector))
    return slant ** 2 / (pose["altitude"][i] * focal), slant / focal


def _layout(poses: dict, width: int, height: int) -> tuple[float, list[np.ndarray], dict]:
    """Resolution and centres (ENU, relative to takeoff) of the analysis cells."""
    checks = sorted({0, len(poses["time_s"]) // 2, len(poses["time_s"]) - 1})
    middle = len(poses["time_s"]) // 2
    basis = camera_basis(poses["yaw"][middle], poses["pitch"][middle], poses["roll"][middle])
    focal, altitude = poses["focal_px"][middle], poses["altitude"][middle]
    # Finest pixel on the water: the bottom row of the frame.
    rays = pixel_rays(np.array([width / 2]), np.array([height * 0.97]), width, height, focal, basis)
    if rays[0, 2] >= -1e-6:
        raise ValueError("Камера не видит воду: наклоните подвес ниже горизонта")
    near = ground_points(rays, altitude)[0]
    slant = np.hypot(np.linalg.norm(near), altitude)
    finest = np.sqrt(slant ** 2 / (altitude * focal) * slant / focal)
    resolution = float(np.clip(2.0 ** np.ceil(np.log2(2.0 * finest)), 1 / 32, 1.0))
    size = CELL_PX * resolution
    yaw = np.deg2rad(poses["yaw"][middle])
    forward, right = np.array([np.sin(yaw), np.cos(yaw)]), np.array([np.cos(yaw), -np.sin(yaw)])
    origin = poses["offset"][middle]
    first = float(near @ forward)
    cells = []
    for row in range(12):
        along = first + size * (0.5 + row)
        for column in sorted(range(-8, 9), key=abs):
            centre = origin + forward * along + right * column * size
            corners = centre + 0.5 * size * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
            worst = 0.0
            for i in checks:
                prints = [_pixel_footprint(poses, i, c, width, height) for c in corners]
                if any(p is None for p in prints):
                    worst = np.inf
                    break
                worst = max(worst, max(p[0] for p in prints), max(p[1] for p in prints))
            # Waves of 5 pixels and longer survive a footprint of up to 2.4 grid steps in range.
            if worst <= 2.4 * resolution:
                cells.append((row, abs(column), centre))
    if not cells:
        raise ValueError("В кадре нет участка воды с достаточным разрешением: волны не различимы "
                         "(слишком большая высота или слишком пологий угол визирования)")
    if len(cells) > MAX_CELLS:
        # Keep the cells spread over rows: every row gives up its outermost cells first.
        cells.sort(key=lambda c: (c[1], c[0]))
        cells = cells[:MAX_CELLS]
    cells.sort(key=lambda c: (c[0], c[1]))
    return resolution, [c[2] for c in cells], {"finest_pixel_m": round(float(finest), 3), "cell_m": size}


def _to_cell(pose_basis, focal: float, altitude: float, camera_offset: np.ndarray, centre: np.ndarray,
             resolution: float, width: int, height: int) -> np.ndarray:
    """Matrix taking cell pixels (north up) to image pixels, for ``warpPerspective(..., WARP_INVERSE_MAP)``."""
    half = CELL_PX * resolution / 2
    relative = centre - camera_offset
    ground = np.array([[resolution, 0, relative[0] - half + resolution / 2],
                       [0, -resolution, relative[1] + half - resolution / 2],
                       [0, 0, -altitude]], dtype=np.float64)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    forward, right, down = pose_basis
    return camera @ np.stack([right, down, forward]) @ ground


def _footprint(poses: dict, i: int, width: int, height: int, reach_m: float) -> np.ndarray:
    """ENU outline of the water seen in frame ``i`` closer than ``reach_m``: far-left, far-right, near-right, near-left."""
    basis = camera_basis(poses["yaw"][i], poses["pitch"][i], poses["roll"][i])
    focal, altitude = float(poses["focal_px"][i]), float(poses["altitude"][i])
    crop = horizon_crop_row(width, height, focal, basis, altitude, reach_m)
    return poses["offset"][i] + ground_corners(width, height, focal, basis, altitude, crop)


def _frame_view(cv2, frame, poses: dict, centres, cell_m: float, qualities) -> np.ndarray:
    """The first frame of the interval with the outlines and numbers of the cells, 1280 px wide."""
    height, width = frame.shape[:2]
    basis = camera_basis(poses["yaw"][0], poses["pitch"][0], poses["roll"][0])
    focal, altitude = float(poses["focal_px"][0]), float(poses["altitude"][0])
    scale = 1280 / width
    view = cv2.resize(frame, (1280, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    colours = {"good": (138, 208, 47), "weak": (46, 176, 255), "failed": (69, 91, 255)}
    for number, (centre, quality) in enumerate(zip(centres, qualities), 1):
        t = np.linspace(-0.5, 0.5, 9) * cell_m
        ring = np.concatenate([np.column_stack([t, np.full(9, -0.5 * cell_m)]), np.column_stack([np.full(9, 0.5 * cell_m), t]),
                               np.column_stack([t[::-1], np.full(9, 0.5 * cell_m)]), np.column_stack([np.full(9, -0.5 * cell_m), t[::-1]])])
        vectors = np.column_stack([centre + ring - poses["offset"][0], np.full(len(ring), -altitude)])
        depth = vectors @ basis[0]
        if np.any(depth <= 0):
            continue
        pixels = np.column_stack([width / 2 + focal * (vectors @ basis[1]) / depth, height / 2 + focal * (vectors @ basis[2]) / depth]) * scale
        outline = np.round(pixels).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [outline], True, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(view, [outline], True, colours[quality], 2, cv2.LINE_AA)
        x, y = pixels.mean(axis=0)
        for colour, thickness in (((0, 0, 0), 4), ((255, 255, 255), 2)):
            cv2.putText(view, str(number), (int(x) - 8, int(y) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thickness, cv2.LINE_AA)
    return view


# --- wave statistics ------------------------------------------------------------------------------------

def _wave_statistics(power, k_east, k_north, omega, current: np.ndarray, resolution: float) -> dict:
    """Directional wavenumber spectrum, frequency spectrum and peak parameters from the shell energy."""
    kx, ky = np.meshgrid(k_east, k_north)
    k = np.hypot(kx, ky)
    shell = np.sqrt(G * k) + kx * current[0] + ky * current[1]
    w = omega[None, None, :]
    on_shell = (np.abs(w - shell[..., None]) < np.maximum(0.18 * shell[..., None], 0.35)) & (k[..., None] > 0.08)
    energy = np.where(on_shell, power, 0)
    spectrum_k = energy.sum(axis=2)                                  # E(kx, ky), propagation direction
    spectrum_w = energy.sum(axis=(0, 1))
    # Noise floor: median of the off-shell power at the same frequencies.
    usable = (k > 2 * np.pi / (CELL_PX * resolution) * 3) & (k < np.pi / resolution * 0.8)
    spectrum_k = np.where(usable, spectrum_k, 0)
    total = float(spectrum_k.sum())
    if total <= 0:
        return {}
    direction = np.rad2deg(np.arctan2(kx, ky)) % 360               # where the waves go
    bins = np.arange(0, 361, 10)
    by_direction, _ = np.histogram(direction[usable], bins=bins, weights=spectrum_k[usable])
    k_bins = np.geomspace(max(k[usable].min(), 1e-3), k[usable].max(), 28)
    by_k, _ = np.histogram(k[usable], bins=k_bins, weights=spectrum_k[usable])
    k_centres = np.sqrt(k_bins[:-1] * k_bins[1:])
    smooth = np.convolve(by_k, [0.25, 0.5, 0.25], mode="same")
    k_peak = float(k_centres[int(smooth.argmax())])
    k_mean = float((k[usable] * spectrum_k[usable]).sum() / total)
    sine = float((spectrum_k[usable] * np.sin(np.deg2rad(direction[usable]))).sum())
    cosine = float((spectrum_k[usable] * np.cos(np.deg2rad(direction[usable]))).sum())
    mean_to = float(np.rad2deg(np.arctan2(sine, cosine)) % 360)
    resultant = float(np.hypot(sine, cosine) / total)
    spread = float(np.rad2deg(np.sqrt(max(2 * (1 - resultant), 0))))
    smooth_dir = np.convolve(np.r_[by_direction[-1:], by_direction, by_direction[:1]], [0.25, 0.5, 0.25], mode="valid")
    order = np.argsort(smooth_dir)[::-1]
    systems = []
    for index in order:
        centre = float(bins[index] + 5)
        if smooth_dir[index] < 0.25 * smooth_dir[order[0]]:
            break
        if all(abs((centre - s["direction_to_deg"] + 180) % 360 - 180) >= 50 for s in systems):
            sector = np.abs((direction - centre + 180) % 360 - 180) <= 30
            part = np.where(usable & sector, spectrum_k, 0)
            if part.sum() <= 0:
                continue
            k_part = float((k * part).sum() / part.sum())
            systems.append({"direction_to_deg": centre, "direction_from_deg": (centre + 180) % 360,
                            "share": round(float(part.sum() / total), 2),
                            "wavelength_m": round(2 * np.pi / k_part, 1),
                            "period_s": round(float(2 * np.pi / np.sqrt(G * k_part)), 2)})
        if len(systems) == 3:
            break
    f = omega / (2 * np.pi)
    positive = (f > 0.08) & (spectrum_w > 0)
    f_peak = float(f[positive][np.convolve(spectrum_w[positive], np.ones(5) / 5, mode="same").argmax()]) if positive.any() else 0.0
    # Compact picture of E(kx, ky) for the browser: north up, east right, log scale, 8 bit.
    k_limit = min(float(np.pi / resolution * 0.8), max(4 * k_mean, 3 * k_peak))
    dk = 2 * np.pi / (CELL_PX * resolution)
    half = int(min(np.ceil(k_limit / dk), CELL_PX // 2 - 1))
    east_index = np.argsort(k_east)
    north_index = np.argsort(k_north)[::-1]
    picture = spectrum_k[np.ix_(north_index, east_index)]
    centre_row = int(np.flatnonzero(k_north[north_index] == 0)[0])
    centre_col = int(np.flatnonzero(k_east[east_index] == 0)[0])
    picture = picture[centre_row - half:centre_row + half + 1, centre_col - half:centre_col + half + 1]
    level = np.log10(np.maximum(picture, picture.max() * 1e-3) / picture.max())
    picture8 = np.round((level + 3) / 3 * 255).astype(int)
    return {
        "peak_wavelength_m": round(2 * np.pi / k_peak, 1), "peak_period_s": round(float(2 * np.pi / np.sqrt(G * k_peak)), 2),
        "mean_wavelength_m": round(2 * np.pi / k_mean, 1), "mean_direction_to_deg": round(mean_to, 0),
        "mean_direction_from_deg": round((mean_to + 180) % 360, 0), "directional_spread_deg": round(spread, 0),
        "observed_peak_frequency_hz": round(f_peak, 3), "systems": systems,
        "direction": {"to_deg": (bins[:-1] + 5).tolist(), "energy": np.round(by_direction / by_direction.max(), 4).tolist()},
        "wavenumber": {"k_rad_m": np.round(k_centres, 4).tolist(), "energy": np.round(by_k / by_k.max(), 4).tolist()},
        "frequency": {"hz": np.round(f[positive], 4).tolist(),
                      "energy": np.round(spectrum_w[positive] / spectrum_w[positive].max(), 4).tolist()},
        "kxky": {"dk_rad_m": round(dk, 5), "half": half, "decades": 3, "rows": picture8.tolist()},
    }


# --- the whole analysis -----------------------------------------------------------------------------------

def _compass(degrees: float) -> str:
    names = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ", "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]
    return names[int(round(degrees / 22.5)) % 16]


def analyse(video_path: str, start_s: float | None, end_s: float | None, project_dir: str,
            yaw_offset_deg: float = 0.0, takeoff_height_m: float = 0.0, horizontal_fov_deg: float = 0.0,
            progress=lambda share, text: None) -> dict:
    """Current vectors on a grid of cells plus wave spectra for an interval of the video."""
    from .georaster import _transformer, aeqd_crs, save_georaster
    from .viewer import safe_media_path

    cv2, geod = _dependencies()
    video = safe_media_path(video_path)
    context = load_context(video)
    meta = context.meta
    width, height, fps = meta["width"], meta["height"], meta["fps"]
    start = 0.0 if start_s is None else max(0.0, float(start_s))
    end = meta["duration"] if end_s is None else min(float(end_s), meta["duration"])
    notes = []
    if end - start > MAX_SECONDS:
        end = start + MAX_SECONDS
        notes.append(f"интервал сокращён до первых {MAX_SECONDS:.0f} с")
    if end - start < MIN_SECONDS:
        raise ValueError(f"Нужно не меньше {MIN_SECONDS:.0f} с записи (лучше 40–90 с): за это время волны проходят "
                         "много периодов, и течение отделяется от их собственной скорости")
    step = max(1, int(round(fps / TARGET_RATE_HZ)))
    indices = np.arange(int(np.ceil(start * fps)), int(np.floor(end * fps)), step)
    indices = indices[indices < max(int(meta["frames"]) or 10 ** 9, 1)]
    if len(indices) % 2:
        indices = indices[:-1]
    times = indices / fps
    dt = step / fps
    poses = frame_poses(context, times, geod, yaw_offset_deg, takeoff_height_m, horizontal_fov_deg)
    moved = float(np.max(np.linalg.norm(poses["offset"] - poses["offset"][0], axis=1)))

    cap = open_video(video)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Не удалось прочитать первый кадр интервала")
        # Attitude from the image where the horizon is visible.
        tracker = horizon.DriftTracker(cv2, frame, float(poses["focal_px"][0]), float(poses["altitude"][0]),
                                       float(poses["log_pitch"][0]))
        attitude = {"horizon": tracker.first is not None, "yaw_from_image": False,
                    "log_pitch_deg": round(float(poses["log_pitch"][0]), 2)}
        if tracker.first is not None and abs(tracker.first["pitch_deg"] - float(poses["log_pitch"][0])) > 1.5:
            # Haze hides the sea horizon; the strongest straight edge is then a fog bank or a slick on the water.
            notes.append(f"найденная линия горизонта даёт наклон {tracker.first['pitch_deg']:.1f}° при {poses['log_pitch'][0]:.1f}° "
                         "в логе — похоже на дымку или полосу на воде; использованы углы из лога")
            attitude["rejected_horizon_pitch_deg"] = round(tracker.first["pitch_deg"], 2)
            tracker.first = None
            attitude["horizon"] = False
        if tracker.first is not None:
            delta = tracker.first["pitch_deg"] - float(poses["pitch"][0])
            poses = dict(poses, pitch=poses["pitch"] + delta,
                         roll=poses["roll"] + (tracker.first["roll_deg"] - float(poses["roll"][0])))
            attitude.update(horizon_pitch_deg=round(tracker.first["pitch_deg"], 3),
                            horizon_roll_deg=round(tracker.first["roll_deg"], 3),
                            pitch_minus_log_deg=round(tracker.first["pitch_deg"] - float(poses["log_pitch"][0]), 2))
        elif "rejected_horizon_pitch_deg" not in attitude:
            notes.append("линия горизонта не найдена: углы камеры взяты из лога, дрейф подвеса не исключён")
        resolution, centres, layout = _layout(poses, width, height)
        cell_m = layout["cell_m"]
        cubes = [np.zeros((len(indices), CELL_PX, CELL_PX), np.uint8) for _ in centres]
        # Overview map under the arrows: north up — the cells, the drone and the part of the frame around them,
        # so that it is seen where on the water the cells lie.
        middle = len(indices) // 2
        reach = 1.5 * (max(float(np.linalg.norm(c - poses["offset"][middle])) for c in centres) + cell_m)
        footprint = _footprint(poses, middle, width, height, reach)
        around = np.vstack([np.asarray(centres) - cell_m, np.asarray(centres) + cell_m, footprint, poses["offset"][[middle]]])
        edge = 0.05 * float(np.max(around.max(axis=0) - around.min(axis=0)))      # room for the drone mark and labels
        lower, upper = around.min(axis=0) - edge, around.max(axis=0) + edge
        map_resolution = float(max(resolution, np.max(upper - lower) / 1400))
        map_w, map_h = (np.ceil((upper - lower) / map_resolution)).astype(int)
        map_sum = np.zeros((map_h, map_w, 3), np.float32)
        map_count = np.zeros((map_h, map_w), np.float32)
        first_frame = frame.copy()
        measured = np.full((len(indices), 3), np.nan)       # pitch, roll, yaw shift from the image
        log_yaw = np.rad2deg(np.unwrap(np.deg2rad(poses["yaw"])))
        yaw_first, shift_last, n_last = float(log_yaw[0]), 0.0, 0
        poses["yaw"] = poses["yaw"].copy()
        position = int(indices[0]) + 1
        for n, index in enumerate(indices):
            if n:
                while position < index:
                    if not cap.grab():
                        raise RuntimeError(f"Не удалось прочитать кадр {position}")
                    position += 1
                ok, frame = cap.read()
                position += 1
                if not ok:
                    raise RuntimeError(f"Не удалось прочитать кадр {index}")
            pitch, roll = float(poses["pitch"][n]), float(poses["roll"][n])
            # Heading: the first frame keeps the logged value (plus the user's offset); later frames follow
            # the image where the strip along the horizon is recognised, and the log's own changes elsewhere.
            yaw = float(yaw_first + shift_last + (log_yaw[n] - log_yaw[n_last]))
            if tracker.first is not None:
                seen = tracker.update(frame)
                if seen is not None:
                    pitch, roll = seen["pitch_deg"], seen["roll_deg"]
                    measured[n, :2] = pitch, roll
                    if seen["yaw_ok"]:
                        measured[n, 2] = shift_last = seen["yaw_shift_deg"]
                        n_last = n
                        yaw = float(yaw_first + shift_last)
            poses["pitch"][n], poses["roll"][n], poses["yaw"][n] = pitch, roll, yaw % 360
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 0.8)
            basis = camera_basis(yaw, pitch, roll)
            for cube, centre in zip(cubes, centres):
                matrix = _to_cell(basis, float(poses["focal_px"][n]), float(poses["altitude"][n]), poses["offset"][n],
                                  centre, resolution, width, height)
                cube[n] = cv2.warpPerspective(gray, matrix, (CELL_PX, CELL_PX),
                                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            if n % 8 == 0:
                relative = lower - poses["offset"][n]
                ground = np.array([[map_resolution, 0, relative[0]], [0, -map_resolution, relative[1] + map_h * map_resolution],
                                   [0, 0, -poses["altitude"][n]]], dtype=np.float64)
                camera = np.array([[poses["focal_px"][n], 0, width / 2], [0, poses["focal_px"][n], height / 2], [0, 0, 1]])
                warped = cv2.warpPerspective(frame, camera @ np.stack([basis[1], basis[2], basis[0]]) @ ground,
                                             (int(map_w), int(map_h)), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                # Only the water this frame really shows: beyond the horizon the homography folds the sky back.
                try:
                    outline = _footprint(poses, n, width, height, reach)
                    inside = np.zeros((map_h, map_w), np.uint8)
                    cv2.fillConvexPoly(inside, np.round(np.column_stack([
                        (outline[:, 0] - lower[0]) / map_resolution,
                        (lower[1] + map_h * map_resolution - outline[:, 1]) / map_resolution])).astype(np.int32), 1)
                    cv2.accumulateProduct(warped.astype(np.float32), cv2.merge([inside.astype(np.float32)] * 3), map_sum)
                    map_count += inside
                except ValueError:
                    pass
            if n % 40 == 0:
                progress(0.05 + 0.55 * n / len(indices), f"Проецирую кадры: {n} из {len(indices)}")
    finally:
        cap.release()

    tracked = np.isfinite(measured[:, 2])
    if tracked.mean() > 0.8:
        difference = np.interp(times, times[tracked], measured[tracked, 2]) - (log_yaw - log_yaw[0])
        attitude.update(yaw_from_image=True, yaw_drift_deg=round(float(difference[-1] - difference[0]), 3),
                        yaw_drift_span_deg=round(float(np.ptp(difference)), 3))
    elif tracker.first is not None:
        attitude["yaw_untracked"] = True     # no fixed detail on the horizon; see the rotation check below
    if tracker.first is not None and np.isfinite(measured[:, 0]).any():
        attitude["pitch_span_deg"] = round(float(np.nanmax(measured[:, 0]) - np.nanmin(measured[:, 0])), 3)

    # --- fits ---
    lat0, lon0 = context.takeoff
    to_lonlat = _transformer(aeqd_crs(lat0, lon0), "EPSG:4326")
    half = len(indices) // 2
    k_min = 2 * np.pi / (cell_m / 6)
    # Waves whose frequency (with a current of ~1 m/s) reaches the Nyquist frequency of the sampling alias.
    k_max = min(np.pi / resolution * 0.7, (0.8 * np.pi / dt) ** 2 / G)
    camera_position = poses["offset"].mean(axis=0)
    cells, pooled_good, pooled_weak = [], None, None
    for number, (cube, centre) in enumerate(zip(cubes, centres)):
        progress(0.6 + 0.35 * number / len(cubes), f"Спектры: участок {number + 1} из {len(cubes)}")
        first, k_east, k_north, omega = spectrum(cube[:half], dt, resolution)
        second = spectrum(cube[half:], dt, resolution)[0]
        both = first + second
        fits = [fit_current(_shell_points(p, k_east, k_north, omega, k_min, k_max)) for p in (both, first, second)]
        lon, lat = to_lonlat.transform(float(centre[0]), float(centre[1]))
        cell = {"east_m": round(float(centre[0]), 1), "north_m": round(float(centre[1]), 1),
                "latitude": round(float(lat), 7), "longitude": round(float(lon), 7), "size_m": cell_m,
                "range_m": round(float(np.linalg.norm(centre - camera_position)), 0), "quality": "failed"}
        fit = fits[0]
        if fit is not None:
            u, v = fit["u"], fit["v"]
            halves = [f for f in fits[1:] if f is not None]
            difference = float(np.hypot(halves[0]["u"] - halves[1]["u"], halves[0]["v"] - halves[1]["v"])) if len(halves) == 2 else None
            scale = best_scale(_shell_points(both, k_east, k_north, omega, k_min, k_max))
            # Energy on the shell: 0.6–0.8 for clean waves, 0.15–0.2 when the cube is noise.
            quality = ("failed" if fit["share"] < 0.25 else
                       "good" if (fit["share"] >= 0.5 and difference is not None and difference < 0.12
                                  and fit["direction_resultant"] < 0.97) else "weak")
            cell.update(u_east_m_s=round(u, 3), v_north_m_s=round(v, 3), speed_m_s=round(float(np.hypot(u, v)), 3),
                        direction_to_deg=round(float(np.rad2deg(np.arctan2(u, v)) % 360), 1),
                        halves_difference_m_s=None if difference is None else round(difference, 3),
                        fit_rms_rad_s=round(fit["rms"], 3), points=fit["points"], shell_share=round(fit["share"], 2),
                        scale=None if scale is None else round(scale[0], 3), quality=quality,
                        one_direction=bool(fit["direction_resultant"] >= 0.97))
            if quality == "good":
                pooled_good = both.copy() if pooled_good is None else pooled_good + both
            elif quality == "weak":
                pooled_weak = both.copy() if pooled_weak is None else pooled_weak + both
        cells.append(cell)
        del first, second, both
    del cubes

    pooled = pooled_good if pooled_good is not None else pooled_weak
    good = [c for c in cells if c["quality"] == "good"] or [c for c in cells if c["quality"] == "weak"]
    if not good:
        raise ValueError("Дисперсионная поверхность волн не найдена ни на одном участке: волнение слишком слабое, "
                         "блики или кадр слишком далёк от воды")
    u_all, v_all = np.array([c["u_east_m_s"] for c in good]), np.array([c["v_north_m_s"] for c in good])
    measured_mean = np.array([np.median(u_all), np.median(v_all)])     # as seen in the (possibly turning) picture
    scatter = float(np.hypot(np.std(u_all), np.std(v_all))) if len(good) > 1 else None
    differences = [c["halves_difference_m_s"] for c in good if c.get("halves_difference_m_s") is not None]
    uncertainty = max(scatter or 0.0, float(np.median(differences)) / 2 if differences else 0.0) / np.sqrt(len(good))
    scales = [c["scale"] for c in good if c.get("scale")]

    duration = float(times[-1] - times[0])
    drone_velocity = (poses["offset"][-1] - poses["offset"][0]) / max(duration, 1e-6)
    drone_speed = float(np.hypot(*drone_velocity))

    # Camera heading drift: a rotation of the cell vectors about the drone.
    mean, rotation = measured_mean, None
    rigid = _rigid_field(cells, camera_position)
    reasons = []
    if rigid is not None:
        separable = rigid["sigma_m_s"] <= 0.3
        # In flight the cell pattern also carries the heading and scale errors times the drone speed: no correction.
        apply = separable and not attitude["yaw_from_image"] and drone_speed <= 0.5
        rotation = {"omega_deg_s": round(rigid["omega_deg_s"], 4), "omega_sigma_deg_s": round(rigid["omega_sigma_deg_s"], 4),
                    "sigma_m_s": round(rigid["sigma_m_s"], 3), "residual_m_s": round(rigid["residual_m_s"], 3),
                    "cells": rigid["cells"], "applied": bool(apply),
                    "false_current_m_s": round(float(abs(rigid["omega_rad_s"]) * np.median([c["range_m"] for c in good])), 3)}
        if apply:
            mean = np.array([rigid["u"], rigid["v"]])
            uncertainty = max(rigid["sigma_m_s"], uncertainty)
            if abs(rigid["omega_deg_s"]) > max(0.01, 2 * rigid["omega_sigma_deg_s"]):
                notes.append(f"картинка поворачивается вокруг дрона на {abs(rigid['omega_deg_s']):.3f}°/с (увод курса камеры, которого нет "
                             f"в логе): это {rotation['false_current_m_s']:.2f} м/с ложного течения поперёк луча зрения — исключено "
                             "по различию векторов на участках слева и справа")
        elif not attitude["yaw_from_image"] and drone_speed <= 0.5:
            reasons.append("курс камеры не проверен: участки лежат слишком кучно, чтобы отделить увод курса от течения")
    elif not attitude["yaw_from_image"]:
        reasons.append("курс камеры не проверен (нужно 3+ участка или детали на горизонте): увод курса на 0.1°/с — "
                       "это ложные 0.17 м/с на каждые 100 м дальности")
    for c in cells:
        if c.get("u_east_m_s") is None:
            continue
        turn = rigid["omega_rad_s"] if rotation and rotation["applied"] else 0.0
        u = c["u_east_m_s"] + turn * (c["north_m"] - camera_position[1])
        v = c["v_north_m_s"] - turn * (c["east_m"] - camera_position[0])
        c.update(measured_u_east_m_s=c["u_east_m_s"], measured_v_north_m_s=c["v_north_m_s"], u_east_m_s=round(float(u), 3),
                 v_north_m_s=round(float(v), 3), speed_m_s=round(float(np.hypot(u, v)), 3),
                 direction_to_deg=round(float(np.rad2deg(np.arctan2(u, v)) % 360), 1))

    # A moving drone: every error of heading and scale becomes a false current of (drone speed × error).
    scale_value = float(np.median(scales)) if scales else None
    if drone_speed > 0.5:
        along = drone_velocity / drone_speed
        across = float(mean[0] * along[1] - mean[1] * along[0])
        reasons.append(f"дрон летит со скоростью {drone_speed:.1f} м/с: ошибка курса камеры в 1° даёт {drone_speed * 0.0175:.2f} м/с "
                       f"ложного течения поперёк пути, ошибка масштаба в 1 % — {drone_speed * 0.01:.2f} м/с вдоль пути. "
                       f"Найденная поперечная составляющая {abs(across):.2f} м/с равна ошибке курса "
                       f"{np.rad2deg(np.arctan2(abs(across), drone_speed)):.0f}° — для течений нужно висение")
    elif moved > 15:
        notes.append(f"дрон сместился на {moved:.0f} м за интервал: для течений лучше висение")
    if scale_value is not None and abs(scale_value - 1) > 0.1:
        reasons.append(f"масштаб по дисперсии волн {scale_value:.2f} вместо 1: высота над водой или наклон камеры неверны на "
                       "эту долю — выровняйте кадр по горизонту или берегу и проверьте «Взлёт над водой»")
    elif scale_value is not None and abs(scale_value - 1) > 0.06:
        notes.append(f"масштаб по дисперсии волн {scale_value:.2f} ≠ 1: проверьте «Взлёт над водой» и наклон камеры")
    has_good = any(c["quality"] == "good" for c in cells)
    if not has_good:
        reasons.append("ни на одном участке волны не дали уверенной дисперсионной поверхности "
                       "(мало энергии на ней или половины записи расходятся)")
    if duration < 40:
        notes.append(f"запись {duration:.0f} с: для течений лучше 40–90 с")
    gross = drone_speed > 0.5 or (scale_value is not None and abs(scale_value - 1) > 0.1)
    reliability = "unreliable" if gross else "weak" if reasons else "good"
    summary = {"u_east_m_s": round(float(mean[0]), 3), "v_north_m_s": round(float(mean[1]), 3),
               "speed_m_s": round(float(np.hypot(*mean)), 3),
               "direction_to_deg": round(float(np.rad2deg(np.arctan2(mean[0], mean[1])) % 360), 0),
               "uncertainty_m_s": round(float(uncertainty), 3),
               "measured_u_east_m_s": round(float(measured_mean[0]), 3), "measured_v_north_m_s": round(float(measured_mean[1]), 3),
               "measured_speed_m_s": round(float(np.hypot(*measured_mean)), 3),
               "rotation": rotation, "reliability": reliability, "reasons": reasons,
               "drone_speed_m_s": round(drone_speed, 2),
               "cells_used": len(good), "cells_total": len(cells),
               "scatter_m_s": None if scatter is None else round(scatter, 3),
               "scale": None if scale_value is None else round(scale_value, 3),
               "depth_of_measurement_m": None}
    summary["direction_to_text"] = _compass(summary["direction_to_deg"])
    # The shell in the pooled spectrum sits where the picture shows it: with the apparent (measured) velocity.
    waves = _wave_statistics(pooled, k_east, k_north, omega, measured_mean, resolution) if pooled is not None else {}
    if waves:
        summary["depth_of_measurement_m"] = round(waves["mean_wavelength_m"] / (4 * np.pi), 2)

    # --- files ---
    progress(0.96, "Сохраняю результаты")
    output_root = Path(project_dir).expanduser().resolve() if project_dir else video.parent / "DroneVision-project"
    folder = output_root / "waves"
    folder.mkdir(parents=True, exist_ok=True)
    stem = output_stem(folder, context, times[0], times[-1], "currents", (".png", ".json", ".csv", ".geo.json"))
    valid = map_count > 0
    overview = (map_sum / np.maximum(map_count, 1)[..., None]).clip(0, 255).astype(np.uint8)
    frame_path = folder / f"{stem}_frame.jpg"
    if not imwrite(frame_path, _frame_view(cv2, first_frame, poses, centres, cell_m, [c["quality"] for c in cells]),
                   [cv2.IMWRITE_JPEG_QUALITY, 88]):
        frame_path = None
    image_path = folder / f"{stem}.png"
    if not imwrite(image_path, overview):
        raise RuntimeError("Не удалось сохранить карту")
    georaster = save_georaster(image_path, overview, valid, lower, lower + np.array([map_w, map_h]) * map_resolution,
                               map_resolution, (lat0, lon0), "wave_current_map", None, video=str(video),
                               times_s=[round(float(times[0]), 3), round(float(times[-1]), 3)],
                               yaw_offset_deg=yaw_offset_deg, takeoff_height_m=takeoff_height_m,
                               **naming_metadata(context))
    # The SRT clock is the camera's local time written as an epoch; the log (UTC) differs by whole hours.
    utc_start = datetime.fromtimestamp(context.start_epoch + float(times[0]) - context.utc_offset_hours * 3600,
                                       timezone.utc).isoformat(timespec="seconds")
    result = {
        "kind": "wave_currents", "video": str(video), "interval_s": [round(float(times[0]), 3), round(float(times[-1]), 3)],
        "utc_start": utc_start, **naming_metadata(context),
        "frames": int(len(indices)), "frame_rate_hz": round(1 / dt, 2), "resolution_m_per_px": resolution,
        "cell_m": cell_m, "layout": layout, "cells": cells, "summary": summary, "waves": waves, "attitude": attitude,
        "notes": notes, "yaw_offset_deg": yaw_offset_deg, "takeoff_height_m": takeoff_height_m,
        "takeoff": {"latitude": lat0, "longitude": lon0},
        "camera_m": np.round(camera_position, 1).tolist(),
        "camera_altitude_m": round(float(poses["altitude"].mean()), 1),
        "map": {"path": str(image_path), "size_px": [int(map_w), int(map_h)], "resolution_m_per_px": map_resolution,
                "bounds_m": {"east": [float(lower[0]), float(lower[0] + map_w * map_resolution)],
                             "north": [float(lower[1]), float(lower[1] + map_h * map_resolution)]},
                "geo_grid": georaster["geo_grid"],
                # Water seen in the middle frame (far-left, far-right, near-right, near-left), for the outline.
                "footprint_m": np.round(footprint, 1).tolist()},
        "frame_view": None if frame_path is None else {"path": str(frame_path), "time_s": round(float(times[0]), 3)},
        "method": "wind-wave dispersion shell fit, deep water; brightness spectra; camera rotation separated by the cell pattern",
    }
    json_path = folder / f"{stem}.json"
    csv_path = folder / f"{stem}.csv"
    result.update(path=str(json_path), csv_path=str(csv_path))
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    header = ["latitude", "longitude", "east_m", "north_m", "range_m", "u_east_m_s", "v_north_m_s", "speed_m_s",
              "direction_to_deg", "measured_u_east_m_s", "measured_v_north_m_s", "halves_difference_m_s",
              "fit_rms_rad_s", "shell_share", "points", "scale", "quality"]
    lines = [";".join(header)] + [";".join("" if c.get(key) is None else str(c.get(key)) for key in header) for c in cells]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    progress(1.0, "Готово")
    return result

