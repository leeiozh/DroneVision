# utils/projection_utils.py

import cv2
import numpy as np


def project_frame_pitch_only(frame, pitch_deg, altitude, fov_h_deg=78.8, resolution_m=0.5, hor=0.12, grid_step=None,
                             debug_show=False):
    H_full, W = frame.shape[:2]

    # === 1. FOV и фокусные расстояния ===
    fov_h = np.radians(fov_h_deg)
    aspect_ratio = H_full / W
    fov_v = 2 * np.arctan(aspect_ratio * np.tan(fov_h / 2))
    fx = (W / 2) / np.tan(fov_h / 2)
    fy = (H_full / 2) / np.tan(fov_v / 2)

    # === 2. Аналитическая обрезка по горизонту ===
    pitch_rad = np.radians(pitch_deg)
    horizon_norm = np.tan(fov_v / 2 + pitch_rad) / 2 / np.tan(fov_v / 2) + hor  # hor - важный отступ
    horizon_norm = np.clip(horizon_norm, 0.0, 1.0)
    y_start = int(H_full * horizon_norm)
    if debug_show:
        print(f"[DEBUG] Горизонт на {horizon_norm * 100:.1f}% → обрезка по {y_start}/{H_full}")
    frame = frame[y_start:, :]
    H = frame.shape[0]

    # === 3. Углы изображения ===
    image_corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

    # === 4. Лучи из камеры и поворот по pitch ===
    rays = []
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)],
    ])

    for (px, py) in image_corners:
        x = (px - W / 2) / fx
        y = (py + y_start - H_full / 2) / fy
        v_cam = np.array([x, y, 1.0])
        v_drone = np.array([v_cam[0], v_cam[2], -v_cam[1]])  # камера (z н нас, y вверх) → дрон (z вверх, y от нас)
        v_world = R_pitch @ v_drone
        v_world /= np.linalg.norm(v_world)
        rays.append(v_world)
    rays = np.stack(rays)
    if debug_show:
        print("[DEBUG] Rays:\n", rays)

    # === 5. Пересечение лучей с землей (Z = 0) ===
    ground_corners = []
    for i, ray in enumerate(rays):
        scale = altitude / ray[2]
        x = ray[0] * scale
        y = ray[1] * scale
        ground_corners.append([x, y])
        if debug_show:
            print(f"[DEBUG] Ground corner {i}: ({x:.2f}, {y:.2f})")
    ground_corners = np.array(ground_corners, dtype=np.float32)

    # === 6. Размер проекции ===
    x_min, y_min = ground_corners.min(axis=0)
    x_max, y_max = ground_corners.max(axis=0)
    width_m = x_max - x_min
    height_m = y_max - y_min
    out_w = int(np.ceil(width_m / resolution_m))
    out_h = int(np.ceil(height_m / resolution_m))
    if debug_show:
        print(f"[DEBUG] Выход: {out_w}×{out_h} px @ {resolution_m:.2f} m/px")

    # === 7. Построение remap-сетки ===
    H_warp = cv2.getPerspectiveTransform(ground_corners, image_corners)

    x_coords = np.linspace(x_min, x_max, out_w)
    y_coords = np.linspace(y_min, y_max, out_h)

    map_x, map_y = np.meshgrid(x_coords[::-1], y_coords)
    pts = np.stack([map_x, map_y], axis=-1).reshape(-1, 2).astype(np.float32)
    src_pts = cv2.perspectiveTransform(pts[None], H_warp)[0]
    map_x_img = src_pts[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y_img = src_pts[:, 1].reshape(out_h, out_w).astype(np.float32)

    ortho = cv2.remap(frame, map_x_img, map_y_img, interpolation=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    # === 8. Отображение через OpenCV с сеткой ===
    if grid_step:
        draw_with_grid_cv2(ortho, resolution_m, x_min, np.abs(y_max), x_max, np.abs(y_min), grid_step_m=grid_step)

    return ortho, x_min, x_max, y_min, y_max, out_w, out_h


def draw_with_grid_cv2(ortho, resolution_m, x_min, y_min, x_max, y_max, grid_step_m=50):
    """
    Показывает ортофото с наложенной метрической сеткой.
    Координаты (0,0) соответствуют положению дрона.
    """
    image = ortho.copy()
    h, w = image.shape[:2]

    color_grid = (0, 0, 0)
    color_text = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Вертикальные линии (ось X)
    x0 = int(x_min // grid_step_m * grid_step_m)
    x1 = int(x_max // grid_step_m * grid_step_m + grid_step_m)
    for x in range(x0, x1 + 1, grid_step_m):
        px = int((x - x_min) / resolution_m)
        if 0 <= px < w:
            cv2.line(image, (px, 0), (px, h - 1), color_grid, 1)
            cv2.putText(image, f'{x:+}', (px + 2, 15), font, 0.4, color_text, 1)

    # Горизонтальные линии (ось Y)
    y0 = int(y_min // grid_step_m * grid_step_m)
    y1 = int(y_max // grid_step_m * grid_step_m + grid_step_m)
    for y in range(y0, y1 + 1, grid_step_m):
        py = int((y_max - y) / resolution_m)
        if 0 <= py < h:
            cv2.line(image, (0, py), (w - 1, py), color_grid, 1)
            cv2.putText(image, f'{y:+}', (5, py - 2), font, 0.4, color_text, 1)

    # Показать изображение
    cv2.imshow("geoprojected frame", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
