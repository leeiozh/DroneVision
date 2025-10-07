# utils/mosaic_utils.py

import os
import cv2
import numpy as np
from tqdm import tqdm
import datetime as dt
from pyproj import Geod
from config import FRAME_FORMAT, PIXEL_SCALE

geod = Geod(ellps='WGS84')


def extract_binary_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray.astype(np.float32)


def latlon_offset_m(lat0, lon0, lat, lon):
    az12, az21, dist = geod.inv(lon0, lat0, lon, lat)
    angle_rad = np.radians(az12)
    dx = dist * np.sin(angle_rad)
    dy = dist * np.cos(angle_rad)
    return dx, dy


def rotate_and_shift_frames(
        projected_dir, shifted_dir,
        lats, lons, yaws,
        resolution_m=PIXEL_SCALE,
        suffix="_proj", out_suffix="_proj_shft", debug=False
):
    csv_path = os.path.join(projected_dir, "proj_coords.csv")
    coords = np.genfromtxt(csv_path, delimiter=",", skip_header=1)

    # === 1. Собираем имена и кадры
    file_names = sorted([f for f in os.listdir(projected_dir) if f.endswith(suffix + "." + FRAME_FORMAT)])
    file_paths = [os.path.join(projected_dir, f) for f in file_names]
    N = len(file_paths)

    # === 2. Вычисляем все повёрнутые и смещённые углы
    lat0, lon0 = lats[0], lons[0]
    global_corners = np.zeros((N, 4, 2), dtype=np.float32)  # [frame, 3 точки, (x,y)]

    for i in range(N):
        # Углы текущего кадра в локальной СК
        x0, x1 = coords[i, 1], coords[i, 2]
        y0, y1 = coords[i, 3], coords[i, 4]
        local_corners = np.array([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1]
        ], dtype=np.float32)

        # Поворот
        yaw_rad = np.radians(yaws[i])
        R = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)]
        ])
        corners_rot = local_corners @ R.T
        dx, dy = latlon_offset_m(lat0, lon0, lats[i], lons[i])
        global_corners[i] = corners_rot + np.array([dx, dy])

        if debug:
            print(f"[DEBUG] Снимок {i:02d} углы (м):")
            for pt in global_corners[i]:
                print(f"    ({pt[0]:+.1f}, {pt[1]:+.1f})")

    # === 3. Границы глобального холста
    xg_min = np.floor(global_corners[:, :, 0].min())
    xg_max = np.ceil(global_corners[:, :, 0].max())
    yg_min = np.floor(global_corners[:, :, 1].min())
    yg_max = np.ceil(global_corners[:, :, 1].max())

    out_w = int(np.ceil((xg_max - xg_min) / resolution_m))
    out_h = int(np.ceil((yg_max - yg_min) / resolution_m))
    print(f"[INFO] Итоговый размер холста: {out_w}×{out_h} px")

    # === 4. Координаты пикселей в метрах
    x_coords_global = np.linspace(xg_min, xg_max, out_w)
    y_coords_global = np.linspace(yg_min, yg_max, out_h)

    # === 5. Преобразование и сохранение
    for i in tqdm(range(N), desc="Поворот + сдвиг"):
        frame = cv2.imread(file_paths[i])
        H, W = frame.shape[:2]

        src_pts = np.float32([
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1]
        ])

        dst_pts = np.float32([
            [(global_corners[i, 0, 0] - xg_min) / resolution_m, (global_corners[i, 0, 1] - yg_min) / resolution_m],
            [(global_corners[i, 1, 0] - xg_min) / resolution_m, (global_corners[i, 1, 1] - yg_min) / resolution_m],
            [(global_corners[i, 2, 0] - xg_min) / resolution_m, (global_corners[i, 2, 1] - yg_min) / resolution_m],
        ])

        M = cv2.getAffineTransform(src_pts, dst_pts)
        canvas = cv2.warpAffine(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        out_path = os.path.join(shifted_dir, f"frame_{i:04d}{out_suffix}.{FRAME_FORMAT}")
        cv2.imwrite(out_path, canvas)

    return x_coords_global, y_coords_global


def render_projected_video_with_grid(
        shifted_dir,
        x_coords, y_coords,
        datetimes, lats, lons, height, pitches, rolls, yaws,
        out_path="mosaic_output.mp4",
        fps=1, grid_step_m=50,
        suffix="_proj_shft"
):
    y_coords = y_coords
    file_names = sorted([f for f in os.listdir(shifted_dir) if f.endswith(suffix + "." + FRAME_FORMAT)])
    file_paths = [os.path.join(shifted_dir, f) for f in file_names]

    sample = cv2.imread(file_paths[0])
    H, W = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    x0_m = x_coords[0]
    y0_m = y_coords[0]
    resolution_m = (x_coords[1] - x_coords[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(H, W) / 1000
    thickness = int(min(H, W) / 500)

    for i, path in enumerate(tqdm(file_paths, desc="Рендер видео")):
        frame = cv2.imread(path)
        canvas = frame.copy()

        # горизонтальные линии
        y1_m = y_coords[-1]
        y_lines = np.arange(np.floor(y0_m / grid_step_m) * grid_step_m,
                            y1_m + grid_step_m, grid_step_m)
        for y in y_lines:
            py = int((y1_m - y) / resolution_m)
            if 0 <= py < H:
                cv2.line(canvas, (0, py), (W - 1, py), (255, 255, 255), 1)
                cv2.putText(canvas, f"{int(y):+}", (10, py), font, font_scale, (255, 255, 255), thickness)

        # вертикальные линии
        x1_m = x_coords[-1]
        x_lines = np.arange(np.floor(x0_m / grid_step_m) * grid_step_m,
                            x1_m + grid_step_m, grid_step_m)
        for x in x_lines:
            px = int((x - x0_m) / resolution_m)
            if 0 <= px < W:
                cv2.line(canvas, (px, 0), (px, H - 1), (255, 255, 255), 1)
                cv2.putText(canvas, f"{int(x):+}", (px, int(0.02 * H)), font, font_scale, (255, 255, 255), thickness)

        # текст
        log_lines = [
            dt.datetime.fromtimestamp(datetimes[i]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-5],
            f"lat:  {lats[i]:.6f}",
            f"lon:  {lons[i]:.6f}",
            f"alt:  {height[i]:.1f} m",
            f"pitch:{pitches[i]:.1f}^o",
            f"roll: {rolls[i]:.1f}^o",
            f"yaw:  {yaws[i]:.1f}^o"
        ]

        scale = H * 0.05

        for j, text in enumerate(log_lines):
            y_pos = H - 10 - int((len(log_lines) - 1 - j) * scale)
            cv2.putText(canvas, text, (int(W * 0.1), y_pos), font, font_scale * 1.2, (255, 255, 255), thickness)

        # --- оставить только красный канал и увеличить контраст ---
        # red = canvas[:, :, 2].astype(np.float32)
        # red -= red.min()
        # if red.max() > 0:
        #     red = 255 * red / red.max()
        # red = red.astype(np.uint8)
        # canvas = cv2.merge([red, red, red])
        writer.write(canvas)

    writer.release()
    print(f"[INFO] Видео сохранено в {out_path}")


def render_projected_image_with_grid(
        shifted_dir,
        x_coords, y_coords,
        out_path="mosaic_output.png",
        grid_step_m=50,
        suffix="_proj_shft"
):
    file_names = sorted([f for f in os.listdir(shifted_dir) if f.endswith(suffix + "." + FRAME_FORMAT)])
    file_paths = [os.path.join(shifted_dir, f) for f in file_names]

    if not file_paths:
        raise FileNotFoundError("Нет подходящих файлов для мозаики.")

    # Загружаем первое изображение, чтобы получить размеры
    sample = cv2.imread(file_paths[0], cv2.IMREAD_COLOR)
    H, W = sample.shape[:2]
    acc_image = np.zeros((H, W, 3), dtype=np.float32)
    weight_mask = np.zeros((H, W), dtype=np.float32)

    for path in tqdm(file_paths, desc="Складываем изображения"):
        frame = cv2.imread(path).astype(np.float32)
        # Маска ненулевых пикселей
        mask = np.any(frame > 10, axis=2).astype(np.float32)
        for c in range(3):
            acc_image[:, :, c] += frame[:, :, c] * mask
        weight_mask += mask

    # Избегаем деления на ноль
    weight_mask[weight_mask == 0] = 1.0
    mosaic = (acc_image / weight_mask[..., None]).astype(np.uint8)
    canvas = mosaic.copy()

    # --- Рисуем сетку ---
    x0_m = x_coords[0]
    y0_m = y_coords[0]
    resolution_m = (x_coords[1] - x_coords[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(H, W) / 1000
    thickness = int(min(H, W) / 500)

    # Горизонтальные линии
    y1_m = y_coords[-1]
    y_lines = np.arange(np.floor(y0_m / grid_step_m) * grid_step_m,
                        y1_m + grid_step_m, grid_step_m)
    for y in y_lines:
        py = int((y1_m - y) / resolution_m)
        if 0 <= py < H:
            cv2.line(canvas, (0, py), (W - 1, py), (255, 255, 255), 1)
            cv2.putText(canvas, f"{int(y):+}", (10, py), font, font_scale, (255, 255, 255), thickness)

    # Вертикальные линии
    x1_m = x_coords[-1]
    x_lines = np.arange(np.floor(x0_m / grid_step_m) * grid_step_m,
                        x1_m + grid_step_m, grid_step_m)
    for x in x_lines:
        px = int((x - x0_m) / resolution_m)
        if 0 <= px < W:
            cv2.line(canvas, (px, 0), (px, H - 1), (255, 255, 255), 1)
            cv2.putText(canvas, f"{int(x):+}", (px, int(0.02 * H)), font, font_scale, (255, 255, 255), thickness)

    cv2.imwrite(out_path, canvas)
    print(f"[INFO] Аппликация сохранена в {out_path}")