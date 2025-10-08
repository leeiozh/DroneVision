# main_projection.py

import os
import cv2
import csv
import shutil
import numpy as np
from tqdm import tqdm
import datetime as dt
from pathlib import Path

from config_projection import *
from utils.projection_utils import project_frame_pitch_only
from utils.io_utils import save_frames_from_video, get_meta_video, parse_json
from utils.mosaic_utils import rotate_and_shift_frames, render_projected_video_with_grid, \
    render_projected_image_with_grid


def setup_output_dirs(base_output="output"):
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_output) / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def main():
    # === 0. Настройка директорий ===
    RUN_DIR = setup_output_dirs()
    FRAME_DIR = RUN_DIR / "frames"
    PROJ_DIR = RUN_DIR / "proj"
    SHIFT_DIR = RUN_DIR / "shift"
    FRAME_DIR.mkdir(exist_ok=True)
    PROJ_DIR.mkdir(exist_ok=True)
    SHIFT_DIR.mkdir(exist_ok=True)

    # === 1. Считываем логи
    logs = parse_json(LOG_PATH, stamp=True, delta=DELTA_HOURS)

    # === 2. Определяем время старта видео, сравниваем с временем старта логов
    start_time = get_meta_video(VIDEO_PATH).timestamp()
    print(f"[INFO] Видео начинается в {dt.datetime.fromtimestamp(start_time)}")
    print(f"[INFO] Логи с {dt.datetime.fromtimestamp(logs['tim'][0])} до"
          f" {dt.datetime.fromtimestamp(logs['tim'][-1])}")
    assert start_time > logs["tim"][0], "[ERROR] Логи должны начинаться раньше видео!!"

    # 3. Получаем fps из видео
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    cap.release()
    print(f"[INFO] FPS видео = {fps:.2f}, длительность = {duration_s:.2f}")

    assert 0 <= VIDEO_START_OFFSET_S < VIDEO_END_OFFSET_S <= duration_s, \
        "[ERROR] Неверные границы обрезки видео!"

    start_frame_idx = int(VIDEO_START_OFFSET_S * fps)
    end_frame_idx = int(VIDEO_END_OFFSET_S * fps)

    # 4. Загружаем кадры
    if not FORCE_EXTRACT and any(fname.endswith(('.jpg', '.png', '.tif')) for fname in os.listdir(FRAME_DIR)):
        print("[INFO] Найдены сохранённые кадры. Пропускаем извлечение.")
    else:
        print(f"[INFO] Извлекаем кадры из видео с {VIDEO_START_OFFSET_S}s до {VIDEO_END_OFFSET_S}s...")
        save_frames_from_video(VIDEO_PATH, save_dir=FRAME_DIR, frame_interval=FRAME_INTERVAL,
                               start_frame=start_frame_idx, end_frame=end_frame_idx)

    frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(('.jpg', '.png'))])

    frame_times = (start_time + VIDEO_START_OFFSET_S) + np.arange(len(frame_files)) * FRAME_INTERVAL / fps
    log_int = {k: np.interp(frame_times, logs["tim"], logs[k]) for k in logs if k != "tim"}

    assert start_time + VIDEO_START_OFFSET_S + len(frame_files) * FRAME_INTERVAL / fps < logs["tim"][-1], \
        "[ERROR] Логи должны заканчиваться позднее видео! Проверьте FRAME_INTERVAL."

    if not FORCE_PROJECT and all(f.startswith("frame_") and "_proj.png" in f for f in os.listdir(PROJ_DIR)):
        print("[INFO] Проекции уже существуют. Пропускаем проецирование.")
    else:
        if os.path.exists(PROJ_DIR):
            shutil.rmtree(PROJ_DIR)
        os.makedirs(PROJ_DIR, exist_ok=True)
        csv_path = os.path.join(PROJ_DIR, "proj_coords.csv")
        with open(csv_path, mode="w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["frame", "x_min", "x_max", "y_min", "y_max", "out_w", "out_h"])  # шапка

            print(f"[INFO] Проецируем кадры и сохраняем в папку {PROJ_DIR}")
            for i, fname in enumerate(tqdm(frame_files, desc="Проекция")):
                frame = cv2.imread(os.path.join(FRAME_DIR, fname))
                proj, x_min, x_max, y_min, y_max, out_w, out_h = project_frame_pitch_only(
                    frame,
                    pitch_deg=log_int["pit_2"][i],
                    altitude=log_int["hgt"][i],
                    fov_h_deg=FOV_DEGREES,
                    resolution_m=PIXEL_SCALE,
                    hor=HORIZON_SHIFT,
                    grid_step=None,
                    debug_show=False
                )
                writer.writerow([i, x_min, x_max, y_min, y_max, out_w, out_h])
                out_path = os.path.join(PROJ_DIR, f"frame_{i:04d}_proj.png")
                cv2.imwrite(out_path, proj)

    # === 8. Расширение и сдвиг проекций
    print("[INFO] Расширяем и сдвигаем проекции...")
    if os.path.exists(SHIFT_DIR):
        shutil.rmtree(SHIFT_DIR)
    os.makedirs(SHIFT_DIR, exist_ok=True)
    x_coords_global, y_coords_global = rotate_and_shift_frames(
        projected_dir=PROJ_DIR, shifted_dir=SHIFT_DIR,
        lats=log_int["lat"], lons=log_int["lon"], yaws=log_int["yaw_2"],
        resolution_m=PIXEL_SCALE, debug=False
    )

    # === 9. (опционально) Сборка видео
    if SAVE_VIDEO:
        render_projected_video_with_grid(
            shifted_dir=SHIFT_DIR,
            x_coords=x_coords_global, y_coords=y_coords_global,  # np.abs(y_coords_global)[::-1],
            datetimes=frame_times,
            lats=log_int["lat"], lons=log_int["lon"], height=log_int["hgt"],
            pitches=log_int["pit_2"], rolls=log_int["rol_2"], yaws=log_int["yaw_2"],
            out_path=RUN_DIR / "mosaic_output.mp4", fps=fps / FRAME_INTERVAL, grid_step_m=1000, suffix="_proj_shft"
        )

    render_projected_image_with_grid(
        shifted_dir=SHIFT_DIR,
        x_coords=x_coords_global, y_coords=y_coords_global,
        out_path=RUN_DIR / "mosaic_output.png", grid_step_m=2000, suffix="_proj_shft"
    )


if __name__ == "__main__":
    main()
