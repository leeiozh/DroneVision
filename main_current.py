# main_current.py
import numpy as np
from tqdm import tqdm
import datetime as dt
from pyproj import Geod
from pathlib import Path
from config_current import *
from utils.tracking_utils import *
from utils.io_utils import parse_json, save_frames_from_video, get_meta_video
from utils.spectral_utils import compute_current, estimate_resolution, \
    plot_and_interpolate_velocity_field, save_velocity_field_to_nc


def setup_output_dirs(base_output="output"):
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_output) / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def calc_drone_speed(lat, lon, dt_s):
    geod = Geod(ellps='WGS84')
    az, _, dist = geod.inv(lon[0], lat[0], lon[-1], lat[-1])
    return [dist * np.sin(np.deg2rad(az)) / dt_s, dist * np.cos(np.deg2rad(az)) / dt_s]


def main():
    # === 0. Настройка директорий ===
    RUN_DIR = setup_output_dirs()
    FRAME_DIR = RUN_DIR / "frames"
    FRAME_DIR.mkdir(exist_ok=True)

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
    duration_s = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    cap.release()
    print(f"[INFO] FPS видео = {fps:.2f}, длительность = {duration_s:.2f}")

    assert 0 <= VIDEO_START_OFFSET_S < VIDEO_END_OFFSET_S <= duration_s, \
        "[ERROR] Неверные границы обрезки видео!"

    start_frame_idx = int(VIDEO_START_OFFSET_S * fps)
    end_frame_idx = int(VIDEO_END_OFFSET_S * fps)

    # 4. Загружаем кадры
    os.makedirs(FRAME_DIR, exist_ok=True)
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

    # 5. Вычисление скоростей
    frames = np.zeros(shape=(len(frame_times), H_IMG, W_IMG))
    for idx in tqdm(range(len(frame_files)), desc="Подгрузка кадров"):
        frames[idx] = cv2.imread(os.path.join(FRAME_DIR, frame_files[idx]), cv2.IMREAD_GRAYSCALE)

    n_frames, h, w = frames.shape
    n_i = h // TILE_SIZE
    n_j = w // TILE_SIZE

    coords = np.zeros(shape=(n_i, n_j, 2))
    speeds = np.zeros(shape=(n_i, n_j, 2))

    for i in range(n_i):
        for j in range(n_j):
            resol = 4 * estimate_resolution(W_IMG, np.mean(log_int["hgt"]), FOV_DEGREES)
            x0, x1 = i * TILE_SIZE, (i + 1) * TILE_SIZE
            y0, y1 = j * TILE_SIZE, (j + 1) * TILE_SIZE
            coords[i, j] = [0.5 * (x0 + x1) * resol, 0.5 * (y0 + y1) * resol]
            speeds[i, j, 0], speeds[i, j, 1] = compute_current(frames[:, x0:x1, y0:y1],
                                                               f_max=fps / FRAME_INTERVAL,
                                                               k_max=2 * np.pi / resol)

    drone_speed = calc_drone_speed(log_int["lat"], log_int["lon"], dt_s=FRAME_INTERVAL / fps * len(frame_times))
    print(f"[INFO] Средняя скорость дрона = {drone_speed[0]:.2f}, {drone_speed[1]:.2f}")

    # === 6. Построение поля ===
    grid_x, grid_y, vx_i, vy_i, spd = plot_and_interpolate_velocity_field(
        coords, speeds, drone_speed[0], drone_speed[1], np.mean(log_int["yaw"] + log_int["yaw_2"]), RUN_DIR)

    # === 7. Сохранение в NetCDF ===
    save_velocity_field_to_nc(grid_x, grid_y, vx_i, vy_i, spd, log_int["lat"][0], log_int["lon"][0], RUN_DIR)


if __name__ == "__main__":
    main()
