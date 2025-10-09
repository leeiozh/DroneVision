# main_tracking.py

from tqdm import tqdm
import datetime as dt
from pathlib import Path
from config_tracking import *
from utils.tracking_utils import *
from utils.io_utils import parse_json, save_frames_from_video, get_meta_video


def setup_output_dirs(base_output="output"):
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_output) / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    # === 0. Настройка директорий ===
    RUN_DIR = setup_output_dirs()
    MASK_DIR = RUN_DIR / "masks"
    TRACK_DIR = RUN_DIR / "track"
    MASK_DIR.mkdir(exist_ok=True)
    TRACK_DIR.mkdir(exist_ok=True)

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

    # 5. Делаем трекинг льда
    tracker = Tracker(max_match_dist_m=MAX_MATCH_DIST_M, len_tr=LEN_TR)

    for idx in tqdm(range(len(frame_files)), desc="Трекинг"):
        frame = cv2.imread(os.path.join(FRAME_DIR, frame_files[idx]))

        if SEG_MODE == "hsv":  # построение маски
            mask = segment_ice_hsv(frame, s_range=HSV_S_RANGE, v_range=HSV_V_RANGE, morph_radius=MORPH_RADIUS)
        elif SEG_MODE == "gray":
            mask = segment_ice_gray(frame, blockSize=ADAPTIVE_BLOCK, C=ADAPTIVE_C, morph_radius=MORPH_RADIUS)
        else:
            raise ("SEG_MODE must be 'hsv' or 'gray'")

        objects = extract_objects_from_mask(mask, min_area_px=MIN_AREA_PX)  # извлечение объектов
        tracker.update(objects, idx, frame_times[idx], log_int, W_IMG, H_IMG, FOV_DEGREES)  # обновление трекера
        save_mask(idx, mask, objects, mask_dir=MASK_DIR)  # сохранение маски с положениями объектов
        plot_object_positions(tracker.tracks, frame_idx=idx, track_dir=TRACK_DIR)  # карта положений в земной СО

    # 6. Вычисляем скорости каждого объекта
    vel_map = tracker.compute_track_mean_velocities()
    plot_tracks_with_velocities(tracker.tracks, vel_map, RUN_DIR)

    # 7. Интерполируем поле скорости
    grid_x, grid_y, vx_i, vy_i, speed = plot_velocity_field_contour(tracker.tracks, vel_map, RUN_DIR)
    save_velocity_field_to_nc(grid_x, grid_y, vx_i, vy_i, speed, log_int["lat"][0], log_int["lon"][0], RUN_DIR)


if __name__ == "__main__":
    main()
