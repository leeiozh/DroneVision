# main_nadir.py

from tqdm import tqdm
import datetime as dt
from config_nadir import *
from utils.drone_log_parser import parse_json
from utils.video_utils import save_frames_from_video, get_meta
from utils.tracker_utils import *


def main():
    # === 1. Считываем логи
    logs = parse_json(LOG_PATH, stamp=True, delta=DELTA_HOURS)

    # === 2. Определяем время старта видео, сравниваем с временем старта логов
    start_time = get_meta(VIDEO_PATH).timestamp()
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
        save_frames_from_video(VIDEO_PATH, save_dir=FRAME_DIR, frame_interval=FRAME_INTERVAL, format=FRAME_FORMAT,
                               start_frame=start_frame_idx, end_frame=end_frame_idx)

    frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(('.jpg', '.png'))])

    frame_times = (start_time + VIDEO_START_OFFSET_S) + np.arange(len(frame_files)) * FRAME_INTERVAL / fps
    log_int = {k: np.interp(frame_times, logs["tim"], logs[k]) for k in logs if k != "tim"}

    assert start_time + VIDEO_START_OFFSET_S + len(frame_files) * FRAME_INTERVAL / fps < logs["tim"][-1], \
        "[ERROR] Логи должны заканчиваться позднее видео! Проверьте FRAME_INTERVAL."

    # 5. Делаем трекинг льда
    tracker = Tracker()
    if os.path.exists(MASK_DIR):
        shutil.rmtree(MASK_DIR)
    if os.path.exists(TRACK_DIR):
        shutil.rmtree(TRACK_DIR)

    for idx in tqdm(range(len(frame_files)), desc="Трекинг"):
        frame = cv2.imread(os.path.join(FRAME_DIR, frame_files[idx]))

        if SEG_MODE == "hsv":  # построение маски
            mask = segment_ice_hsv(frame)
        elif SEG_MODE == "gray":
            mask = segment_ice_gray(frame)
        else:
            raise ("SEG_MODE must be 'hsv' or 'gray'")

        objects = extract_objects_from_mask(mask)  # извлечение объектов
        tracker.update(objects, idx, frame_times[idx], log_int)  # обновление трекера
        save_mask(idx, mask, objects)  # сохранение маски с положениями объектов
        plot_object_positions(tracker.tracks, frame_idx=idx)  # карта положений в земной СО

    # 6. Вычисляем скорости каждого объекта
    vel_map = tracker.compute_track_mean_velocities()
    plot_tracks_with_velocities(tracker.tracks, vel_map, SPEED_DIR)

    # 7. Интерполируем поле скорости
    grid_x, grid_y, vx_i, vy_i, speed = plot_velocity_field_contour(tracker.tracks, vel_map, SPEED_DIR)
    save_velocity_field_to_nc(grid_x, grid_y, vx_i, vy_i, speed, log_int["lat"][0], log_int["lon"][0], SPEED_DIR)


if __name__ == "__main__":
    main()
