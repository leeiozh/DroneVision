# utils/io_utils.py

import os
import cv2
import json
import shutil
import subprocess
import numpy as np
from tqdm import tqdm
import datetime as dt
from scipy.signal import savgol_filter


def parse_json(file_path, stamp=True, delta=3):
    file = json.load(open(file_path))

    size = len(file["frames"])
    tim = np.zeros(size)
    lat = np.zeros(size)
    lon = np.zeros(size)
    hgt = np.zeros(size)
    pit = np.zeros(size)
    rol = np.zeros(size)
    yaw = np.zeros(size)
    pit_2 = np.zeros(size)
    rol_2 = np.zeros(size)
    yaw_2 = np.zeros(size)

    for i in range(size):
        if stamp:
            try:
                tim[i] = (dt.datetime.strptime(file["frames"][i]["custom"]["dateTime"],
                                               "%Y-%m-%dT%H:%M:%S.%fZ") + dt.timedelta(hours=delta)).timestamp()
            except ValueError:
                tim[i] = (dt.datetime.strptime(file["frames"][i]["custom"]["dateTime"],
                                               "%Y-%m-%dT%H:%M:%SZ") + dt.timedelta(hours=delta)).timestamp()
        lat[i] = file["frames"][i]["osd"]["latitude"]
        lon[i] = file["frames"][i]["osd"]["longitude"]
        hgt[i] = file["frames"][i]["osd"]["height"]
        pit[i] = file["frames"][i]["osd"]["pitch"]  # это углы дрона относительно земли
        rol[i] = file["frames"][i]["osd"]["roll"]  # это углы дрона относительно земли
        yaw[i] = file["frames"][i]["osd"]["yaw"]  # это углы дрона относительно земли
        pit_2[i] = file["frames"][i]["gimbal"]["pitch"]  # это углы подвеса камеры относительно ДРОНА
        rol_2[i] = file["frames"][i]["gimbal"]["roll"]  # это углы подвеса камеры относительно ДРОНА
        yaw_2[i] = file["frames"][i]["gimbal"]["yaw"]  # это углы подвеса камеры относительно ДРОНА

    hgt = savgol_filter(hgt, window_length=10, polyorder=3)
    lat = savgol_filter(lat, window_length=10, polyorder=3)
    lon = savgol_filter(lon, window_length=10, polyorder=3)

    # если нужно отрисовать логи
    # import matplotlib.pyplot as plt
    # # tim_2 = [dt.datetime.fromtimestamp(t) for t in tim]
    # ts = dt.datetime(2021, 8, 20, 20, 48, 21).timestamp()
    # tim2 = tim - ts
    # plt.plot(tim2, hgt)
    # plt.plot(tim2, yaw, label="yaw")
    # plt.plot(tim2, yaw_2, label="yaw2")
    # # plt.plot(tim, yaw + yaw_2)
    # # plt.plot(tim, pit + pit_2)
    # # plt.plot(tim, pit)
    # plt.plot(tim2, pit, label="pit")
    # plt.plot(tim2, pit_2, label="pit2")
    # plt.xlim(-20, 305)
    # plt.legend()
    # plt.show()

    return {"tim": tim, "lat": lat, "lon": lon, "hgt": hgt,
            "pit": pit, "rol": rol, "yaw": yaw,
            "pit_2": pit_2, "rol_2": rol_2, "yaw_2": yaw_2}


def save_frames_from_video(video_path, save_dir=None, frame_interval=10, start_frame=0, end_frame=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame if end_frame is not None else total_frames, total_frames)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    saved_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in tqdm(range(start_frame, end_frame), desc="Извлечение кадров"):
        ret, frame = cap.read()
        if not ret:
            break
        if (i - start_frame) % frame_interval == 0:
            save_path = os.path.join(save_dir, f"frame_{saved_count:04d}.png")
            cv2.imwrite(save_path, frame)
            saved_count += 1

    cap.release()
    print(f"\n[INFO] Извлечено {saved_count} кадров.")


def load_frames_from_folder(folder_path):
    frames = []
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.tif'))])
    for f in tqdm(frame_files, desc="Загрузка сохраненных кадров"):
        frame = cv2.imread(os.path.join(folder_path, f))
        frames.append(frame)
    print(f"Загружено {len(frames)} кадров из {folder_path}")
    return frames


def get_meta_video(video_path):
    command = ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
               "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start_time_str = result.stdout.strip()
    try:
        start_time = dt.datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        start_time = dt.datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%SZ")
    return start_time


def save_video_from_masks(mask_dir, out_path, fps=10):
    files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                    if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))])
    if not files:
        print("[WARN] Нет файлов масок для сборки видео.")
        return

    frame0 = cv2.imread(files[0])
    h, w = frame0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for fname in files:
        frame = cv2.imread(fname)
        if frame is None:
            continue
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
    print(f"[INFO] Видео из масок сохранено: {out_path}")
