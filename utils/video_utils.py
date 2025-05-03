# utils/video_utils.py

import cv2
import os
import shutil
import subprocess
from tqdm import tqdm
import datetime as dt
from config import FRAME_FORMAT


def save_frames_from_video(video_path, save_dir=None, frame_interval=10, format=FRAME_FORMAT,
                           start_frame=0, end_frame=None):
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
            save_path = os.path.join(save_dir, f"frame_{saved_count:04d}.{format}")
            cv2.imwrite(save_path, frame)
            saved_count += 1

    cap.release()
    print(f"[INFO] Извлечено {saved_count} кадров.")


def load_frames_from_folder(folder_path):
    """
    Загружает все кадры из указанной папки.

    Args:
        folder_path: путь к папке с изображениями

    Returns:
        frames: список numpy-кадров
    """
    frames = []
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.tif'))])
    for f in tqdm(frame_files, desc="Загрузка сохраненных кадров"):
        frame = cv2.imread(os.path.join(folder_path, f))
        frames.append(frame)
    print(f"Загружено {len(frames)} кадров из {folder_path}")
    return frames


def get_meta(video_path):
    command = ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
               "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start_time_str = result.stdout.strip()
    try:
        start_time = dt.datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        start_time = dt.datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%SZ")
    return start_time
