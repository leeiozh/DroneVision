# видео обработки
VIDEO_PATH = '/media/leeiozh/KINGSTON/DRONE/PU2023-drone/24_07_nadir/DJI_0118.MP4'

# логи, время которых покрывает время видео
LOG_PATH = '/media/leeiozh/KINGSTON/DRONE/PU2023-drone/logs/20230724T094438.json'

# разница во времени между файлом логов и видео
DELTA_HOURS = 3 - 0.95 / 3600

# принудительно заново извлекать кадры из видео
FORCE_EXTRACT = True

# интервал для извлечения кадров (1 - каждый кадр, 30 - каждый 30ый)
FRAME_INTERVAL = 30

FOV_DEGREES = 78.8  # угол обзора камеры
W_IMG = 3840
H_IMG = 2160

VIDEO_START_OFFSET_S = 0  # время в секундах от начала видео
VIDEO_END_OFFSET_S = 32  # время конца обрезки от начала видео

TILE_SIZE = 300
