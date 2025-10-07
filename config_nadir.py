# видео обработки
VIDEO_PATH = '/media/leeiozh/KINGSTON/DRONE/PU2023-drone/24_07_nadir/DJI_0365.MP4'

# логи, время которых покрывает время видео
LOG_PATH = '/media/leeiozh/KINGSTON/DRONE/PU2023-drone/logs/20230724T170938.json'

# разница во времени между файлом логов и видео
DELTA_HOURS = 3 - 0.95 / 3600

# в каком формате сохранять промежуточные кадры
FRAME_FORMAT = "png"

# куда сохранять извлеченные кадры
FRAME_DIR = 'output/frames/'

# куда сохранять маски
MASK_DIR = 'output/masks/'

# куда сохранять объекты и треки
TRACK_DIR = 'output/tracks/'

# куда сохранять скорости
SPEED_DIR = 'output/speed/'

# принудительно заново извлекать кадры из видео
FORCE_EXTRACT = True

# интервал для извлечения кадров (1 - каждый кадр, 30 - каждый 30ый)
FRAME_INTERVAL = 30 * 10

FOV_DEGREES = 78.8  # угол обзора камеры
W_IMG = 3840
H_IMG = 2160

VIDEO_START_OFFSET_S = 0  # время в секундах от начала видео
VIDEO_END_OFFSET_S = 229  # время конца обрезки от начала видео

SEG_MODE = "hsv"  # 'hsv' or 'gray' or 'both' (both сохранит обе маски, в обработке можно выбрать)
HSV_S_RANGE = (0, 110)  # S channel thresholds for ice (inclusive)
HSV_V_RANGE = (180, 255)  # V channel thresholds for ice
ADAPTIVE_BLOCK = 101  # параметр в segment_ice_gray
ADAPTIVE_C = 10  # параметр в segment_ice_gray
MORPH_RADIUS = 4  # параметр в segment_ice_gray
MIN_AREA_PX = 1000  # минимальная площадь объекта в пикселях (отброс мелочи)
MAX_MATCH_DIST_M = 80.0  # макс расстояние для сопоставления центроидов между кадрами
LEN_TR = 10  # минимальная длина трека для вычисления скорости
