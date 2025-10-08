# utils/tracker_utils.py
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from pyproj import Geod
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict
from scipy.interpolate import griddata

GEOD = Geod(ellps='WGS84')


def latlon2local(lat0, lon0, lat1, lon1):
    az, _, dist = GEOD.inv(lon0, lat0, lon1, lat1)
    return dist * np.sin(np.deg2rad(az)), dist * np.cos(np.deg2rad(az))


def local2latlon(east, north, lat0, lon0):
    az = np.rad2deg(np.arctan2(east, north))
    dist = np.hypot(east, north)
    az_flat = az.ravel()
    dist_flat = dist.ravel()
    lon, lat, _ = GEOD.fwd(np.full_like(az_flat, lon0), np.full_like(az_flat, lat0), az_flat, dist_flat)
    return lat.reshape(az.shape), lon.reshape(az.shape)


def segment_ice_hsv(frame, s_range, v_range, morph_radius):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    mask_sv = (s >= s_range[0]) & (s <= s_range[1]) & (v >= v_range[0]) & (v <= v_range[1])
    mask_uint8 = (mask_sv.astype(np.uint8)) * 255

    # строим элемент структурирования: эллипс радиуса morph_radius
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_radius + 1, 2 * morph_radius + 1))

    # закрытие (close) — сначала дилатация, затем эрозия: заполняет мелкие дыры
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, k)

    # открытие (open) — сначала эрозия, затем дилатация: удаляет маленькие шумные пятна
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, k)

    # возвращаем булевую маску (True там, где лёд)
    return (mask_opened > 0)


def segment_ice_gray(frame, blockSize, C, morph_radius):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # адаптивный порог: локальный порог по блоку blockSize, с постоянной C
    # возвращает изображение 0/255
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, blockSize, C)

    # структурный элемент для морфологии
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_radius + 1, 2 * morph_radius + 1))

    # закрываем и открываем для очистки
    th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th_opened = cv2.morphologyEx(th_closed, cv2.MORPH_OPEN, k)

    return (th_opened > 0)


def extract_objects_from_mask(mask, min_area_px):
    mask_u8 = mask.astype(np.uint8) * 255

    # находим компоненты связности (8-connectivity)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    objects = []
    # пропускаем label=0 (фон)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])  # площадь в пикселях
        if area < min_area_px:
            continue

        # левая верхняя точка bbox и размеры
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])

        # центр масс (cx, cy) в координатах изображения (x по ширине, y по высоте)
        cx, cy = float(centroids[lab, 0]), float(centroids[lab, 1])

        # маска (булева) конкретного объекта
        obj_mask = (labels_im == lab)

        objects.append({
            "label": int(lab),
            "area": area,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "mask": obj_mask
        })
    return objects


class Tracker:
    def __init__(self, max_match_dist_m, len_tr):
        self.tracks: List[Dict] = []  # список треков
        self.max_match_dist_m = max_match_dist_m  # порог для сопоставления (метры)
        self.len_tr = len_tr  # минимальное количество точек в треке для вычисления скорости

    def create_new_track(self, frame_idx: int, time_s: float, pixel_centroid: Tuple[float, float],
                         world_pos: Tuple[float, float], area_px: float):
        tr = {
            "frames": [frame_idx],  # индексы кадров
            "times": [time_s],  # времена (сек/epoch)
            "pixel_centroids": [pixel_centroid],  # пиксельные центроиды
            "world_positions": [world_pos],  # (east_m, north_m)
            "areas": [area_px],  # площади в px
            "id": len(self.tracks) + 1
        }
        self.tracks.append(tr)
        return tr

    def update(self, objects_pixel: List[Dict], frame_idx: int, frame_time_s: float,
               logs_interp: Dict[str, np.ndarray], w_img, h_img, fov):
        """
          - objects_pixel: список объектов (с centroid и area)
          - frame_idx, frame_time_s: индекс и время кадра
          - logs_interp: словарь с интерполированными логами на время кадра
        """

        coords = {"lat": logs_interp["lat"][frame_idx],
                  "lon": logs_interp["lon"][frame_idx],
                  "hgt": logs_interp["hgt"][frame_idx], }
        orients = {"yaw": np.deg2rad(logs_interp["yaw"][frame_idx]),
                   "pit": np.deg2rad(logs_interp["pit"][frame_idx]),
                   "rol": np.deg2rad(logs_interp["rol"][frame_idx]),
                   "yaw2": np.deg2rad(logs_interp["yaw_2"][frame_idx]),
                   "pit2": np.deg2rad(logs_interp["pit_2"][frame_idx] + 90),
                   "rol2": np.deg2rad(logs_interp["rol_2"][frame_idx]),
                   }

        # orients = {"yaw": 0,
        #            "pit": 0,
        #            "rol": 0,
        #            "yaw2": np.deg2rad(0),
        #            "pit2": np.deg2rad(0),
        #            "rol2": np.deg2rad(0),
        #            }

        R_total = build_total_rotation_matrix(orients)

        # --- 1) Переводим все пиксельные центроиды текущего кадра в мировые координаты
        current_world_positions = []
        for obj in objects_pixel:
            cx, cy = obj["centroid"]
            # print(*(f"{np.rad2deg(orients[o]):.0f}" for o in orients))
            # print("cxcy", cx, cy)
            east_m, north_m = pixel_to_world_ray_intersection(cx, cy, R_total, coords,
                                                              logs_interp["lat"][0], logs_interp["lon"][0],
                                                              w_img, h_img, fov)
            current_world_positions.append((east_m, north_m))

        # self._save_debug_positions(current_world_positions, frame_idx)
        # exit()

        # --- 2) Если треков ещё нет, создаём новые для всех текущих объектов
        if len(self.tracks) == 0:
            for obj, world_pos in zip(objects_pixel, current_world_positions):
                self.create_new_track(frame_idx, frame_time_s, obj["centroid"], world_pos, obj["area"])
            return

        # --- 3) Сопоставляем: строим KDTree из текущих позиций треков (их последних мировых позиций)
        active_positions = []
        active_track_indices = []
        for ti, tr in enumerate(self.tracks):
            # возьмём последнюю позицию трека
            last_pos = tr["world_positions"][-1]
            active_positions.append(last_pos)
            active_track_indices.append(ti)

        active_positions_np = np.array(active_positions) if len(active_positions) > 0 else np.empty((0, 2))
        curr_positions_np = np.array(current_world_positions) if len(current_world_positions) > 0 else np.empty(
            (0, 2))

        # если нет активных треков, создаём новые
        if active_positions_np.shape[0] == 0:
            for obj, world_pos in zip(objects_pixel, current_world_positions):
                self.create_new_track(frame_idx, frame_time_s, obj["centroid"], world_pos, obj["area"])
            return

        # KDTree для быстрого поиска ближайших треков к текущим позициям
        tree = cKDTree(active_positions_np)
        dists, idxs = tree.query(curr_positions_np, k=1)

        assigned_tracks = set()
        assigned_curr = set()

        for cur_i, (d, idx) in enumerate(zip(dists, idxs)):  # добавление к текущим трекам
            if d <= self.max_match_dist_m and idx not in assigned_tracks:
                ti = active_track_indices[int(idx)]
                tr = self.tracks[ti]
                tr["frames"].append(frame_idx)
                tr["times"].append(frame_time_s)
                tr["pixel_centroids"].append(objects_pixel[cur_i]["centroid"])
                tr["world_positions"].append(current_world_positions[cur_i])
                tr["areas"].append(objects_pixel[cur_i]["area"])
                assigned_tracks.add(idx)
                assigned_curr.add(cur_i)

        for cur_i, obj in enumerate(objects_pixel):  # создание новых треков
            if cur_i in assigned_curr:
                continue
            world_pos = current_world_positions[cur_i]
            self.create_new_track(frame_idx, frame_time_s, obj["centroid"], world_pos, obj["area"])

    def _save_debug_positions(self, positions: List[Tuple[float, float]], frame_idx: int):

        if len(positions) == 0:
            return

        positions_np = np.array(positions)
        east, north = positions_np[:, 0], positions_np[:, 1]

        plt.figure(figsize=(6, 6))
        plt.scatter(east, north, c='blue', s=20, label="Centroids")
        plt.scatter(0, 0)
        plt.xlabel("East, m")
        plt.ylabel("North, m")
        plt.title(f"Frame {frame_idx}")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')

        plt.savefig(f"output/debug/frame_{frame_idx:04d}.png", dpi=150)
        plt.close()

    def compute_track_mean_velocities(self) -> Dict[int, Tuple[float, float, float]]:
        """
        Для каждого трека возвращаем среднюю скорость (vx_east, vy_north) в м/с и среднюю площадь в px.
        Возвращаем dict: track_id -> (vx, vy, mean_area)
        """
        res = {}
        for tr in self.tracks:
            if len(tr["times"]) < self.len_tr:
                continue
            # используем первым и последним наблюдением (можно улучшить LS)
            t0 = tr["times"][0]
            t1 = tr["times"][-1]
            if t1 <= t0:
                continue
            x0, y0 = tr["world_positions"][0]
            x1, y1 = tr["world_positions"][-1]
            vx = (x1 - x0) / (t1 - t0)
            vy = (y1 - y0) / (t1 - t0)
            mean_area = float(np.mean(tr["areas"]))
            tid = tr["id"]
            res[tid] = (vx, vy, mean_area)
        return res

    def draw_velocities_on_image(self, image_bgr: np.ndarray, vel_map: Dict[int, Tuple[float, float, float]],
                                 scale_px_per_m: float = 0.5):
        """
        Draw velocity vectors on provided image and save to out_path.
        scale_px_per_m: how many pixels correspond to 1 meter in visualization (tune for readability).
        """
        im = image_bgr.copy()
        for tr in self.tracks:
            tid = tr["id"]
            if tid not in vel_map:
                continue
            vx, vy, area = vel_map[tid]
            # representative point on image: first pixel centroid
            px, py = tr["pixel_centroids"][0]
            px_i, py_i = int(round(px)), int(round(py))
            # convert m/s -> pixels for visualization (arbitrary dt_vis=1s)
            dx_px = int(round(vx * scale_px_per_m))
            dy_px = int(round(-vy * scale_px_per_m))  # north -> up in image y
            pt1 = (px_i, py_i)
            pt2 = (px_i + dx_px, py_i + dy_px)
            cv2.arrowedLine(im, pt1, pt2, (0, 0, 255), 2, tipLength=0.25)
            cv2.putText(im, f"ID{tid}", (px_i + 3, py_i - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return im


def rotation_matrix_from_yaw_pitch_roll(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    """
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    yaw: rotation about Z (heading), pitch: about Y, roll: about X
    Возвращает 3x3 матрицу.
    """
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    Rz = np.array([[cy, -sy, 0.0],
                   [sy, cy, 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[cr, 0, sr],
                   [0, 1, 0],
                   [-sr, 0, cr]])
    Rx = np.array([[1, 0, 0],
                   [0, cp, -sp],
                   [0, sp, cp]])
    return Rz @ Rx @ Ry


def build_total_rotation_matrix(orients):
    R_osd = rotation_matrix_from_yaw_pitch_roll(orients["yaw"], orients["pit"], orients["rol"])
    R_gim = rotation_matrix_from_yaw_pitch_roll(orients["yaw2"], orients["pit2"], orients["rol2"])
    return R_osd @ R_gim


# --------------------------- Pixel -> World (ray-plane) ---------------------
def pixel_to_world_ray_intersection(u, v, R_total, coords,
                                    lat0, lon0, img_w, img_h, fov_x_deg):
    # --- 1) focal length in pixels
    f_px = (img_w / 2.0) / np.tan(np.deg2rad(fov_x_deg) / 2.0)

    # --- 2) principal point (image center)
    cx = img_w / 2.0
    cy = img_h / 2.0

    # pixel coords relative to center direction in camera coordinates (normalized)
    d_cam = np.array([(u - cx) / f_px, -(v - cy) / f_px, -1.0])
    d_cam /= np.linalg.norm(d_cam)  # normalize

    # --- 3) rotation: build R_world_cam from logs for this frame
    d_world = R_total @ d_cam

    # --- 5) camera position in ENU relative to lat0/lon0
    east_cam, north_cam = latlon2local(lat0, lon0, coords["lat"], coords["lon"])
    if np.isnan(east_cam) or np.isnan(north_cam):
        east_cam, north_cam = 0, 0
    cam_pos = np.array([east_cam, north_cam, coords["hgt"]], dtype=float)

    # --- 6) intersect ray with ground plane z=0
    dz = d_world[2]
    if np.abs(dz) < 1e-6:
        # ray is (almost) parallel to ground plane — fallback: return NaNs
        return np.nan, np.nan
    t = -coords["hgt"] / dz
    intersect = cam_pos + t * d_world  # [east, north, up(=0)]
    # print("d_cam", d_cam)
    # print("d_world", d_world)
    # print(intersect)
    # print("\n###########################\n")

    return intersect[0], intersect[1]


def plot_object_positions(tracks, frame_idx, track_dir):
    os.makedirs(track_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(0, 0, c="k")
    for tr in tracks:
        if frame_idx in tr["frames"]:
            i = tr["frames"].index(frame_idx)
            x, y = tr["world_positions"][i]
            plt.scatter(x, y, s=30, label=f"ID {tr['id']}")
            plt.text(x + 2, y + 2, str(tr["id"]), fontsize=8)
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(True)
    # plt.axis("equal")
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.tight_layout()
    out_path = os.path.join(track_dir, f"pos_frame_{frame_idx:04d}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_tracks_with_velocities(tracks, vel_map, out_dir):
    """Рисует все треки объектов и средние векторы скоростей"""
    plt.figure(figsize=(6, 6))
    plt.scatter(0, 0, c="k")
    for tr in tracks:
        pos = np.array(tr["world_positions"])
        plt.plot(pos[:, 0], pos[:, 1], "o", ms=3, label=f"ID {tr['id']}")
        plt.text(pos[0, 0], pos[0, 1], str(tr["id"]), fontsize=8, color='k')
        if tr["id"] in vel_map:
            vx, vy, _ = vel_map[tr["id"]]
            plt.arrow(pos[0, 0], pos[0, 1], vx * 100, vy * 100, color="k", alpha=0.5,
                      head_width=5, length_includes_head=True, zorder=10)
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "track.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Треки и скорости сохранены в {out_dir}")


def plot_velocity_field_contour(tracks, vel_map, out_dir, nx=100, ny=100):
    """Интерполирует и рисует поле скоростей по данным треков"""
    # --- 1. Собираем центры и скорости
    points, vx_list, vy_list = [], [], []

    for tr in tracks:
        tid = tr["id"]
        if tid not in vel_map:
            continue
        pos = np.array(tr["world_positions"])
        if len(pos) == 0:
            continue

        mean_pos = np.nanmean(pos, axis=0)
        vx, vy, _ = vel_map[tid]
        points.append(mean_pos)
        vx_list.append(vx)
        vy_list.append(vy)

    if len(points) < 3:
        print("[WARN] Недостаточно точек для интерполяции поля скоростей")
        return

    points = np.array(points)
    vx_list = np.array(vx_list)
    vy_list = np.array(vy_list)

    # --- 2. Интерполяция на регулярную сетку
    grid_x, grid_y = np.mgrid[
                     np.min(points[:, 0]):np.max(points[:, 0]):complex(nx),
                     np.min(points[:, 1]):np.max(points[:, 1]):complex(ny)
                     ]

    vx_i = griddata(points, vx_list, (grid_x, grid_y), method='linear')
    vy_i = griddata(points, vy_list, (grid_x, grid_y), method='linear')

    # --- 3. Полная скорость
    speed = np.sqrt(vx_i ** 2 + vy_i ** 2) * 100

    # --- 4. Контурплот
    plt.figure(figsize=(7, 6))
    plt.contourf(grid_x, grid_y, speed, levels=20, cmap="viridis")
    plt.colorbar(label="Speed (cm/s)")
    plt.quiver(points[:, 0], points[:, 1], vx_list * 100, vy_list * 100,
               color="k", scale=1, scale_units="xy", width=0.002)
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.axis("equal")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "contourplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Карта скоростей сохранена в {out_dir}")

    return grid_x, grid_y, vx_i, vy_i, speed


def save_velocity_field_to_nc(grid_x, grid_y, vx_i, vy_i, speed, lat0, lon0, out_dir):
    """
    Сохраняет интерполированное поле скоростей в NetCDF,
    используя географические координаты (lat, lon), вычисленные по lat0/lon0.
    """

    nc = Dataset(os.path.join(out_dir, "speed.nc"), "w", format="NETCDF4")
    nc.createDimension("x", grid_x.shape[0])
    nc.createDimension("y", grid_x.shape[1])
    lat_var = nc.createVariable("lat", "f4", ("x", "y"))
    lon_var = nc.createVariable("lon", "f4", ("x", "y"))
    vx_var = nc.createVariable("u", "f4", ("x", "y"))
    vy_var = nc.createVariable("v", "f4", ("x", "y"))
    spd_var = nc.createVariable("speed", "f4", ("x", "y"))

    lat_grid, lon_grid = local2latlon(grid_x, grid_y, lat0, lon0)
    lat_var[:, :] = lat_grid
    lon_var[:, :] = lon_grid
    vx_var[:, :] = vx_i
    vy_var[:, :] = vy_i
    spd_var[:, :] = speed

    nc.description = "Interpolated surface velocity field (from drift tracking)"
    nc.close()

    print(f"[INFO] Поле скорости сохранено в {out_dir}")


def save_mask(idx, mask, objects, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    mask_gray = (mask * 255).astype(np.uint8)
    mask_vis = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    mask_vis[mask_gray > 0] = (255, 255, 255)

    for obj in objects:
        x, y, w, h = obj["bbox"]
        cx, cy = obj["centroid"]
        obj_id = obj["label"]
        cv2.rectangle(mask_vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.circle(mask_vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(mask_vis, f"{obj_id}", (int(cx + 5), int(cy - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

    out_path = os.path.join(mask_dir, f"mask_{idx:04d}.png")
    cv2.imwrite(out_path, mask_vis)
