from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def detect(cv2, frame, s_max=110, v_min=180, min_area=1000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] <= s_max) & (hsv[:, :, 2] >= v_min)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    objects = [(float(centroids[i, 0]), float(centroids[i, 1]), int(stats[i, cv2.CC_STAT_AREA]))
               for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return mask, objects


class Tracker:
    def __init__(self, max_distance=80.0, max_gap=2):
        self.max_distance, self.max_gap, self.next_id = max_distance, max_gap, 1
        self.tracks = []

    def update(self, frame_no, time_s, observations):
        active = [t for t in self.tracks if frame_no - t["last_frame"] <= self.max_gap and time_s - t["time"][-1] <= 10]
        pairs = []
        for oi, obs in enumerate(observations):
            for track in active:
                pairs.append((float(np.hypot(obs[0] - track["x"][-1], obs[1] - track["y"][-1])), oi, track))
        used_obs, used_tracks = set(), set()
        for distance, oi, track in sorted(pairs, key=lambda p: p[0]):
            if distance > self.max_distance or oi in used_obs or track["id"] in used_tracks:
                continue
            x, y, area = observations[oi]
            track["frame"].append(frame_no); track["time"].append(time_s); track["x"].append(x); track["y"].append(y); track["area"].append(area); track["last_frame"] = frame_no
            used_obs.add(oi); used_tracks.add(track["id"])
        for oi, (x, y, area) in enumerate(observations):
            if oi not in used_obs:
                self.tracks.append({"id": self.next_id, "frame": [frame_no], "time": [time_s], "x": [x], "y": [y], "area": [area], "last_frame": frame_no})
                self.next_id += 1

    def save(self, directory: Path):
        tracks_path, velocity_path = directory / "ice_tracks.csv", directory / "ice_velocities.csv"
        with tracks_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(["track_id", "frame", "time_utc_s", "east_m", "north_m", "area_px"])
            for t in self.tracks:
                writer.writerows(zip([t["id"]] * len(t["frame"]), t["frame"], t["time"], t["x"], t["y"], t["area"]))
        velocities = []
        for t in self.tracks:
            if len(t["time"]) < 3 or t["time"][-1] <= t["time"][0]: continue
            time = np.asarray(t["time"]) - t["time"][0]
            vx = float(np.polyfit(time, t["x"], 1)[0]); vy = float(np.polyfit(time, t["y"], 1)[0])
            residual = float(np.sqrt(np.mean((np.asarray(t["x"]) - (vx * time + t["x"][0])) ** 2 + (np.asarray(t["y"]) - (vy * time + t["y"][0])) ** 2)))
            velocities.append((t["id"], vx, vy, float(np.hypot(vx, vy)), len(time), residual))
        with velocity_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(["track_id", "u_east_m_s", "v_north_m_s", "speed_m_s", "observations", "fit_rmse_m"]); writer.writerows(velocities)
        return tracks_path, velocity_path, velocities
