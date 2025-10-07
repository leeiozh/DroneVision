# utils/drone_log_parser.py

import json
import numpy as np
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
