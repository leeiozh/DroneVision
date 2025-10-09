# utils/spectral_utils.py
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import numpy as np
from pyproj import Geod
from netCDF4 import Dataset
from scipy.signal import welch
from scipy.interpolate import griddata


def omega_th(kx, ky, g=9.81):
    k = np.sqrt(kx ** 2 + ky ** 2)
    return np.sqrt(g * k)


def estimate_resolution(width, height, fov):
    return height * np.tan(fov * np.pi / 360) / width


def compute_current(tile_series, f_max, k_max):
    _, back_f3 = welch(np.fft.fftshift(np.fft.fft2(tile_series, axes=(1, 2))), axis=0, nperseg=tile_series.shape[0],
                       return_onesided=False)
    HALF = back_f3.shape[1] // 2
    back_f3[:, :, HALF - 1:HALF + 2] = 0
    back_f3[:, HALF - 1:HALF + 2, :] = 0

    oms = np.linspace(0, 2 * np.pi * f_max, back_f3.shape[0])
    kx = np.linspace(-k_max, k_max, back_f3.shape[1])
    ky = np.linspace(-k_max, k_max, back_f3.shape[2])

    # omega_max = np.zeros((len(kx), len(ky)))

    omega_max = oms[np.argmax(back_f3, axis=0).astype(int)]
    mask = (np.log(np.max(back_f3, axis=0)) < 16)
    omega_max[mask] = np.nan

    # for i, kxi in enumerate(kx):
    #     for j, kyi in enumerate(ky):
    #         sp_slice = back_f3[:, i, j]
    #         sp_slice[:5] = 0
    #         idx_max = np.argmax(sp_slice)
    #         omega_max[i, j] = idx_max#oms[idx_max]
    # cc = plt.imshow(omega_max)
    # cb = plt.colorbar(cc)
    # plt.show()
    A = []
    b = []
    for i, kxi in enumerate(kx):
        for j, kyi in enumerate(ky):
            if not np.isnan(omega_max[i, j]):
                omega_diff = omega_max[i, j] - omega_th(kxi, kyi)
                A.append([kxi, kyi])
                b.append(omega_diff)
    A = np.array(A)
    b = np.array(b)

    u_hat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    ux, uy = u_hat
    # FOUR_TIME = back_f3.shape[0]

    # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # port = np.zeros(shape=(FOUR_TIME, back_f3.shape[1] // 2))
    # for ii in range(0, back_f3.shape[1] // 2):
    #     for jj in range(0, back_f3.shape[1] // 2):
    #         k = round(np.sqrt(ii * ii + jj * jj))
    #         if ii + HALF < back_f3.shape[1] and jj + HALF < back_f3.shape[2] and k < back_f3.shape[1] // 2:
    #             port[:, k] += back_f3[:, ii + HALF, jj + HALF] #  np.sign(ii) *
    # port[0, 0] = 0.
    # ax[1].imshow(port, origin="lower", aspect="auto", extent=[0, k_max / 2, 0, f_max])
    # kkk = np.linspace(0, k_max, 100)
    # ax[1].plot(kkk, np.sqrt(9.81 * kkk) / 2 / np.pi, c="w", label=r"$\omega = \sqrt{gk}$")
    # ax[1].set_xlim(0, k_max / 2)
    # ax[1].set_ylim(0, f_max)
    # ax[1].set_xlabel("k, rad/m")
    # ax[1].set_ylabel(r"$\omega$, Hz")
    # ax[0].set_xlabel("x, px")
    # ax[0].set_ylabel("y, px")
    # ax[1].legend(loc="lower right")
    # ax[0].imshow(tile_series[0], origin="lower", aspect="auto", cmap="gray")
    # plt.title(f"ux = {ux * 100:.0f} см/с, uy = {uy * 100:.0f} см/с")
    # plt.savefig(f"output/tmp{ij}.png", bbox_inches="tight", dpi=300)

    return ux, uy


def local2latlon(lat0, lon0, x, y):
    geod = Geod(ellps='WGS84')
    az = np.degrees(np.arctan2(x, y))
    dist = np.sqrt(x ** 2 + y ** 2)
    lon, lat, _ = geod.fwd(np.full_like(dist, lon0), np.full_like(dist, lat0), az, dist)
    return lat, lon


def plot_and_interpolate_velocity_field(coords, speeds, time, drone_v, yaw_deg, out_dir, nx=100, ny=100):
    coords = coords.reshape(-1, 2)
    speeds = speeds.reshape(-1, 2)

    # --- Поворот на yaw
    yaw_rad = np.deg2rad(yaw_deg)
    rot = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                    [np.sin(yaw_rad), np.cos(yaw_rad)]])
    speeds_rot = speeds @ rot.T
    coords_rot = coords @ rot.T

    # --- Вычитание скорости дрона
    speeds_corr = (speeds_rot - drone_v) * 100
    coords_corr = coords_rot - np.array([drone_v[0] * time, drone_v[1] * time])

    vx, vy = speeds_corr[:, 0], speeds_corr[:, 1]

    # --- Интерполяция
    grid_x, grid_y = np.mgrid[
                     np.min(coords_corr[:, 0]):np.max(coords_corr[:, 0]):complex(nx),
                     np.min(coords_corr[:, 1]):np.max(coords_corr[:, 1]):complex(ny)
                     ]
    vx_i = griddata(coords_corr, vx, (grid_x, grid_y), method='linear')
    vy_i = griddata(coords_corr, vy, (grid_x, grid_y), method='linear')
    spd = np.sqrt(vx_i ** 2 + vy_i ** 2)

    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, spd, levels=20, cmap="viridis")
    plt.colorbar(label="Speed (cm/s)")
    plt.quiver(coords_corr[:, 0], coords_corr[:, 1], vx, vy, color="k", scale=1, scale_units="xy", width=0.002)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Corrected current velocity field")
    plt.axis("equal")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "currents.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Поле скоростей сохранено в {out_path}")

    return grid_x, grid_y, vx_i, vy_i, spd


def save_velocity_field_to_nc(grid_x, grid_y, vx_i, vy_i, spd, lat0, lon0, out_dir):
    lat_grid, lon_grid = local2latlon(lat0, lon0, grid_x, grid_y)
    nc_path = os.path.join(out_dir, "currents.nc")

    nc = Dataset(nc_path, "w", format="NETCDF4")
    nc.createDimension("x", grid_x.shape[0])
    nc.createDimension("y", grid_x.shape[1])

    lat_var = nc.createVariable("lat", "f4", ("x", "y"))
    lon_var = nc.createVariable("lon", "f4", ("x", "y"))
    u_var = nc.createVariable("u", "f4", ("x", "y"))
    v_var = nc.createVariable("v", "f4", ("x", "y"))
    spd_var = nc.createVariable("speed", "f4", ("x", "y"))

    lat_var[:, :] = lat_grid
    lon_var[:, :] = lon_grid
    u_var[:, :] = vx_i
    v_var[:, :] = vy_i
    spd_var[:, :] = spd

    nc.description = "Interpolated surface current velocity field (motion-compensated)"
    nc.close()
    print(f"[INFO] NetCDF сохранён: {nc_path}")
